"""Per-pick yaw correction from trajectory consistency, and a diagnosis of the ~1 deg per-pick yaw bias.

Pass 1: single-frame predictions for every frame of a pick (P, range, angles, class) and the voted pose (R_k, c_k).
Fit: a single yaw offset eps about the pallet y axis applied to every frame's rotation, R_k(eps) = Ry(eps) R_k, chosen on a grid
to minimise the inconsistency between consecutive re-voted camera centres and the known inter-frame motion:
    J(eps) = sum_k | c_{k+1}(eps) - propagate(c_k(eps), R_{k+1}(eps), dT_k) |^2
Pass 2: re-vote with the corrected rotations. Reports origin error before/after per bin, and compares the fitted eps with the
GT-measured per-pick yaw bias: agreement => the bias is in the model and observable from the sequence; eps ~ 0 while the GT bias
is ~1 deg => the labels' heading, not the model, is what disagrees with the trajectory.

python -m pallet.yawfit --run runs/b256_run6 --split runs/fixed_val --picks 30 --out runs/yawfit_run6
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import targets as T
from .dataset import PalletDataset
from .metrics import aggregate, format_table
from .model import PalletNet
from .pose import estimate_pose, rotation_error_deg
from .sequence import relative_motion


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def yaw_about_y(dR):
    """Signed rotation angle of dR about the y axis (deg), from its rotation vector."""
    ang = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
    axis = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
    return float(np.degrees(ang) * axis[1] / max(np.linalg.norm(axis), 1e-12))


@torch.no_grad()
def predict_pick(net, ds, idx, dev):
    frames = []
    for j in idx:
        s = ds[j]; H, W = s["cls"].shape; fx, fy, cx, cy = s["intrinsics"].tolist(); real = s["pixel_mask"].numpy()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x, logits = net.generate(s["image"][None].to(dev), steps=1)
        dec = T.decode(x[0].cpu().numpy(), np.ones((6, H, W), bool)); cls_pred = logits[0].argmax(0).cpu().numpy()
        rays = T.pixel_rays(fx, fy, cx, cy, H, W)
        vote = np.isin(cls_pred, (T.CLS_PALLET, T.CLS_LOAD)) & real
        est = estimate_pose(dec["P"], dec["range"], dec["theta"], dec["phi"], rays, vote, real)
        frames.append({"P": dec["P"][:, vote].T, "r": dec["range"][vote], "d": rays[vote], "R": est["R"], "c": est["c"],
                       "pose_gt": s["pallet_P_cam"].numpy().astype(np.float64), "face": float(s["face_dist_m"]), "key": s["key"], "scan": int(s["scan"])})
    return frames


def revote(f, R):
    v = f["P"] - f["r"][:, None] * (f["d"] @ R.T)
    return np.median(v, axis=0)


def fit_yaw(frames, grid_deg=np.arange(-3.0, 3.01, 0.1), max_face=8.0, objective="trajectory"):
    """Grid search of the yaw offset over frames within max_face metres. objective:
    "trajectory": inconsistency between consecutive re-voted centres and the odometry (INVARIANT to a global yaw: a yaw rotates
    the voted and the propagated trajectory together -> uninformative, kept as the negative result);
    "spread": pooled per-frame vote dispersion (median |vote - median|), which grows when the offsets r R d are rotated wrongly
    relative to the pallet-frame P of the same pixels."""
    use = [i for i, f in enumerate(frames) if f["face"] < max_face and np.all(np.isfinite(f["c"]))]
    if len(use) < 4:
        return 0.0, None
    best = (np.inf, 0.0); J = []
    for e in grid_deg:
        Re = Ry(np.radians(e))
        if objective == "spread":
            res = 0.0
            for i in use:
                v = frames[i]["P"] - frames[i]["r"][:, None] * (frames[i]["d"] @ (Re @ frames[i]["R"]).T)
                res += float(np.median(np.linalg.norm(v - np.median(v, axis=0), axis=1)))
            J.append(res)
            if res < best[0]:
                best = (res, float(e))
            continue
        cs = {i: revote(frames[i], Re @ frames[i]["R"]) for i in use}
        res = 0.0
        for a, b in zip(use[:-1], use[1:]):
            dT = relative_motion(frames[a]["pose_gt"], frames[b]["pose_gt"]); A, bb = dT[:3, :3], dT[:3, 3]
            Rb = Re @ frames[b]["R"]
            c_prop = cs[a] - Rb @ bb                       # c_{k+1} = c_k - R_{k+1} b  (R_{k+1} = R_k A^T, consistent to first order)
            res += float(np.sum((cs[b] - c_prop) ** 2))
        J.append(res)
        if res < best[0]:
            best = (res, float(e))
    return best[1], np.array(J)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/b256_run6"); ap.add_argument("--split", default="runs/fixed_val")
    ap.add_argument("--picks", type=int, default=30); ap.add_argument("--out", default="runs/yawfit")
    ap.add_argument("--objective", default="trajectory", choices=["trajectory", "spread"])
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    args = json.load(open(Path(a.run) / "args.json"))
    df = pd.read_csv(Path(a.split) / "split.csv", dtype={"robot": str}); df = df[df.split == "val"]
    picks = df.drop_duplicates(["robot", "pick"])[["robot", "pick"]].values.tolist()
    sel = {tuple(picks[i]) for i in np.linspace(0, len(picks) - 1, min(a.picks, len(picks))).round().astype(int)}
    df = df[[(r, p) in sel for r, p in zip(df.robot, df.pick)]].sort_values(["robot", "pick", "scan"]).reset_index(drop=True)
    ds = PalletDataset(df, max_range_m=args.get("max_range_m"))
    ck = torch.load(Path(a.run) / "checkpoint-last.pth", map_location="cpu", weights_only=False)
    net = PalletNet(model=args["model"], feature_embedding_type=args["feature_embedding_type"]); net.load_state_dict(ck["ema"]); net = net.cuda().eval()
    rows_before, rows_after, fits = [], [], []
    for _, g in df.groupby(["robot", "pick"], sort=False):
        frames = predict_pick(net, ds, list(g.index), torch.device("cuda"))
        eps, J = fit_yaw(frames, objective=a.objective)
        gt_bias = np.mean([yaw_about_y(f["R"] @ f["pose_gt"][:9].reshape(3, 3).T) for f in frames if f["face"] < 8 and np.all(np.isfinite(f["c"]))])
        fits.append({"key": frames[0]["key"], "eps_fit_deg": eps, "gt_yaw_bias_deg": float(gt_bias), "n": len(frames)})
        for f in frames:
            R_gt, c_gt = f["pose_gt"][:9].reshape(3, 3), f["pose_gt"][9:]
            base = {"key": f["key"], "scan": f["scan"], "face_dist_m": f["face"]}
            rows_before.append({**base, "origin_err_cm": float(np.linalg.norm(f["c"] - c_gt) * 100), "rot_err_deg": rotation_error_deg(f["R"], R_gt)})
            Rc = Ry(np.radians(eps)) @ f["R"]
            rows_after.append({**base, "origin_err_cm": float(np.linalg.norm(revote(f, Rc) - c_gt) * 100), "rot_err_deg": rotation_error_deg(Rc, R_gt)})
    fits = pd.DataFrame(fits); fits.to_csv(out / "fits.csv", index=False)
    print(fits.to_string(index=False))
    r = np.corrcoef(fits.eps_fit_deg, fits.gt_yaw_bias_deg)[0, 1]
    print(f"\ncorrelation(fitted eps, GT-measured yaw bias) = {r:.2f}; sign convention: eps corrects R_est, so eps ~ -bias if the bias is model-side")
    keys = ("origin_err_cm", "rot_err_deg")
    def agg(rows):
        d = np.array([x["face_dist_m"] for x in rows]); bins = [(0, 2), (2, 3), (3, 4.5), (4.5, 6), (6, 8), (8, 12), (0, 99)]
        return {f"{lo}-{hi}": {k: float(np.median([x[k] for x, f in zip(rows, d) if lo <= f < hi])) for k in keys} for lo, hi in bins if any((d >= lo) & (d < hi))}
    A, B = agg(rows_before), agg(rows_after)
    print("\nbin        origin before -> after (cm)   rot before -> after (deg)")
    for b in A: print(f"{b:10s} {A[b]['origin_err_cm']:6.1f} -> {B[b]['origin_err_cm']:6.1f}          {A[b]['rot_err_deg']:.2f} -> {B[b]['rot_err_deg']:.2f}")
    json.dump({"before": rows_before, "after": rows_after}, open(out / "rows.json", "w"))


if __name__ == "__main__":
    main()
