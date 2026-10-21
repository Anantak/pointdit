"""Sequential estimation along an approach: the previous frame's dense estimate, carried to the new view by the known camera
motion, becomes a per-pixel prior from which the flow model is warm-started (PalletNet.generate(..., x_prior, t0)).

Prior for frame k+1 from frame k (all in the pallet frame, which is static):
  * pose: (R_k, c_k) propagated by the inter-frame motion dT (X_{k+1} = A X_k + b): R' = R_k A^T, c' = c_k - R' b
  * theta/phi: exact from the propagated pose for every pixel (pure ray geometry)
  * P/range: frame k's hit pixels (P in the pallet frame does not change) re-rendered into frame k+1 with a z-buffer splat,
    plus the ground plane y = 0 analytically where nothing nearer was rendered; unknown elsewhere (mask false)
Sources: "pred" (own previous maps + own voted pose), "pred_gtpose" (own maps, GT pose), "oracle_predpose" (GT maps, own pose),
"oracle" (GT maps + GT pose), "gtcur" (current frame's GT at t0, sanity ceiling).

python -m pallet.sequence --run runs/b256_run6 --split runs/fixed_val --picks 40 --t0 0.5 0.8 --steps 1 3 --out runs/seq_eval
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import targets as T
from .dataset import PalletDataset
from .metrics import aggregate, format_table, image_metrics
from .model import PalletNet
from .pose import estimate_pose


def pose_mat(R, p):
    M = np.eye(4); M[:3, :3] = np.asarray(R, np.float64).reshape(3, 3); M[:3, 3] = np.asarray(p, np.float64); return M


def relative_motion(pose_k: np.ndarray, pose_k1: np.ndarray) -> np.ndarray:
    """pose_* = pallet_P_cam [12] of the two frames -> dT with X_{k+1} = A X_k + b (camera k -> camera k+1)."""
    Tk, Tk1 = pose_mat(pose_k[:9], pose_k[9:]), pose_mat(pose_k1[:9], pose_k1[9:])
    return np.linalg.inv(Tk1) @ Tk


def propagate_pose(R: np.ndarray, c: np.ndarray, dT: np.ndarray):
    A, b = dT[:3, :3], dT[:3, 3]
    R2 = R @ A.T
    return R2, c - R2 @ b


def consistent_mask(P, rng, c_prev, rays_prev_pal, tol_m):
    """Pixels whose predicted P and predicted range agree along the (previous frame's) pallet-frame ray: |P - c - r d| < tol."""
    on_ray = c_prev[:, None, None] + rng[None] * np.transpose(rays_prev_pal, (2, 0, 1))
    return np.linalg.norm(P - on_ray, axis=0) < tol_m


def render_prior(P: np.ndarray, rng: np.ndarray, cls: np.ndarray, valid: np.ndarray, R: np.ndarray, c: np.ndarray,
                 intrinsics, H: int, W: int, max_range=12.0, splat=3, ground=True, angles_only=False, ground_side_m=0.9):
    """Previous frame's maps (P [3,H,W] m, rng [H,W], cls [H,W], valid [H,W] bool) seen from the pose (R, c) of the new frame
    -> (cont [6,H,W] encoded, mask [6,H,W] bool, cls_prior [H,W] with 255 where unknown)."""
    fx, fy, cx, cy = intrinsics
    rays = T.pixel_rays(fx, fy, cx, cy, H, W)                      # camera frame of the new view
    d_pal = rays @ R.T
    theta = np.arccos(np.clip(d_pal[..., 1], -1, 1)); phi = np.arctan2(d_pal[..., 2], d_pal[..., 0])
    zbuf = np.full((H, W), np.inf); Pn = np.full((3, H, W), np.nan, np.float32); cls_n = np.full((H, W), T.CLS_IGNORE, np.uint8)
    pts = P[:, valid].T; cl = cls[valid]
    if angles_only:
        pts = pts[:0]; ground = False
    if len(pts):
        Xc = (pts - c) @ R                                         # R^T (X_pal - c): camera coords in the new view
        front = Xc[:, 2] > 0.05; Xc, pts, cl = Xc[front], pts[front], cl[front]
        u = fx * Xc[:, 0] / Xc[:, 2] + cx; v = fy * Xc[:, 1] / Xc[:, 2] + cy; r = np.linalg.norm(Xc, axis=1)
        order = np.argsort(-r); u, v, r, pts, cl = u[order], v[order], r[order], pts[order], cl[order]   # nearest written last
        for du in range(-(splat // 2), splat // 2 + 1):
            for dv in range(-(splat // 2), splat // 2 + 1):
                ui = np.floor(u).astype(int) + du; vi = np.floor(v).astype(int) + dv
                m = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
                zbuf[vi[m], ui[m]] = r[m]; Pn[:, vi[m], ui[m]] = pts[m].T; cls_n[vi[m], ui[m]] = cl[m]
    if ground:
        # analytic ground plane y = 0, only where the ground can actually be seen: in front of the pallet face (z < 0) or beside
        # the pallet (|x| > ground_side_m); never under/behind it, where an uncovered pallet pixel would otherwise get a
        # "ground 1 m further back" prior that the network copies (P displaced along the ray, range following)
        with np.errstate(divide="ignore", invalid="ignore"):
            tg = -c[1] / d_pal[..., 1]
        G = c + tg[..., None] * d_pal
        visible = (G[..., 2] < -0.05) | (np.abs(G[..., 0]) > ground_side_m)
        gm = (d_pal[..., 1] > 1e-6) & (tg > 0.3) & (tg <= max_range) & (tg < zbuf) & visible
        zbuf[gm] = tg[gm]; Pn[:, gm] = G[gm].T; cls_n[gm] = T.CLS_GROUND
    hit = np.isfinite(zbuf) & (zbuf <= max_range)
    rng_n = np.where(hit, zbuf, np.nan).astype(np.float32); Pn[:, ~hit] = np.nan
    cont, mask, _ = T.encode({"P": Pn, "range": rng_n, "theta": theta.astype(np.float32), "phi": phi.astype(np.float32), "cls": cls_n})
    return cont, mask, cls_n


def frame_pose_from_pred(dec: dict, cls_pred: np.ndarray, real: np.ndarray, intrinsics, H, W):
    fx, fy, cx, cy = intrinsics
    rays = T.pixel_rays(fx, fy, cx, cy, H, W)
    vote = np.isin(cls_pred, (T.CLS_PALLET, T.CLS_LOAD)) & real & np.isfinite(dec["range"])
    est = estimate_pose(dec["P"], dec["range"], dec["theta"], dec["phi"], rays, vote, real)
    return est["R"], est["c"], est["n_votes"]


def smooth_pose(R_prop, c_prop, R_new, c_new, alpha):
    """Blend the propagated previous (smoothed) pose with the new frame's vote: alpha = weight of the new vote."""
    if R_prop is None:
        return R_new, c_new
    M = (1 - alpha) * R_prop + alpha * R_new
    U, _, Vt = np.linalg.svd(M); R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return R, (1 - alpha) * c_prop + alpha * c_new


@torch.no_grad()
def run_pick(net, ds: PalletDataset, idx: list[int], source: str, t0: float, steps: int, dev, prior_fill="noise", splat=3,
             prior_classes=(T.CLS_PALLET, T.CLS_LOAD), consistency_m=0.0, pose_alpha=1.0, prior_channels="all"):
    """Frames idx of one pick in time order. Returns per-frame metric rows (+ 'prior_valid_frac').
    source: pred | pred_gtpose | oracle | oracle_predpose | gtcur | pose (angles only, own smoothed pose) | pose_gtpose.
    Own-prediction maps are restricted to prior_classes and to pixels whose P and range agree within consistency_m (metres);
    the own pose used for propagation is exponentially smoothed with weight pose_alpha on the newest vote (1 = raw vote)."""
    rows, prev = [], None
    gen = torch.Generator(device=dev); gen.manual_seed(0)              # reproducible noise fill
    for j in idx:
        s = ds[j]
        H, W = s["cls"].shape; intr = s["intrinsics"].tolist(); real = s["pixel_mask"].numpy()
        pose_gt = s["pallet_P_cam"].numpy().astype(np.float64)
        x_prior = m_prior = None; pvf = 0.0
        if prev is not None and t0 > 0:
            dT = relative_motion(prev["pose_gt"], pose_gt)                         # perfect odometry for now
            if source == "gtcur":
                cont, mask = s["target"].numpy(), s["valid_mask"].numpy()
            else:
                # maps: GT ("oracle*") or own prediction ("pred*", "pose*"); pose: GT ("*gtpose", "oracle") or own smoothed vote
                use_gt_maps = source.startswith("oracle"); use_gt_pose = source in ("oracle", "pred_gtpose", "pose_gtpose")
                maps = prev["gt"] if use_gt_maps else prev["pred"]; cls0 = prev["cls_gt"] if use_gt_maps else prev["cls_pred"]
                if use_gt_pose:
                    R0, c0 = prev["pose_gt"][:9].reshape(3, 3), prev["pose_gt"][9:]
                else:
                    R0, c0 = prev["R_sm"], prev["c_sm"]
                R1, c1 = propagate_pose(R0, c0, dT)
                valid0 = np.isfinite(maps["range"]) & prev["real"]
                if not use_gt_maps:
                    valid0 &= np.isin(cls0, prior_classes)
                    if consistency_m:
                        valid0 &= consistent_mask(maps["P"], maps["range"], c0, prev["rays"] @ R0.T, consistency_m)
                cont, mask, _ = render_prior(maps["P"], maps["range"], cls0, valid0, R1, c1, intr, H, W, splat=splat,
                                             angles_only=source.startswith("pose"))
            mask = mask & real[None]
            if prior_channels == "maps":                                   # leave theta/phi to the image (no pose-derived angle prior)
                mask[T.RAY_CHANNELS] = False
            pvf = float(mask[0][real].mean())
            x_prior = torch.from_numpy(cont)[None].to(dev); m_prior = torch.from_numpy(mask)[None].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x, logits = net.generate(s["image"][None].to(dev), steps=steps, x_prior=x_prior, prior_mask=m_prior,
                                     t0=(t0 if x_prior is not None else 0.0), prior_fill=prior_fill, generator=gen)
        row = image_metrics(s, x[0], logits[0]); row["prior_valid_frac"] = pvf; row["warm"] = x_prior is not None
        rows.append(row)
        dec = T.decode(x[0].cpu().numpy(), np.ones((6, H, W), bool)); cls_pred = logits[0].argmax(0).cpu().numpy()
        R_est, c_est, _ = frame_pose_from_pred(dec, cls_pred, real, intr, H, W)
        if prev is None or not np.all(np.isfinite(c_est)):
            R_sm, c_sm = (R_est, c_est) if prev is None else propagate_pose(prev["R_sm"], prev["c_sm"], relative_motion(prev["pose_gt"], pose_gt))
        else:
            R_pp, c_pp = propagate_pose(prev["R_sm"], prev["c_sm"], relative_motion(prev["pose_gt"], pose_gt))
            R_sm, c_sm = smooth_pose(R_pp, c_pp, R_est, c_est, pose_alpha)
        fx, fy, cx, cy = intr
        prev = {"pose_gt": pose_gt, "gt": T.decode(s["target"].numpy(), s["valid_mask"].numpy()), "cls_gt": s["cls"].numpy(),
                "pred": dec, "cls_pred": cls_pred, "R_est": R_est, "c_est": c_est, "R_sm": R_sm, "c_sm": c_sm, "real": real,
                "rays": T.pixel_rays(fx, fy, cx, cy, H, W)}
        row["smoothed_origin_err_cm"] = float(np.linalg.norm(c_sm - pose_gt[9:]) * 100)
        # fused pose: the frame's own votes re-projected with the fused rotation R_sm (yaw scatter averages out along the approach)
        vote = np.isin(cls_pred, (T.CLS_PALLET, T.CLS_LOAD)) & real & np.isfinite(dec["range"])
        if vote.any():
            c_f = np.median(dec["P"][:, vote].T - dec["range"][vote][:, None] * (prev["rays"][vote] @ R_sm.T), axis=0)
            e = c_f - pose_gt[9:]; R_gt = pose_gt[:9].reshape(3, 3); dR = R_sm @ R_gt.T
            ang = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)); ax = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
            row.update({"fused_origin_err_cm": float(np.linalg.norm(e) * 100), "fused_lateral_err_cm": float(abs(e[0]) * 100),
                        "fused_depth_err_cm": float(abs(e[2]) * 100), "fused_yaw_err_deg": float(abs(np.degrees(ang) * ax[1] / max(np.linalg.norm(ax), 1e-12)))})
    return rows


def jitter(rows):
    """Median over picks of the frame-to-frame std of the origin error vector (cm), within 6 m."""
    out = []
    for key in sorted({r["key"] for r in rows}):
        e = np.array([r["origin_err_xyz_cm"] for r in rows if r["key"] == key and r["face_dist_m"] < 6 and np.all(np.isfinite(r["origin_err_xyz_cm"]))])
        if len(e) >= 3:
            out.append(float(np.linalg.norm(np.diff(e, axis=0), axis=1).mean()))
    return float(np.median(out)) if out else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/b256_run6")
    ap.add_argument("--split", default="runs/fixed_val")
    ap.add_argument("--picks", type=int, default=40, help="number of val picks (evenly chosen)")
    ap.add_argument("--sources", nargs="+", default=["pred", "pred_gtpose", "pose", "pose_gtpose", "oracle_predpose", "oracle", "gtcur"])
    ap.add_argument("--t0", type=float, nargs="+", default=[0.5, 0.8])
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--steps3_t0", type=float, nargs="*", default=None, help="restrict multi-step configs to these t0 values")
    ap.add_argument("--prior_fill", default="noise", choices=["noise", "mixed", "zeros"])
    ap.add_argument("--splat", type=int, default=3)
    ap.add_argument("--consistency_cm", type=float, default=5.0, help="own-map pixels must have |P - c - r d| below this (0 = off)")
    ap.add_argument("--prior_ground", action="store_true", help="also splat predicted ground pixels (ground plane is drawn analytically anyway)")
    ap.add_argument("--pose_alpha", type=float, default=1.0, help="weight of the newest vote in the smoothed own pose (1 = raw vote)")
    ap.add_argument("--prior_channels", default="all", choices=["all", "maps"], help="maps: prior for P/range only, angles left unknown")
    ap.add_argument("--out", default="runs/seq_eval")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    args = json.load(open(Path(a.run) / "args.json"))
    df = pd.read_csv(Path(a.split) / "split.csv", dtype={"robot": str}); df = df[df.split == "val"]
    picks = df.drop_duplicates(["robot", "pick"])[["robot", "pick"]].values.tolist()
    sel = [tuple(picks[i]) for i in np.linspace(0, len(picks) - 1, min(a.picks, len(picks))).round().astype(int)]
    df = df[[(r, p) in set(sel) for r, p in zip(df.robot, df.pick)]].sort_values(["robot", "pick", "scan"]).reset_index(drop=True)
    ds = PalletDataset(df, max_range_m=args.get("max_range_m"))
    groups = [list(g.index) for _, g in df.groupby(["robot", "pick"], sort=False)]
    print(f"{len(groups)} picks, {len(df)} frames", flush=True)
    ck = torch.load(Path(a.run) / "checkpoint-last.pth", map_location="cpu", weights_only=False)
    net = PalletNet(model=args["model"], feature_embedding_type=args["feature_embedding_type"]); net.load_state_dict(ck["ema"]); net = net.cuda().eval()
    dev = torch.device("cuda")
    configs = [("single", 0.0, 1)] + [(src, t0, st) for src in a.sources for t0 in a.t0 for st in a.steps
               if st == 1 or a.steps3_t0 is None or t0 in a.steps3_t0]
    summary = []
    for src, t0, st in configs:
        t_start = time.time(); rows = []
        for idx in groups:
            rows += run_pick(net, ds, idx, src, t0, st, dev, a.prior_fill, a.splat,
                             prior_classes=(T.CLS_PALLET, T.CLS_LOAD) + ((T.CLS_GROUND,) if a.prior_ground else ()),
                             consistency_m=a.consistency_cm / 100.0, pose_alpha=a.pose_alpha, prior_channels=a.prior_channels)
        agg = aggregate(rows); jit = jitter(rows)
        name = f"{src}_t{t0}_s{st}" + ("" if a.prior_channels == "all" else "_maps") + ("" if a.prior_fill == "noise" else f"_{a.prior_fill}")
        json.dump({"config": name, "per_image": rows, "aggregate": agg, "jitter_cm": jit}, open(out / f"{name}.json", "w"), indent=1)
        warm = [r for r in rows if r["warm"]]
        sm = float(np.median([r["smoothed_origin_err_cm"] for r in rows if np.isfinite(r.get("smoothed_origin_err_cm", np.nan))]))
        line = {"config": name, "jitter_cm": round(jit, 2), "prior_valid": round(float(np.mean([r["prior_valid_frac"] for r in warm])), 2) if warm else 0.0,
                **{r["bin"]: round(r["origin_err_cm"], 1) for r in agg}, "P_all": round(agg[-1]["P_err_cm"], 1), "yaw_all": round(agg[-1]["yaw_err_deg"], 2),
                **{"lat " + r["bin"]: round(r["lateral_err_cm"], 1) for r in agg},
                **{"fused lat " + r["bin"]: round(r["fused_lateral_err_cm"], 1) for r in agg}, "fused_yaw_all": round(agg[-1]["fused_yaw_err_deg"], 2)}
        summary.append(line)
        print(f"[{name}] {time.time() - t_start:.0f}s", {k: v for k, v in line.items() if k != "config"}, flush=True)
        print(format_table(agg).split("\n\n")[0], flush=True)
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
