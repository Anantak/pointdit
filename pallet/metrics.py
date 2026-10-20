"""Per-image metrics for the 7-channel prediction and their aggregation by face distance."""
from __future__ import annotations

import numpy as np
import torch

from . import targets as T
from .pose import estimate_pose, rotation_error_deg

DIST_BINS = (0.0, 2.0, 3.0, 4.5, 6.0, 8.0, 12.0)


def image_metrics(sample: dict, x_pred: torch.Tensor, logits: torch.Tensor, rot_from: str = "all") -> dict:
    """sample: one dataset item (CPU tensors); x_pred [6,H,W] encoded, logits [4,H,W].
    rot_from: "all" = rotation from theta/phi of every real pixel, "obj" = from predicted pallet/load pixels (>= 500, else all)."""
    H, W = sample["cls"].shape
    real = sample["pixel_mask"].numpy()
    cls_gt = sample["cls"].numpy()
    cls_pred = logits.argmax(0).cpu().numpy()
    valid = sample["valid_mask"].numpy()
    gt = T.decode(sample["target"].numpy(), valid)
    pr = T.decode(x_pred.cpu().numpy(), np.ones_like(valid))
    m = {"key": sample["key"], "scan": int(sample["scan"]), "face_dist_m": float(sample["face_dist_m"])}
    m["cls_acc"] = float((cls_pred[real] == cls_gt[real]).mean())
    for k, name in enumerate(T.CLASS_NAMES):
        g, p = (cls_gt == k) & real, (cls_pred == k) & real
        m[f"iou_{name}"] = float((g & p).sum() / max(1, (g | p).sum()))
    obj = np.isin(cls_gt, (T.CLS_PALLET, T.CLS_LOAD)) & valid[0]
    if obj.any():
        m["P_err_cm"] = float(np.median(np.linalg.norm((pr["P"] - gt["P"])[:, obj], axis=0)) * 100)
        m["range_err_cm"] = float(np.median(np.abs(pr["range"] - gt["range"])[obj]) * 100)
    m["theta_err_deg"] = float(np.degrees(np.median(np.abs(pr["theta"] - gt["theta"])[real])))
    m["phi_err_deg"] = float(np.degrees(np.median(np.abs(pr["phi"] - gt["phi"])[real])))
    fx, fy, cx, cy = sample["intrinsics"].tolist()
    rays = T.pixel_rays(fx, fy, cx, cy, H, W)
    pose_gt = sample["pallet_P_cam"].numpy()
    R_gt, c_gt = pose_gt[:9].reshape(3, 3), sample["cam_in_pallet"].numpy()
    vote_pred = np.isin(cls_pred, (T.CLS_PALLET, T.CLS_LOAD)) & real
    ray_mask = vote_pred if (rot_from == "obj" and vote_pred.sum() >= 500) else real
    est = estimate_pose(pr["P"], pr["range"], pr["theta"], pr["phi"], rays, vote_pred, ray_mask)
    m["rot_err_deg"] = rotation_error_deg(est["R"], R_gt)
    dR = est["R"] @ R_gt.T                                                   # signed error as a rotation vector (deg), pallet frame
    ang = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
    axis = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
    m["rot_err_vec_deg"] = (np.degrees(ang) * axis / max(np.linalg.norm(axis), 1e-12)).tolist()
    m["origin_err_cm"] = float(np.linalg.norm(est["c"] - c_gt) * 100)
    m["origin_err_xyz_cm"] = ((est["c"] - c_gt) * 100).tolist()
    m["vote_spread_cm"] = est["spread"] * 100
    m["n_votes"] = est["n_votes"]
    est_gt = estimate_pose(pr["P"], pr["range"], pr["theta"], pr["phi"], rays, obj, real, R=R_gt)   # GT mask + GT R
    m["origin_err_gtmask_cm"] = float(np.linalg.norm(est_gt["c"] - c_gt) * 100)
    return m


def aggregate(rows: list[dict], bins=DIST_BINS) -> list[dict]:
    """Median of the key metrics per face-distance bin, plus an 'all' row."""
    out = []
    keys = ("cls_acc", "iou_pallet", "iou_load", "P_err_cm", "range_err_cm", "theta_err_deg", "phi_err_deg",
            "rot_err_deg", "origin_err_cm", "origin_err_gtmask_cm", "vote_spread_cm")
    d = np.array([r["face_dist_m"] for r in rows])
    for lo, hi in list(zip(bins[:-1], bins[1:])) + [(0.0, np.inf)]:
        sel = [r for r, x in zip(rows, d) if lo <= x < hi]
        if not sel:
            continue
        row = {"bin": "all" if hi == np.inf else f"{lo:g}-{hi:g} m", "n": len(sel)}
        for k in keys:
            v = [r[k] for r in sel if k in r and np.isfinite(r[k])]
            row[k] = float(np.median(v)) if v else float("nan")
        out.append(row)
    return out


def format_table(agg: list[dict]) -> str:
    cols = ["bin", "n", "cls_acc", "iou_pallet", "iou_load", "P_err_cm", "range_err_cm", "theta_err_deg", "phi_err_deg",
            "rot_err_deg", "origin_err_cm", "origin_err_gtmask_cm", "vote_spread_cm"]
    lines = ["  ".join(f"{c:>14s}" for c in cols)]
    for r in agg:
        lines.append("  ".join(f"{r[c]:>14}" if isinstance(r[c], (str, int)) else f"{r[c]:>14.2f}" for c in cols))
    return "\n".join(lines)
