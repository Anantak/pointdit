"""Pallet pose from the per-pixel predictions.

Each real pixel's predicted (theta, phi) gives the ray direction in the pallet frame, d_pal = (sin th cos ph, cos th,
sin th sin ph); with the known camera ray d_cam the rotation R (pallet_P_cam) follows by Kabsch over all pixels.
Each pallet/load pixel then votes for the camera centre in the pallet frame: c_i = P_i - range_i * R d_i.
The pose is (R, c = robust centre of the votes); cam_in_pallet is the ground truth for c.
"""
from __future__ import annotations

import numpy as np

from . import targets as T


def dirs_from_angles(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    st = np.sin(theta)
    return np.stack([st * np.cos(phi), np.cos(theta), st * np.sin(phi)], -1)


def kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """R minimising |b - R a| over paired unit vectors a, b [N,3]."""
    H = a.T @ b
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def estimate_pose(P: np.ndarray, rng: np.ndarray, theta: np.ndarray, phi: np.ndarray, rays_cam: np.ndarray,
                  vote_mask: np.ndarray, ray_mask: np.ndarray, R: np.ndarray | None = None) -> dict:
    """P [3,H,W] m, rng/theta/phi [H,W], rays_cam [H,W,3] unit, vote_mask (pallet/load pixels), ray_mask (real pixels).
    R=None: estimate it from theta/phi; else use the given pallet_P_cam rotation. Returns R, c (median vote), c_mean,
    spread (median |vote - c|) and n_votes."""
    if R is None:
        R = kabsch(rays_cam[ray_mask], dirs_from_angles(theta[ray_mask], phi[ray_mask]))
    if vote_mask.sum() == 0:
        return {"R": R, "c": np.full(3, np.nan), "c_mean": np.full(3, np.nan), "spread": np.nan, "n_votes": 0}
    d_pal = rays_cam[vote_mask] @ R.T
    votes = P[:, vote_mask].T - rng[vote_mask][:, None] * d_pal
    c = np.median(votes, axis=0)
    return {"R": R, "c": c, "c_mean": votes.mean(0), "spread": float(np.median(np.linalg.norm(votes - c, axis=1))),
            "n_votes": int(vote_mask.sum())}


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    """Geodesic angle between two rotations, computed from the Frobenius distance (well conditioned near 0, unlike arccos)."""
    d = np.linalg.norm(np.asarray(R_est, np.float64) - np.asarray(R_gt, np.float64))
    return float(np.degrees(2.0 * np.arcsin(np.clip(d / (2.0 * np.sqrt(2.0)), 0.0, 1.0))))
