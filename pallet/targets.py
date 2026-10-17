"""The 7 per-pixel pallet targets, their fixed encoding, padding to a patch-16 size, and pixel rays.

Channels (per pixel of the upright 424x240 forkview image, padded here to 432x240):
  cls    int   0 pallet, 1 load, 2 ground, 3 background; 255 = ignore (padding)
  P      3 x m hit point in the pallet frame (origin: face centre on the ground; x lateral, y DOWN, z into the pallet)
  range  m     ray length from the camera centre to the hit
  theta  rad   angle between the ray and gravity-down (= +y of the pallet frame)
  phi    rad   atan2(d_z, d_x) of the ray projected on the ground plane; a ray straight into the face has phi = +pi/2
The 6 continuous channels are stacked in the order CHANNELS below. P/range are valid on hit pixels (finite range),
theta/phi on every real (non-padded) pixel. Everything is metric and in the pallet frame: no per-frame centring/scaling,
only the fixed, invertible per-channel affine (TARGET_OFFSET, TARGET_SCALE) so the network sees O(1) numbers
(chosen once from the data statistics on 2026-10-15; never per frame).
"""
from __future__ import annotations

import numpy as np
import torch

CLS_PALLET, CLS_LOAD, CLS_GROUND, CLS_BG, CLS_IGNORE = 0, 1, 2, 3, 255
NUM_CLASSES = 4
CLASS_NAMES = ("pallet", "load", "ground", "background")

CHANNELS = ("px", "py", "pz", "range", "theta", "phi")
NUM_CONT = len(CHANNELS)
HIT_CHANNELS = slice(0, 4)      # P and range: valid where the ray hits something with a known range
RAY_CHANNELS = slice(4, 6)      # theta, phi: pure ray geometry, valid on every real pixel

# Fixed per-channel affine, enc = (raw - OFFSET) / SCALE. Identity for now; to be chosen once (step 3) and never per frame.
TARGET_OFFSET = np.array([0.0, 0.0, 0.0, 0.0, np.pi / 2, np.pi / 2], np.float32)   # angles centred on "straight at the face"
TARGET_SCALE = np.array([2.0, 2.0, 4.0, 4.0, 0.5, 0.5], np.float32)                  # ~ per-channel std over hit pixels <= 12 m

TARGET_OFFSET_T = torch.from_numpy(TARGET_OFFSET)
TARGET_SCALE_T = torch.from_numpy(TARGET_SCALE)

# Upright tensor size from the pipeline and the padded size the ViT needs (multiples of the 16 px patch).
RAW_H, RAW_W = 424, 240
PAD_TOP, PAD_BOTTOM = 4, 4
PAD_H, PAD_W = RAW_H + PAD_TOP + PAD_BOTTOM, RAW_W
assert PAD_H % 16 == 0 and PAD_W % 16 == 0

# Inspection colours (identical to SensorUnit's pallet_rays.py so panels compare 1:1).
COL_SCALE = {"x": (-0.75, 0.75), "h": (0.0, 2.0), "z": (0.0, 1.3)}


def upright(raw: np.ndarray, rot_code: int) -> np.ndarray:
    """Raw camera frame -> upright image, as the pipeline's UprightImage (code 2 = 90 deg counter-clockwise)."""
    k = {0: 0, 1: 2, 2: 1, 3: 3}[int(rot_code)]
    return np.ascontiguousarray(np.rot90(raw, k))


def encode(npz: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ray tensor (dict of arrays as in cam2_NNNN.npz, un-padded) -> (cont [6,H,W] f32 with NaN->0, valid [6,H,W] bool, cls [H,W] int64)."""
    rng = np.asarray(npz["range"], np.float32)
    cont = np.concatenate([np.asarray(npz["P"], np.float32), rng[None], np.asarray(npz["theta"], np.float32)[None],
                           np.asarray(npz["phi"], np.float32)[None]], 0)
    valid = np.isfinite(cont)
    cont = (np.nan_to_num(cont, nan=0.0) - TARGET_OFFSET[:, None, None]) / TARGET_SCALE[:, None, None]
    cont[~valid] = 0.0
    return cont.astype(np.float32), valid, np.asarray(npz["cls"]).astype(np.int64)


def decode(cont: np.ndarray, valid: np.ndarray) -> dict:
    """Inverse of encode: -> {"P": [3,H,W], "range", "theta", "phi"} in metres/radians with NaN where invalid."""
    raw = np.asarray(cont, np.float32) * TARGET_SCALE[:, None, None] + TARGET_OFFSET[:, None, None]
    raw = np.where(valid, raw, np.nan).astype(np.float32)
    return {"P": raw[0:3], "range": raw[3], "theta": raw[4], "phi": raw[5]}


def pad(a: np.ndarray, fill=0) -> np.ndarray:
    """Pad the last two axes (H, W) from 424x240 to 432x240 (PAD_TOP rows above, PAD_BOTTOM below)."""
    assert a.shape[-2:] == (RAW_H, RAW_W), a.shape
    widths = [(0, 0)] * (a.ndim - 2) + [(PAD_TOP, PAD_BOTTOM), (0, 0)]
    return np.pad(a, widths, mode="constant", constant_values=fill)


def unpad(a: np.ndarray) -> np.ndarray:
    assert a.shape[-2:] == (PAD_H, PAD_W), a.shape
    return a[..., PAD_TOP:PAD_TOP + RAW_H, :]


def pad_intrinsics(fx: float, fy: float, cx: float, cy: float) -> tuple[float, float, float, float]:
    """Principal point of the padded image (pixel (0,0) is the top-left corner of the first padded row)."""
    return fx, fy, cx, cy + PAD_TOP


def pixel_rays(fx: float, fy: float, cx: float, cy: float, H: int, W: int) -> np.ndarray:
    """Unit ray of every pixel centre in the camera frame (x right, y down, z forward): [H,W,3] f64."""
    u = np.arange(W, dtype=np.float64) + 0.5
    v = np.arange(H, dtype=np.float64) + 0.5
    uu, vv = np.meshgrid(u, v)
    d = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], -1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def targets_from_pose(rays_cam: np.ndarray, rng: np.ndarray, pallet_P_cam: np.ndarray) -> dict:
    """What P, theta and phi must be, given the per-pixel ray, the range and the camera pose (R 3x3, p 3 of pallet_P_cam).
    Used by the round-trip test and later by the pose voting (P - p = range * R d)."""
    R, p = np.asarray(pallet_P_cam[:9], np.float64).reshape(3, 3), np.asarray(pallet_P_cam[9:12], np.float64)
    d = rays_cam @ R.T                                   # rays in the pallet frame; gravity-down is +y there
    P = p + rng[..., None] * d
    theta = np.arccos(np.clip(d[..., 1], -1.0, 1.0))
    phi = np.arctan2(d[..., 2], d[..., 0])
    return {"P": np.transpose(P, (2, 0, 1)).astype(np.float32), "theta": theta.astype(np.float32), "phi": phi.astype(np.float32)}


def reconstruct(cls: np.ndarray, rng: np.ndarray, pallet_P_cam: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> dict:
    """Compact record (cls, range, pose) -> the full target dict {"cls","range","P","theta","phi"} for the un-padded image.
    P = cam + range * R d on the pixel ray (identical to the pipeline's P to < 1 mm for pallet/load/ground pixels; background
    pixels, which the pipeline stores as splatted lidar points up to ~16 cm off the ray, become on-ray points), NaN where range is."""
    H, W = cls.shape
    out = targets_from_pose(pixel_rays(fx, fy, cx, cy, H, W), np.nan_to_num(rng), pallet_P_cam)
    out["P"][:, ~np.isfinite(rng)] = np.nan
    return {"cls": cls, "range": rng, **out}
