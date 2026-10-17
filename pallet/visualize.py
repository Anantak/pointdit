"""Inspection panels for decoded targets, same colour scheme as SensorUnit's pallet_rays.py --panels.

python -m pallet.visualize --picks 249/1720 --scans 0 20 40 --out /tmp/panels      (defaults: first pick, 3 scans)
Panel (2x3): photo + pallet-frame position colour overlay | range 0-8 m | position colour | theta | phi | class masks.
Position colour on a FIXED metre scale: R = x lateral -0.75..0.75, G = height 0..2 (height = -y), B = depth z 0..1.3.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import targets as T


def colourise(P, cls):
    x, y, z = np.nan_to_num(P[0]), np.nan_to_num(P[1]), np.nan_to_num(P[2])
    r = np.clip((x - T.COL_SCALE["x"][0]) / (T.COL_SCALE["x"][1] - T.COL_SCALE["x"][0]), 0, 1)
    g = np.clip((-y - T.COL_SCALE["h"][0]) / (T.COL_SCALE["h"][1] - T.COL_SCALE["h"][0]), 0, 1)
    b = np.clip((z - T.COL_SCALE["z"][0]) / (T.COL_SCALE["z"][1] - T.COL_SCALE["z"][0]), 0, 1)
    rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    rgb[~np.isin(cls, (T.CLS_PALLET, T.CLS_LOAD))] = 0
    return rgb


def heat(v, vmin, vmax, invalid=(0, 0, 0)):
    import matplotlib
    cm = matplotlib.colormaps["turbo"]
    t = np.clip((v - vmin) / max(1e-9, vmax - vmin), 0, 1)
    bad = ~np.isfinite(v)
    rgb = (cm(np.nan_to_num(t))[..., :3] * 255).astype(np.uint8)
    rgb[bad] = invalid
    return rgb


def class_masks(cls, rng):
    masks = np.zeros(cls.shape + (3,), np.uint8)
    masks[cls == T.CLS_PALLET] = (230, 160, 40)
    masks[cls == T.CLS_LOAD] = (60, 120, 230)
    masks[cls == T.CLS_GROUND] = (90, 140, 90)
    masks[(cls == T.CLS_BG) & np.isfinite(rng)] = (120, 120, 120)
    return masks


def panel(dec: dict, caption: str = "") -> Image.Image:
    """dec: {"image" u8 [H,W,3], "cls", "range", "P", "theta", "phi"} (un-padded or padded, NaN where invalid)."""
    photo, cls = dec["image"], dec["cls"]
    H, W = cls.shape
    col = colourise(dec["P"], cls)
    ov = photo.astype(np.float32).copy()
    obj = np.isin(cls, (T.CLS_PALLET, T.CLS_LOAD)); ov[obj] = 0.35 * ov[obj] + 0.65 * col[obj]
    gnd = cls == T.CLS_GROUND; ov[gnd] = 0.6 * ov[gnd] + 0.4 * np.array([90, 140, 90], np.float32)
    m2 = (0.5 * photo + 0.5 * class_masks(cls, dec["range"])).astype(np.uint8)
    tiles = [ov.astype(np.uint8), heat(dec["range"], 0, 8), col, heat(dec["theta"], 0, np.pi), heat(dec["phi"], -np.pi, np.pi), m2]
    names = ["photo + position colour (R=x G=height B=depth)", "range 0-8 m", "pallet-frame position colour",
             "theta: ray vs gravity-down (0-180 deg)", "phi: ray on ground vs pallet x (-180..180)",
             "classes: pallet orange, load blue, ground green, bg+range grey"]
    pad, cap = 6, 14
    sheet = Image.new("RGB", (3 * W + 4 * pad, 2 * (H + cap) + 3 * pad), (25, 25, 25))
    dr = ImageDraw.Draw(sheet)
    for k, (t, nm) in enumerate(zip(tiles, names)):
        x0, y0 = pad + (k % 3) * (W + pad), pad + (k // 3) * (H + cap + pad)
        sheet.paste(Image.fromarray(t), (x0, y0))
        dr.text((x0 + 2, y0 + H + 1), nm[:60], fill=(200, 200, 200))
    dr.text((pad, 2 * (H + cap) + 2 * pad - 2), caption, fill=(255, 210, 50))
    return sheet


def main():
    from .dataset import PalletDataset, decode_sample
    ap = argparse.ArgumentParser()
    ap.add_argument("--picks", nargs="+", default=None, help="robot/pick ...; default: the first pick on disk")
    ap.add_argument("--scans", nargs="*", type=int, default=None, help="scan ids; default first/middle/last of each pick")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--padded", action="store_true", help="render the padded 432x240 tensors instead of the raw 424x240")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if not a.picks:
        from .dataset import load_manifest
        r = load_manifest().iloc[0]
        a.picks = [f"{r.robot}/{r.pick}"]
    for pick in a.picks:
        ds = PalletDataset(picks=[pick])
        scans = a.scans if a.scans else sorted({int(ds.df.scan.iloc[j]) for j in (0, len(ds) // 2, len(ds) - 1)})
        for j in range(len(ds)):
            if int(ds.df.scan.iloc[j]) not in scans:
                continue
            s = ds[j]
            dec = decode_sample(s) if not a.padded else {
                "image": (s["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8), "cls": s["cls"].numpy().astype(np.uint8),
                **T.decode(s["target"].numpy(), s["valid_mask"].numpy())}
            cap = f"{s['key']} scan {s['scan']:04d}  face {float(s['face_dist_m']):.2f} m  cam in pallet {np.round(s['cam_in_pallet'].numpy().astype(np.float64), 2).tolist()}"
            out = a.out / f"{pick.replace('/', '_')}_cam2_{s['scan']:04d}.png"
            panel(dec, cap).save(out)
            print(out)


if __name__ == "__main__":
    main()
