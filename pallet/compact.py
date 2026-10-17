"""Compact ray records: keep only what cannot be recomputed (cls, range, pose); P/theta/phi are rebuilt on load
(targets.reconstruct). ~250 KB per image instead of ~1.4 MB. Layout in the training store (default
~/Anantak/Data/PalletTrainImages): <robot>/<pick>/rays/{cameras.json, cam2_NNNN.npz}, next to <robot>/<pick>/02/ images.

python -m pallet.compact --src-root ~/Anantak/Data/PalletModels [--delete-src] [--min-age-min 2] [--picks 249/1720 ...]
    converts every <src-root>/<robot>/<pick>/rays not yet in the store (verified against the source before any deletion).
python -m pallet.compact --src-root <dir with <robot>/<pick>/rays> ...     e.g. a temporary rsync of ana-ml-02 picks.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from . import targets as T

STORE = Path("~/Anantak/Data/PalletTrainImages").expanduser()
KEEP = ("cls", "range", "pallet_P_cam", "cam_in_pallet", "scan", "ts_img_us")


def compact_file(src: Path, dst: Path, cam: dict, verify=True) -> dict:
    d = dict(np.load(src))
    if "P" not in d:                                   # already compact
        shutil.copyfile(src, dst)
        return {"ok": True, "already": True}
    rec = {k: d[k] for k in KEEP if k in d}
    if verify:
        f, cx, cy = cam["f_px"], cam["cx"], cam["cy"]
        full = T.reconstruct(rec["cls"], rec["range"], rec["pallet_P_cam"], f, f, cx, cy)
        obj = np.isfinite(d["range"]) & np.isin(d["cls"], (T.CLS_PALLET, T.CLS_LOAD, T.CLS_GROUND))
        p_err = float(np.linalg.norm((full["P"] - d["P"])[:, obj], axis=0).max()) if obj.any() else 0.0
        a_err = float(max(np.abs(full["theta"] - d["theta"]).max(), np.abs(full["phi"] - d["phi"]).max()))
        if not (p_err < 2e-3 and a_err < 1e-4):
            return {"ok": False, "p_err": p_err, "a_err": a_err}
    tmp = dst.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **rec)
    tmp.rename(dst)
    return {"ok": True, "already": False}


def compact_pick(src_rays: Path, dst_rays: Path, delete_src=False, verify=True) -> dict:
    cam = json.load(open(src_rays / "cameras.json"))
    dst_rays.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_rays / "cameras.json", dst_rays / "cameras.json")
    n, bad = 0, []
    for src in sorted(src_rays.glob("cam2_*.npz")):
        r = compact_file(src, dst_rays / src.name, cam, verify)
        if r["ok"]:
            n += 1
        else:
            bad.append((src.name, r))
    if delete_src and not bad:
        for src in src_rays.glob("cam2_*.npz"):
            src.unlink()
    return {"n": n, "bad": bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True, help="dir holding <robot>/<pick>/rays/")
    ap.add_argument("--store", default=str(STORE))
    ap.add_argument("--picks", nargs="*", default=None)
    ap.add_argument("--delete-src", action="store_true", help="remove the source cam2_*.npz after a verified conversion")
    ap.add_argument("--min-age-min", type=float, default=2.0, help="skip rays dirs modified more recently than this (being written)")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()
    src_root, store = Path(a.src_root).expanduser(), Path(a.store).expanduser()
    want = set(a.picks) if a.picks else None
    tot, picks, skipped = 0, 0, 0
    for cj in sorted(src_root.glob("*/*/rays/cameras.json")):
        rays = cj.parent
        robot, pick = rays.parts[-3], rays.parts[-2]
        if want is not None and f"{robot}/{pick}" not in want:
            continue
        srcs = list(rays.glob("cam2_*.npz"))
        if not srcs:
            continue
        if time.time() - max(f.stat().st_mtime for f in srcs) < a.min_age_min * 60:
            skipped += 1
            continue
        dst = store / robot / pick / "rays"
        if dst.exists() and len(list(dst.glob("cam2_*.npz"))) >= len(srcs) and not a.delete_src:
            continue
        r = compact_pick(rays, dst, a.delete_src, not a.no_verify)
        tot += r["n"]; picks += 1
        if r["bad"]:
            print(f"{robot}/{pick}: {len(r['bad'])} files FAILED verification, source kept: {r['bad'][:2]}")
    print(f"compacted {tot} images of {picks} picks into {store} (skipped {skipped} picks still being written)")


if __name__ == "__main__":
    main()
