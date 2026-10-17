"""Training manifest from the training store (no dependency on the SensorUnit models tree):
one row per image that has a compact/full ray record AND an image on disk. Kinds come from SensorUnit's picks.parquet.

python -m pallet.manifest [--store ~/Anantak/Data/PalletTrainImages] [--picks-parquet ~/Anantak/Data/PalletModels/manifest/picks.parquet]
writes <store>/manifest/images.parquet (+ .csv). Columns match the subset that pallet/dataset.py uses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

STORE = Path("~/Anantak/Data/PalletTrainImages").expanduser()
PICKS = Path("~/Anantak/Data/PalletModels/manifest/picks.parquet").expanduser()


def build(store: Path, picks_parquet: Path | None) -> pd.DataFrame:
    kinds = {}
    if picks_parquet and picks_parquet.exists():
        pk = pd.read_parquet(picks_parquet)
        kinds = {k: (kid, bool(nap)) for k, kid, nap in zip(pk["key"], pk["kind_id"], pk["not_a_pick"].fillna(False))}
    rows = []
    for cj in sorted(store.glob("*/*/rays/cameras.json")):
        rays = cj.parent
        robot, pick = rays.parts[-3], rays.parts[-2]
        cam = json.load(open(cj))
        key = f"thd-vb-{robot}-{int(pick)}"
        kind_id, nap = kinds.get(key, (np.nan, False))
        for im in cam["images"]:
            tensor = rays / f"cam2_{im['scan']:04d}.npz"
            if not tensor.exists():
                continue
            image = store / robot / pick / im["image"]
            R, p = im["pallet_P_cam"]["R"], im["pallet_P_cam"]["p"]
            row = {"key": key, "robot": robot, "pick": int(pick), "scan": int(im["scan"]), "image": im["image"], "image_abs": str(image),
                   "image_on_disk": image.exists(), "rot_code": cam["rot_code"], "ts_img_us": int(im["ts_img_us"]),
                   "face_dist_m": float(im["face_dist_m"]), "cam_in_pallet_x": im["cam_in_pallet"][0], "cam_in_pallet_y": im["cam_in_pallet"][1],
                   "cam_in_pallet_z": im["cam_in_pallet"][2], "tensor": str(tensor), "kind_id": kind_id, "not_a_pick": nap, "split": None,
                   "tensor_h": cam["image_height"], "tensor_w": cam["image_width"]}
            row.update({f"pallet_P_cam_R{i}": R[i] for i in range(9)})
            row.update({f"pallet_P_cam_p{i}": p[i] for i in range(3)})
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=str(STORE))
    ap.add_argument("--picks-parquet", default=str(PICKS))
    a = ap.parse_args()
    store = Path(a.store).expanduser()
    m = build(store, Path(a.picks_parquet).expanduser())
    out = store / "manifest"; out.mkdir(exist_ok=True)
    m.to_parquet(out / "images.parquet", index=False); m.to_csv(out / "images.csv", index=False)
    ok = m[m.image_on_disk]
    print(f"wrote {out / 'images.parquet'}: {len(m)} images / {m.groupby(['robot', 'pick']).ngroups} picks "
          f"({len(ok)} with image on disk, {ok.kind_id.notna().sum()} with a kind)")


if __name__ == "__main__":
    main()
