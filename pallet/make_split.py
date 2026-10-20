"""Build a split.csv (manifest rows + split column) for THIS machine's manifest from a pick-level assignment, so that
training/evaluation on another machine uses exactly the same picks.

python -m pallet.make_split --manifest <images.parquet> --picks <csv with robot,pick[,split]> --out <dir>/split.csv
    [--default train|drop]   split of picks not listed (default: drop them); a picks csv without a split column marks its picks 'val'.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .dataset import load_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--picks", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--default", default="drop", choices=["train", "drop"])
    a = ap.parse_args()
    m = load_manifest(a.manifest)
    pk = pd.read_csv(a.picks, dtype={"robot": str})
    if "split" not in pk:
        pk["split"] = "val"
    assign = {(str(r), int(p)): s for r, p, s in zip(pk.robot, pk["pick"], pk.split)}
    m["split"] = [assign.get((str(r), int(p)), a.default) for r, p in zip(m.robot, m["pick"])]
    m = m[m.split != "drop"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); m.to_csv(a.out, index=False)
    found = {k for k in assign if k in set(zip(m.robot.astype(str), m["pick"].astype(int)))}
    print(f"wrote {a.out}: {len(m)} rows; train {int((m.split == 'train').sum())} / val {int((m.split == 'val').sum())} images; "
          f"{len(found)} of {len(assign)} listed picks present in this manifest")


if __name__ == "__main__":
    main()
