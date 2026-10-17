"""Evaluate a pallet/train.py checkpoint on its own validation split with a given number of sampling steps.

python -m pallet.evaluate --run runs/b256_run3 [--steps 1 3 5] [--hybrid 3] [--rot-from all|obj] [--val_max_images 600] [--raw]
--hybrid k: angles + class logits from the 1-step pass, P/range from a k-step pass (3 steps sharpen P, 1 step is better for angles).
Uses <run>/split.csv (val rows), the EMA weights of <run>/checkpoint-last.pth, writes <run>/eval_steps<k>.json and prints the tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import PalletDataset
from .metrics import aggregate, format_table, image_metrics
from .model import PalletNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--val_max_images", type=int, default=600)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--raw", action="store_true", help="use the raw weights instead of the EMA")
    ap.add_argument("--hybrid", type=int, default=0, help="k > 0: also evaluate the hybrid 1-step angles + k-step P/range")
    ap.add_argument("--rot-from", default="all", choices=["all", "obj"])
    ap.add_argument("--split-from", default=None, help="use another run's split.csv (compare checkpoints on the same images)")
    a = ap.parse_args()
    run = Path(a.run)
    args = json.load(open(run / "args.json"))
    df = pd.read_csv(Path(a.split_from or run) / "split.csv", dtype={"robot": str})
    val = df[df.split == "val"]
    if a.val_max_images and len(val) > a.val_max_images:
        val = val.iloc[np.linspace(0, len(val) - 1, a.val_max_images).round().astype(int)]
    ds = PalletDataset(val, max_range_m=args.get("max_range_m"))
    ck = torch.load(run / "checkpoint-last.pth", map_location="cpu", weights_only=False)
    net = PalletNet(model=args["model"], feature_embedding_type=args["feature_embedding_type"])
    net.load_state_dict(ck["model" if a.raw else "ema"])
    net = net.cuda().eval()
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers)
    configs = [("steps%d" % k, k, 0) for k in a.steps] + ([("hybrid%d" % a.hybrid, 1, a.hybrid)] if a.hybrid else [])
    for name, steps, k_hybrid in configs:
        rows = []
        with torch.no_grad():
            for batch in loader:
                img = batch["image"].cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    x, logits = net.generate(img, steps=steps)
                    if k_hybrid:
                        xk, _ = net.generate(img, steps=k_hybrid)
                        x = torch.cat([xk[:, :4], x[:, 4:]], 1)          # P/range from k steps, theta/phi from 1 step
                for j in range(x.size(0)):
                    s = {k: (v[j] if torch.is_tensor(v) else v[j]) for k, v in batch.items()}
                    rows.append(image_metrics(s, x[j], logits[j], rot_from=a.rot_from))
        agg = aggregate(rows)
        tag = ("raw" if a.raw else "ema") + ("" if a.rot_from == "all" else "_rotobj") + (f"_split-{Path(a.split_from).name}" if a.split_from else "")
        json.dump({"config": name, "tag": tag, "per_image": rows, "aggregate": agg}, open(run / f"eval_{name}_{tag}.json", "w"), indent=1)
        print(f"[{run.name} {tag} {name} on {len(ds)} val images]\n" + format_table(agg), flush=True)


if __name__ == "__main__":
    main()
