"""Near-real-time fit statistics from the training logs: train losses and per-channel RMSE per step, validation
pallet-origin error per face-distance bin per epoch. Reads runs/<run>.log and runs/<run>/val_*_ema.json.

python -m pallet.plot_runs --runs b256_baseline b256_run2 b256_run3 --out runs/fit_stats.png [--loop 300]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]   # fixed categorical order
LINE = re.compile(r"^ep (\d+) step (\d+) loss ([\d.]+) flow ([\d.]+) ce ([\d.]+) rmse\[m,rad\] P\(([\d.]+),([\d.]+),([\d.]+)\) rng ([\d.]+) th ([\d.]+) ph ([\d.]+)")
BINS = ["0-2 m", "2-3 m", "3-4.5 m", "4.5-6 m", "6-8 m"]


def read_train(log: Path):
    rows = [LINE.match(l) for l in open(log, errors="ignore")]
    rows = [tuple(float(x) for x in m.groups()) for m in rows if m]
    return np.array(rows) if rows else np.zeros((0, 11))


def read_val(run_dir: Path, tag="ema"):
    out = {}
    for f in sorted(glob.glob(str(run_dir / f"val_*_{tag}.json"))) or sorted(glob.glob(str(run_dir / "val_*.json"))):
        d = json.load(open(f))
        out[d["epoch"]] = {r["bin"]: r for r in d["aggregate"]}
    return out


def smooth(y, k=10):
    if len(y) < k:
        return y
    c = np.cumsum(np.insert(y, 0, 0.0))
    s = (c[k:] - c[:-k]) / k
    return np.concatenate([y[: k - 1], s])


def render(runs, root: Path, out: Path):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5), facecolor="#fcfcfb")
    for a in ax:
        a.set_facecolor("#fcfcfb"); a.grid(True, color="#e6e5e1", lw=0.6); a.spines[["top", "right"]].set_visible(False)
    cur = runs[-1]
    # A: train losses per run (step within run)
    for i, r in enumerate(runs):
        t = read_train(root / f"{r}.log")
        if len(t) == 0:
            continue
        ax[0].plot(t[:, 1], smooth(t[:, 3]), color=COL[i], lw=1.6, label=f"{r} flow")
        ax[0].plot(t[:, 1], smooth(t[:, 4]), color=COL[i], lw=1.6, ls=":", label=f"{r} class CE")
    ax[0].set_yscale("log"); ax[0].set_xlabel("step"); ax[0].set_title("train loss (10-pt mean): flow solid, class CE dotted", loc="left", fontsize=10)
    ax[0].legend(fontsize=7, frameon=False)
    # B: per-channel train RMSE of the current run
    t = read_train(root / f"{cur}.log")
    names = ["P x (m)", "P y (m)", "P z (m)", "range (m)", "theta (rad)", "phi (rad)"]
    for j, nm in enumerate(names):
        if len(t):
            ax[1].plot(t[:, 1], smooth(t[:, 5 + j]), color=COL[j], lw=1.6, label=nm)
    ax[1].set_yscale("log"); ax[1].set_xlabel("step"); ax[1].set_title(f"{cur}: train RMSE per channel (valid pixels, raw units)", loc="left", fontsize=10)
    ax[1].legend(fontsize=7, frameon=False, ncol=2)
    # C: validation origin error per bin per epoch, current run solid, previous run dashed
    for k, r in enumerate(runs[-2:]):
        v = read_val(root / r)
        if not v:
            continue
        ep = sorted(v)
        for j, b in enumerate(BINS):
            y = [v[e][b]["origin_err_cm"] if b in v[e] else np.nan for e in ep]
            ax[2].plot([e + 1 for e in ep], y, color=COL[j], lw=1.6, ls="-" if r == cur else "--", marker="o", ms=3,
                       label=f"{b}" if r == cur else None)
            if r == cur and len(ep):
                ax[2].annotate(f"{y[-1]:.0f}", (ep[-1] + 1, y[-1]), xytext=(4, 0), textcoords="offset points", fontsize=7, color="#52514e", va="center")
    ax[2].set_ylim(0, 60); ax[2].set_xlabel("epoch"); ax[2].set_ylabel("cm")
    ax[2].set_title(f"val pallet-origin error (median, EMA): {cur} solid, {runs[-2] if len(runs) > 1 else ''} dashed", loc="left", fontsize=10)
    ax[2].legend(title="face distance", fontsize=7, title_fontsize=8, frameon=False)
    last = t[-1] if len(t) else None
    sub = f"updated {time.strftime('%H:%M')}" + (f" | {cur} at epoch {int(last[0])} step {int(last[1])} loss {last[2]:.4f}" if last is not None else "")
    fig.suptitle(sub, fontsize=9, color="#52514e", x=0.01, ha="left")
    fig.tight_layout()
    tmp = out.with_suffix(".tmp.png"); fig.savefig(tmp, dpi=110); tmp.rename(out); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["b256_baseline", "b256_run2", "b256_run3"])
    ap.add_argument("--root", default="runs")
    ap.add_argument("--out", default="runs/fit_stats.png")
    ap.add_argument("--loop", type=int, default=0, help="seconds between refreshes; 0 = once")
    a = ap.parse_args()
    while True:
        render(a.runs, Path(a.root), Path(a.out))
        if not a.loop:
            break
        time.sleep(a.loop)


if __name__ == "__main__":
    main()
