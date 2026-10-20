"""Fine-tune PalletNet on the pallet manifest (single GPU, bf16 autocast, EMA, cosine LR, tensorboard).

.venv/bin/python -m pallet.train --out runs/b256_baseline            # defaults: B/256 weights, 432x240, batch 8, 1-step
Validation (EMA weights) every --eval_every epochs: metrics from pallet/metrics.py binned by face distance, written to
<out>/val_<epoch>.json and the log, plus prediction-vs-GT panels in <out>/panels/. Resume with --resume.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from . import targets as T
from .dataset import DEFAULT_MANIFEST, PalletDataset, PriorConfig, assign_splits, load_manifest
from .metrics import aggregate, format_table, image_metrics
from .model import PalletNet
from .visualize import panel


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--picks", nargs="*", default=None, help="restrict to picks robot/pick; default all with tensors")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pretrained", default="pretrained/pointditb-256-scenenet-nodinov3-58f0b231.pth")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--model", default="PointDiT-B/16")
    ap.add_argument("--feature_embedding_type", default="dinov3_vitb16")
    ap.add_argument("--steps", type=int, default=1, help="sampling steps at validation")
    ap.add_argument("--zero_t_ratio", type=float, default=0.1, help="1.0 = pure regression at t=0")
    ap.add_argument("--noise_scale", type=float, default=1.0)
    ap.add_argument("--cls_weight", type=float, default=1.0)
    ap.add_argument("--consist_weight", type=float, default=0.0, help="weight of the P-range on-ray consistency loss (metres)")
    ap.add_argument("--channel_weights", type=float, nargs=6, default=None)
    ap.add_argument("--max_range_m", type=float, default=12.0)
    ap.add_argument("--crop", type=int, nargs=2, default=None)
    ap.add_argument("--color_jitter", action="store_true")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--warmup_steps", type=int, default=300)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--eval_every", type=int, default=5, help="epochs")
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--val_max_images", type=int, default=600, help="cap the validation set (even subsample); 0 = all")
    ap.add_argument("--n_panels", type=int, default=6)
    ap.add_argument("--val_raw", action="store_true", help="also validate the raw (non-EMA) weights")
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=None, help="stop early (smoke tests)")
    ap.add_argument("--prior_p", type=float, default=0.0, help="fraction of samples trained from a propagated previous-frame prior")
    ap.add_argument("--prior_back", type=int, default=3)
    ap.add_argument("--prior_rot_deg", type=float, default=1.0)
    ap.add_argument("--prior_trans_cm", type=float, default=5.0)
    ap.add_argument("--prior_noise_cm", type=float, default=4.0)
    ap.add_argument("--prior_bias_cm", type=float, default=4.0)
    ap.add_argument("--prior_dropout", type=float, default=0.2)
    return ap.parse_args()


def lr_at(step, total, args):
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    p = (step - args.warmup_steps) / max(1, total - args.warmup_steps)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * min(1.0, p)))


@torch.no_grad()
def ema_update(ema, model, decay, step):
    decay = min(decay, (1.0 + step) / (10.0 + step))          # warmup: the EMA tracks the model closely at first
    for pe, pm in zip(ema.parameters(), model.parameters()):
        if pm.requires_grad:
            pe.lerp_(pm, 1.0 - decay)


def to_dev(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def pred_panel(sample, x_pred, logits, caption):
    """GT panel over prediction panel for one sample (padded size)."""
    img = (sample["image"].numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
    valid = sample["valid_mask"].numpy()
    gt = {"image": img, "cls": sample["cls"].numpy().astype(np.uint8), **T.decode(sample["target"].numpy(), valid)}
    cls_pred = logits.argmax(0).cpu().numpy().astype(np.uint8)
    pr = {"image": img, "cls": cls_pred, **T.decode(x_pred.cpu().numpy(), np.ones_like(valid))}
    pr["range"] = np.where(sample["pixel_mask"].numpy(), pr["range"], np.nan)
    a, b = panel(gt, "GT  " + caption), panel(pr, "PRED  " + caption)
    from PIL import Image
    sheet = Image.new("RGB", (a.width, a.height + b.height), (25, 25, 25))
    sheet.paste(a, (0, 0)); sheet.paste(b, (0, a.height))
    return sheet


@torch.no_grad()
def validate(ema, val_ds, args, dev, epoch, out: Path, writer, tag="ema"):
    ema.eval()
    loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    rows, done, panel_ids = [], 0, set(np.linspace(0, len(val_ds) - 1, args.n_panels).astype(int).tolist())
    (out / "panels").mkdir(exist_ok=True)
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x, logits = ema.generate(batch["image"].to(dev), steps=args.steps)
        for j in range(x.size(0)):
            s = {k: (v[j] if torch.is_tensor(v) else v[j]) for k, v in batch.items()}
            rows.append(image_metrics(s, x[j], logits[j]))
            if done in panel_ids:
                cap = f"{s['key']} scan {int(s['scan']):04d} face {float(s['face_dist_m']):.2f} m origin err {rows[-1]['origin_err_cm']:.1f} cm"
                pred_panel(s, x[j], logits[j], cap).save(out / "panels" / f"ep{epoch:03d}_{tag}_{done:03d}_{s['key']}_{int(s['scan']):04d}.png")
            done += 1
    agg = aggregate(rows)
    json.dump({"epoch": epoch, "tag": tag, "per_image": rows, "aggregate": agg}, open(out / f"val_{epoch:03d}_{tag}.json", "w"), indent=1)
    print(f"[val epoch {epoch} {tag}]\n" + format_table(agg), flush=True)
    for r in agg:
        for k in ("origin_err_cm", "P_err_cm", "cls_acc", "rot_err_deg"):
            writer.add_scalar(f"val_{tag}/{k}/{r['bin']}", r[k], epoch)
    return agg


def main():
    args = get_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "args.json", "w"), indent=1)
    dev = torch.device("cuda")

    df = assign_splits(load_manifest(args.manifest, args.picks), args.val_frac, args.split_seed)
    df.to_csv(out / "split.csv", index=False)
    prior_cfg = PriorConfig(args.prior_back, args.prior_rot_deg, args.prior_trans_cm, args.prior_noise_cm, args.prior_bias_cm, args.prior_dropout) if args.prior_p > 0 else None
    train_ds = PalletDataset(df[df.split == "train"], crop=args.crop, color_jitter=args.color_jitter, max_range_m=args.max_range_m, prior=prior_cfg)
    val_df = df[df.split == "val"]
    if args.val_max_images and len(val_df) > args.val_max_images:          # even subsample across the val sequences
        val_df = val_df.iloc[np.linspace(0, len(val_df) - 1, args.val_max_images).round().astype(int)]
    val_ds = PalletDataset(val_df, max_range_m=args.max_range_m)
    print(f"train {len(train_ds)} images / {df[df.split == 'train'].groupby(['robot', 'pick']).ngroups} picks, "
          f"val {len(val_ds)} images (of {(df.split == 'val').sum()}) / {val_df.groupby(['robot', 'pick']).ngroups} picks", flush=True)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True,
                        pin_memory=True, persistent_workers=args.num_workers > 0)

    model = PalletNet(model=args.model, feature_embedding_type=args.feature_embedding_type, steps=args.steps,
                      zero_t_ratio=args.zero_t_ratio, noise_scale=args.noise_scale, cls_weight=args.cls_weight,
                      channel_weights=args.channel_weights, consist_weight=args.consist_weight)
    if not args.no_pretrained:
        print("pretrained:", model.load_pretrained(args.pretrained), flush=True)
    model.to(dev)
    ema = deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    params = model.trainable_parameters()
    print(f"trainable params {sum(p.numel() for p in params) / 1e6:.1f} M", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    writer = SummaryWriter(out / "tb")

    step, start_epoch = 0, 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["optimizer"])
        step, start_epoch = ck["step"], ck["epoch"] + 1
        print(f"resumed from {args.resume} at epoch {start_epoch} step {step}", flush=True)
    total = args.epochs * len(loader)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, seen = time.time(), 0
        for batch in loader:
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, args)
            batch = to_dev(batch, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, logs = model(batch, prior_p=args.prior_p)
            if not torch.isfinite(loss):
                print(f"non-finite loss at step {step}, skipping", flush=True); opt.zero_grad(set_to_none=True); step += 1; continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip:
                gn = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            ema_update(ema, model, args.ema_decay, step)
            seen += batch["image"].size(0)
            if step % args.log_every == 0:
                r = logs["rmse_raw"].tolist()
                print(f"ep {epoch} step {step} loss {loss.item():.4f} flow {logs['flow'].item():.4f} ce {logs['ce'].item():.4f} cons {logs['consist'].item():.3f} "
                      f"rmse[m,rad] P({r[0]:.2f},{r[1]:.2f},{r[2]:.2f}) rng {r[3]:.2f} th {r[4]:.3f} ph {r[5]:.3f} "
                      f"lr {opt.param_groups[0]['lr']:.2e} gn {float(gn) if args.grad_clip else 0:.2f} "
                      f"{seen / (time.time() - t0):.1f} img/s mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
                writer.add_scalar("train/loss", loss.item(), step); writer.add_scalar("train/flow", logs["flow"].item(), step)
                writer.add_scalar("train/ce", logs["ce"].item(), step); writer.add_scalar("train/lr", opt.param_groups[0]["lr"], step)
                for c, name in enumerate(T.CHANNELS):
                    writer.add_scalar(f"train/rmse_{name}", r[c], step)
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        ck = {"model": model.state_dict(), "ema": ema.state_dict(), "optimizer": opt.state_dict(), "step": step, "epoch": epoch, "args": vars(args)}
        torch.save(ck, out / "checkpoint-last.pth")
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs or (args.max_steps and step >= args.max_steps):
            validate(ema, val_ds, args, dev, epoch, out, writer, "ema")
            if args.val_raw:
                validate(model, val_ds, args, dev, epoch, out, writer, "raw")
        if args.max_steps and step >= args.max_steps:
            break
    writer.close()


if __name__ == "__main__":
    main()
