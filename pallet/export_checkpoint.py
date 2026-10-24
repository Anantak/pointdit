"""Export an inference-only PalletDiT checkpoint for other units: EMA weights without the gated DINOv3 encoder, fp16, plus the
training args and provenance. Load with PalletNet.load_inference(path) (DINOv3 comes from DINOV3_WEIGHTS_DIR as usual).

python -m pallet.export_checkpoint --run runs/b256_prior2 --out pallet/models/palletdit-b256-prior2-nodinov3-fp16.pth
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch


def export(run: Path, out: Path, dtype=torch.float16, keep_encoder=False, note="") -> dict:
    ck = torch.load(run / "checkpoint-last.pth", map_location="cpu", weights_only=False)
    sd = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in ck["ema"].items()
          if keep_encoder or not k.startswith("net.y_embedder.")}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=run.parent.parent, text=True).strip()
    except Exception:
        commit = "unknown"
    meta = {"name": out.stem, "source_run": str(run), "source_step": ck.get("step"), "source_epoch": ck.get("epoch"),
            "weights": "EMA", "dtype": str(dtype).replace("torch.", ""), "dinov3_stripped": not keep_encoder, "note": note,
            "feature_embedding_type": ck["args"]["feature_embedding_type"], "model": ck["args"]["model"],
            "input": "RGB 432x240 (424x240 upright + 4 pad rows top/bottom), [0,1]", "channels": "px,py,pz,range,theta,phi + 4 class logits",
            "target_offset": [0, 0, 0, 0, 1.5707963, 1.5707963], "target_scale": [2, 2, 4, 4, 0.5, 0.5],
            "git_commit": commit, "train_args": ck["args"]}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": sd, "meta": meta}, out)
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    meta["sha256"], meta["bytes"] = h, out.stat().st_size
    json.dump(meta, open(out.with_suffix(".json"), "w"), indent=1)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--keep-encoder", action="store_true", help="include net.y_embedder (needed for runs whose encoder is NOT the DINOv3 release)")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    meta = export(Path(a.run), Path(a.out), torch.float32 if a.fp32 else torch.float16, a.keep_encoder, a.note)
    print(json.dumps({k: v for k, v in meta.items() if k != "train_args"}, indent=1))


if __name__ == "__main__":
    main()
