"""GPU smoke test of PalletNet with the B/256 weights: load, one training step, one generate. Skipped without them."""
import os
from pathlib import Path

import pytest
import torch

from pallet import targets as T
from pallet.dataset import PalletDataset, load_manifest
from pallet.model import PalletNet

CKPT = Path(os.environ.get("PALLET_CKPT", Path(__file__).resolve().parents[2] / "pretrained" / "pointditb-256-scenenet-nodinov3-58f0b231.pth"))


@pytest.mark.skipif(not (torch.cuda.is_available() and CKPT.exists()), reason="needs a GPU and the B/256 checkpoint")
def test_load_train_step_generate():
    m = load_manifest()
    if len(m) == 0:
        pytest.skip("no pallet data on this machine")
    ds = PalletDataset(picks=[f"{m.iloc[0].robot}/{m.iloc[0].pick}"])
    net = PalletNet()
    info = net.load_pretrained(str(CKPT))
    assert set(info["dropped"]) == {"net.x_embedder.proj1.weight", "net.final_layer.linear.weight", "net.final_layer.linear.bias"}
    assert all(not p.requires_grad for p in net.net.y_embedder.parameters())
    net = net.cuda()
    batch = torch.utils.data.default_collate([ds[0], ds[1]])
    batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
    opt = torch.optim.AdamW(net.trainable_parameters(), lr=1e-4)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, logs = net(batch)
    assert torch.isfinite(loss)
    loss.backward()
    opt.step()
    net.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        x, logits = net.generate(batch["image"])
    assert x.shape == (2, T.NUM_CONT, T.PAD_H, T.PAD_W) and logits.shape == (2, T.NUM_CLASSES, T.PAD_H, T.PAD_W)
    assert torch.isfinite(x).all() and torch.isfinite(logits).all()
