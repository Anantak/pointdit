"""The DINOv3 encoder inside PalletNet must carry the release weights (PointDiT's initialize_weights wipes them; we reload)."""
import os
from pathlib import Path

import pytest
import torch

from pallet.model import PalletNet


def test_encoder_matches_release_file():
    d = Path(os.environ.get("DINOV3_WEIGHTS_DIR", Path(__file__).resolve().parents[2] / "pretrained" / "dinov3"))
    f = d / PalletNet.DINOV3_FILES["dinov3_vitb16"]
    if not f.is_file():
        pytest.skip("DINOv3 weights not available")
    net = PalletNet()
    assert net.dinov3_loaded
    raw = torch.load(f, map_location="cpu")
    sd = net.net.y_embedder.state_dict()
    for k in ("blocks.5.attn.proj.weight", "blocks.0.attn.qkv.weight", "patch_embed.proj.weight"):
        assert torch.equal(sd[k], raw[k]), k
