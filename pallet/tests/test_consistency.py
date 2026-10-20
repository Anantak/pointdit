"""The on-ray consistency term is ~0 for ground-truth targets and positive for a shifted P."""
import numpy as np
import pytest
import torch

from pallet.dataset import PalletDataset, load_manifest
from pallet.model import PalletNet


def test_consistency_zero_on_gt():
    m = load_manifest()
    if len(m) == 0:
        pytest.skip("no pallet data on this machine")
    r = m.iloc[0]; ds = PalletDataset(picks=[f"{r.robot}/{r.pick}"], max_range_m=12.0)
    batch = torch.utils.data.default_collate([ds[0], ds[len(ds) // 2]])
    net = PalletNet.__new__(PalletNet)                      # no weights needed for the static geometry
    c = PalletNet.consistency(net, batch["target"], batch, batch["valid_mask"])
    assert float(c) < 0.002, float(c)                        # GT P lies on the ray at the GT range (< 2 mm)
    shifted = batch["target"].clone(); shifted[:, 0] += 0.05  # +10 cm in x (scale 2 m)
    assert abs(float(PalletNet.consistency(net, shifted, batch, batch["valid_mask"])) - 0.10) < 0.01
