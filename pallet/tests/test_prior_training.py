"""Training priors: rendered from a previous frame of the same pick, corrupted, at the padded size; the model consumes them."""
import numpy as np
import pytest
import torch

from pallet import targets as T
from pallet.dataset import PalletDataset, PriorConfig, load_manifest


def test_training_prior_matches_current_gt_roughly():
    m = load_manifest()
    if len(m) == 0:
        pytest.skip("no pallet data on this machine")
    r = m.iloc[0]
    ds = PalletDataset(picks=[f"{r.robot}/{r.pick}"], max_range_m=12.0, prior=PriorConfig(back=1, rot_deg=0.0, trans_cm=0.0, p_noise_cm=0.0, p_bias_cm=0.0, dropout=0.0))
    s0, s1 = ds[0], ds[len(ds) // 2]
    assert not bool(s0["has_prior"])                                  # first frame of the pick has no previous frame
    assert bool(s1["has_prior"]) and s1["prior"].shape == (6, T.PAD_H, T.PAD_W) and s1["prior_mask"].dtype == torch.bool
    gt = T.decode(s1["target"].numpy(), s1["valid_mask"].numpy()); pr = T.decode(s1["prior"].numpy(), s1["prior_mask"].numpy())
    obj = np.isin(s1["cls"].numpy(), (0, 1)) & np.isfinite(gt["range"]) & np.isfinite(pr["range"])
    assert obj.sum() > 200
    assert np.median(np.linalg.norm((pr["P"] - gt["P"])[:, obj], axis=0)) < 0.03   # uncorrupted prior: pixel-footprint agreement
    real = s1["pixel_mask"].numpy()
    assert np.abs(pr["theta"] - gt["theta"])[real].max() < 1e-4
    assert not s1["prior_mask"][:, ~real].any()
    # corrupted prior stays in a sane range
    ds2 = PalletDataset(picks=[f"{r.robot}/{r.pick}"], max_range_m=12.0, prior=PriorConfig())
    s2 = ds2[len(ds2) // 2]; pr2 = T.decode(s2["prior"].numpy(), s2["prior_mask"].numpy())
    obj2 = obj & np.isfinite(pr2["range"])
    assert 0.01 < np.median(np.linalg.norm((pr2["P"] - gt["P"])[:, obj2], axis=0)) < 0.30
