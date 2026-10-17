"""Pose voting recovers the ground-truth pose exactly from ground-truth targets."""
import numpy as np
import pytest

from pallet import targets as T
from pallet.dataset import PalletDataset, load_manifest
from pallet.pose import estimate_pose, rotation_error_deg, dirs_from_angles


@pytest.fixture(scope="module")
def ds():
    m = load_manifest()
    if len(m) == 0:
        pytest.skip("no pallet data on this machine")
    r = m.iloc[0]
    return PalletDataset(picks=[f"{r.robot}/{r.pick}"])


def test_dirs_from_angles_roundtrip():
    th, ph = np.random.uniform(0.1, 3.0, 100), np.random.uniform(-3.1, 3.1, 100)
    d = dirs_from_angles(th, ph)
    assert np.allclose(np.linalg.norm(d, axis=1), 1)
    assert np.allclose(np.arccos(d[:, 1]), th) and np.allclose(np.arctan2(d[:, 2], d[:, 0]), ph)


def test_gt_targets_give_gt_pose(ds):
    for j in (0, len(ds) // 2, len(ds) - 1):
        s = ds[j]
        H, W = s["cls"].shape
        fx, fy, cx, cy = s["intrinsics"].tolist()
        rays = T.pixel_rays(fx, fy, cx, cy, H, W)
        gt = T.decode(s["target"].numpy(), s["valid_mask"].numpy())
        cls, real = s["cls"].numpy(), s["pixel_mask"].numpy()
        obj = np.isin(cls, (T.CLS_PALLET, T.CLS_LOAD)) & np.isfinite(gt["range"])
        est = estimate_pose(gt["P"], gt["range"], gt["theta"], gt["phi"], rays, obj, real)
        R_gt = s["pallet_P_cam"][:9].numpy().reshape(3, 3)
        assert rotation_error_deg(est["R"], R_gt) < 1e-3      # R_gt is float32 (6 digits)
        assert np.linalg.norm(est["c"] - s["cam_in_pallet"].numpy()) < 1e-3
        assert est["spread"] < 1e-3 and est["n_votes"] == obj.sum()
