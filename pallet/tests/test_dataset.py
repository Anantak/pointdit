"""Round-trip and geometry checks on one pick (needs ~/Anantak/Data/PalletModels on this machine).  pytest pallet/tests"""
import numpy as np
import pytest
import torch

from pallet import targets as T
from pallet.dataset import PalletDataset, decode_sample, load_manifest, assign_splits



@pytest.fixture(scope="module")
def ds():
    m = load_manifest()
    if len(m) == 0:
        pytest.skip("no pallet data on this machine")
    r = m.iloc[0]
    return PalletDataset(picks=[f"{r.robot}/{r.pick}"])


def test_shapes_and_dtypes(ds):
    s = ds[0]
    assert s["image"].shape == (3, T.PAD_H, T.PAD_W) and s["image"].dtype == torch.float32
    assert 0.0 <= s["image"].min() and s["image"].max() <= 1.0
    assert s["target"].shape == (6, T.PAD_H, T.PAD_W) and s["valid_mask"].shape == (6, T.PAD_H, T.PAD_W)
    assert s["cls"].shape == (T.PAD_H, T.PAD_W) and s["cls"].dtype == torch.int64
    assert T.PAD_H % 16 == 0 and T.PAD_W % 16 == 0
    assert torch.isfinite(s["target"]).all()
    pad_rows = ~s["pixel_mask"]
    assert pad_rows.sum() == (T.PAD_TOP + T.PAD_BOTTOM) * T.PAD_W
    assert (s["cls"][pad_rows] == T.CLS_IGNORE).all() and not s["valid_mask"][:, pad_rows].any()
    assert (s["image"][:, pad_rows] == 0).all()
    assert s["valid_mask"][T.RAY_CHANNELS][:, s["pixel_mask"]].all()          # theta/phi valid on every real pixel
    assert (s["valid_mask"][0] == s["valid_mask"][3]).all()                     # P valid exactly where range is


def test_roundtrip_equals_npz(ds):
    for j in (0, len(ds) // 2, len(ds) - 1):
        s = ds[j]
        img, npz, _, _ = ds.load_raw(j)
        dec = decode_sample(s)
        assert np.array_equal(dec["cls"], npz["cls"])
        assert np.array_equal(dec["image"], img.transpose(1, 2, 0))
        for k in ("range", "P", "theta", "phi"):
            assert np.array_equal(np.isnan(dec[k]), np.isnan(npz[k])), k
            assert np.allclose(np.nan_to_num(dec[k]), np.nan_to_num(npz[k]), atol=1e-6), k


def test_targets_consistent_with_rays_and_pose(ds):
    """P = cam + range * R d for every non-background hit (<1 mm); theta/phi are the ray angles (<1e-5 rad)."""
    for j in (0, len(ds) - 1):
        s = ds[j]
        fx, fy, cx, cy = s["intrinsics"].tolist()
        rays = T.pixel_rays(fx, fy, cx, cy, T.PAD_H, T.PAD_W)                   # padded intrinsics, padded grid
        dec = T.decode(s["target"].numpy(), s["valid_mask"].numpy())
        want = T.targets_from_pose(rays, np.nan_to_num(dec["range"]), s["pallet_P_cam"].numpy())
        cls = s["cls"].numpy()
        hit = np.isfinite(dec["range"]) & np.isin(cls, (T.CLS_PALLET, T.CLS_LOAD, T.CLS_GROUND))
        err = np.linalg.norm((want["P"] - dec["P"]).transpose(1, 2, 0)[hit], axis=-1)
        assert err.max() < 1e-3, err.max()
        real = s["pixel_mask"].numpy()
        assert np.abs(want["theta"] - dec["theta"])[real].max() < 1e-5
        assert np.abs(want["phi"] - dec["phi"])[real].max() < 1e-5
        # camera centre from one pallet pixel's vote must equal cam_in_pallet (the pose-voting identity)
        R = s["pallet_P_cam"][:9].numpy().reshape(3, 3)
        v, u = np.argwhere(hit & (cls == T.CLS_PALLET))[0]
        vote = dec["P"][:, v, u] - dec["range"][v, u] * (R @ rays[v, u])
        assert np.allclose(vote, s["cam_in_pallet"].numpy(), atol=1e-3)


def test_crop_adjusts_intrinsics(ds):
    r = ds.df.iloc[0]
    d = PalletDataset(picks=[f"{r.robot}/{r.pick}"], crop=(400, 240))
    s = d[0]
    assert s["image"].shape[-2:] == (400, 240) and s["cls"].shape == (400, 240)
    fx, fy, cx, cy = s["intrinsics"].tolist()
    rays = T.pixel_rays(fx, fy, cx, cy, 400, 240)
    dec = T.decode(s["target"].numpy(), s["valid_mask"].numpy())
    want = T.targets_from_pose(rays, np.nan_to_num(dec["range"]), s["pallet_P_cam"].numpy())
    real = s["pixel_mask"].numpy()
    assert np.abs(want["theta"] - dec["theta"])[real].max() < 1e-5


def test_split_by_pick():
    m = assign_splits(load_manifest(), val_frac=0.25, seed=0)
    per_pick = m.groupby(["robot", "pick"])["split"].nunique()
    assert (per_pick == 1).all()
    assert set(m["split"]) == {"train", "val"}
