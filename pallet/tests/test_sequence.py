"""The propagated prior of a GT frame matches the next frame's GT where both are defined."""
import numpy as np
import pytest

from pallet import targets as T
from pallet.dataset import PalletDataset, load_manifest
from pallet.sequence import propagate_pose, relative_motion, render_prior


def test_oracle_prior_matches_next_frame():
    m = load_manifest()
    if len(m) == 0:
        pytest.skip("no pallet data on this machine")
    r = m.iloc[0]; ds = PalletDataset(picks=[f"{r.robot}/{r.pick}"])
    k = len(ds) // 2
    a, b = ds[k], ds[k + 1]
    H, W = a["cls"].shape; intr = b["intrinsics"].tolist(); real = a["pixel_mask"].numpy()
    pa, pb = a["pallet_P_cam"].numpy().astype(np.float64), b["pallet_P_cam"].numpy().astype(np.float64)
    dT = relative_motion(pa, pb)
    R1, c1 = propagate_pose(pa[:9].reshape(3, 3), pa[9:], dT)
    assert np.allclose(R1, pb[:9].reshape(3, 3), atol=1e-5) and np.allclose(c1, pb[9:], atol=1e-5)   # pose propagation is exact
    ga = T.decode(a["target"].numpy(), a["valid_mask"].numpy()); gb = T.decode(b["target"].numpy(), b["valid_mask"].numpy())
    cont, mask, cls_p = render_prior(ga["P"], ga["range"], a["cls"].numpy(), np.isfinite(ga["range"]) & real, R1, c1, intr, H, W)
    pr = T.decode(cont, mask)
    realb = b["pixel_mask"].numpy()
    assert np.abs(pr["theta"] - gb["theta"])[realb].max() < 1e-4 and np.abs(pr["phi"] - gb["phi"])[realb].max() < 1e-4
    obj = np.isin(b["cls"].numpy(), (T.CLS_PALLET, T.CLS_LOAD)) & np.isfinite(pr["range"]) & np.isfinite(gb["range"])
    assert obj.sum() > 200
    err = np.linalg.norm((pr["P"] - gb["P"])[:, obj], axis=0)
    assert np.median(err) < 0.03, np.median(err)              # pixel-footprint level agreement on pallet/load
    gnd = (b["cls"].numpy() == T.CLS_GROUND) & np.isfinite(pr["range"]) & np.isfinite(gb["range"]) & (gb["range"] < 8)
    assert np.median(np.abs(pr["range"] - gb["range"])[gnd]) < 0.05
