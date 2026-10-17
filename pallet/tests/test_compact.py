"""Compact records reproduce the full ray tensors through the dataset."""
import numpy as np
import pytest

from pallet import targets as T
from pallet.compact import compact_pick
from pallet.dataset import PalletDataset, load_manifest, decode_sample


def test_compact_roundtrip(tmp_path):
    m = load_manifest()
    if len(m) == 0:
        pytest.skip("no pallet data on this machine")
    r = m.iloc[0]
    src = __import__("pathlib").Path(r.tensor).parent
    res = compact_pick(src, tmp_path / "rays", delete_src=False)
    assert res["n"] > 0 and not res["bad"]
    # a manifest row pointing at the compact file must decode to the same targets (P within 1 mm on non-background hits)
    row = m[(m.robot == r.robot) & (m.pick == r.pick)].iloc[:3].copy()
    row["tensor"] = [str(tmp_path / "rays" / __import__("pathlib").Path(t).name) for t in row.tensor]
    ds_c = PalletDataset(row); ds_f = PalletDataset(m[(m.robot == r.robot) & (m.pick == r.pick)].iloc[:3])
    for j in range(len(ds_c)):
        a, b = decode_sample(ds_c[j]), decode_sample(ds_f[j])
        assert np.array_equal(a["cls"], b["cls"]) and np.array_equal(np.isnan(a["range"]), np.isnan(b["range"]))
        assert np.allclose(np.nan_to_num(a["range"]), np.nan_to_num(b["range"]))
        obj = np.isfinite(b["range"]) & np.isin(b["cls"], (0, 1, 2))
        assert np.linalg.norm((a["P"] - b["P"])[:, obj], axis=0).max() < 1e-3
        assert np.abs(a["theta"] - b["theta"]).max() < 1e-4 and np.abs(a["phi"] - b["phi"]).max() < 1e-4
    sz_c = sum(f.stat().st_size for f in (tmp_path / "rays").glob("cam2_*.npz")); sz_f = sum(f.stat().st_size for f in src.glob("cam2_*.npz"))
    if "P" in np.load(next(src.glob("cam2_*.npz"))).files:          # only meaningful when the source holds full tensors
        assert sz_c < 0.4 * sz_f, (sz_c, sz_f)
