"""PalletDataset: forkview image + 7-channel pallet targets from the SensorUnit manifest (images.parquet).

One sample (torch tensors, H=432, W=240 after padding; see targets.py):
  image         f32 [3,H,W] in [0,1], upright RGB (raw PNG rotated by rot_code), padding rows black
  target        f32 [6,H,W] encoded (px,py,pz,range,theta,phi); 0 where invalid
  valid_mask    bool [6,H,W] per-channel validity (P/range: hit pixels; theta/phi: every real pixel)
  cls           i64 [H,W] 0 pallet 1 load 2 ground 3 background, 255 ignore (padding)
  pixel_mask    bool [H,W] real (non-padded) pixels
  sky_mask      bool [H,W] all False (kept so engine.py's batch contract holds)
  pallet_P_cam  f32 [12] R row-major then p: X_pallet = R X_cam + p
  cam_in_pallet f32 [3]
  intrinsics    f32 [4] fx, fy, cx, cy of the padded (and cropped) image
  face_dist_m   f32 scalar;  key str;  scan int;  dataset_name "pallet"
No horizontal flip (it would break the pallet frame), no per-frame centring or scaling. Optional random crop
(crop=(h,w), multiples of 16) shifts cx/cy exactly. Optional colour jitter on the image only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import targets as T

# Manifest built against the image snapshot (~/Anantak/Data/PalletTrainImages): the raw PicksCopies folders are a rolling
# window that the SensorUnit rebuild job deletes from, the snapshot is ours and stable.
DEFAULT_MANIFEST = Path("~/Anantak/Data/PalletTrainImages/manifest/images.parquet").expanduser()


@dataclass
class ManifestRow:
    key: str
    robot: str
    pick: int
    scan: int
    image_abs: str
    tensor: str
    rot_code: int
    face_dist_m: float
    kind_id: int


def load_manifest(path=DEFAULT_MANIFEST, picks=None, split=None) -> pd.DataFrame:
    """Rows whose ray tensor and image both exist on disk now, optionally restricted to picks ["249/1697", ...] or a split name."""
    m = pd.read_parquet(path)
    m = m[m["tensor"].notna() & (m["tensor"] != "") & m["image_on_disk"].astype(bool) & ~m["not_a_pick"].astype(bool)]
    m = m[[Path(a).exists() and Path(b).exists() for a, b in zip(m["image_abs"], m["tensor"])]]
    if picks is not None:
        want = {tuple(p.split("/")) for p in picks}
        m = m[[(str(r), str(p)) in want for r, p in zip(m["robot"], m["pick"])]]
    if split is not None:
        m = m[m["split"] == split]
    return m.sort_values(["robot", "pick", "scan"]).reset_index(drop=True)


def assign_splits(m: pd.DataFrame, val_frac=0.2, seed=0) -> pd.DataFrame:
    """Fill the empty `split` column: by pick (never split a sequence), stratified by kind_id where kinds are known
    (unknown kind = its own group), topped up with random picks so that round(val_frac * n_picks) picks are val. Returns a copy."""
    m = m.copy()
    picks = m.drop_duplicates(["robot", "pick"])[["robot", "pick", "kind_id"]].copy()
    picks["kind_id"] = picks["kind_id"].fillna(-1)
    rng = np.random.default_rng(seed)
    val = set()
    for _, g in picks.groupby("kind_id"):
        idx = rng.permutation(len(g))
        n_val = int(round(val_frac * len(g))) if len(g) > 1 else 0
        val.update(map(tuple, g.iloc[idx[:n_val]][["robot", "pick"]].values.tolist()))
    want = int(round(val_frac * len(picks)))
    rest = [tuple(x) for x in picks[["robot", "pick"]].values.tolist() if tuple(x) not in val]
    for k in rng.permutation(len(rest))[: max(0, want - len(val))]:
        val.add(rest[k])
    m["split"] = ["val" if (r, p) in val else "train" for r, p in zip(m["robot"], m["pick"])]
    return m


def camera_from_manifest(root: Path) -> dict:
    """cameras.json of the pick that owns a tensor path (<model>/rays/cam2_NNNN.npz)."""
    import json
    return json.load(open(root / "cameras.json"))


class PalletDataset(Dataset):
    def __init__(self, manifest=DEFAULT_MANIFEST, picks=None, split=None, crop=None, color_jitter=False,
                 max_range_m=None, pad=True):
        self.df = load_manifest(manifest, picks, split) if not isinstance(manifest, pd.DataFrame) else manifest.reset_index(drop=True)
        self.crop = tuple(crop) if crop else None
        if self.crop:
            assert self.crop[0] % 16 == 0 and self.crop[1] % 16 == 0, "crop must be a multiple of the 16 px patch"
        self.max_range_m = max_range_m
        self.pad = pad
        self._cams = {}
        self.jitter = None
        if color_jitter:
            import torchvision.transforms as tv
            self.jitter = tv.ColorJitter(0.1, 0.1, 0.1, 0.05)

    def __len__(self):
        return len(self.df)

    def camera(self, tensor_path: str) -> dict:
        rays = str(Path(tensor_path).parent)
        if rays not in self._cams:
            self._cams[rays] = camera_from_manifest(Path(rays))
        return self._cams[rays]

    def load_raw(self, i: int):
        """Un-padded numpy pieces of sample i: image u8 [3,424,240], npz dict, intrinsics (fx,fy,cx,cy), row."""
        row = self.df.iloc[i]
        cam = self.camera(row["tensor"])
        raw = np.asarray(Image.open(row["image_abs"]).convert("RGB"))
        img = T.upright(raw, int(row["rot_code"])).transpose(2, 0, 1)        # CHW u8
        npz = dict(np.load(row["tensor"]))
        fx, fy, cx, cy = cam["f_px"], cam["f_px"], cam["cx"], cam["cy"]
        if "P" not in npz:                                    # compact record (pallet/compact.py): rebuild P/theta/phi from the rays
            npz.update(T.reconstruct(npz["cls"], npz["range"], npz["pallet_P_cam"], fx, fy, cx, cy))
        assert img.shape[1:] == npz["cls"].shape == (T.RAW_H, T.RAW_W), (img.shape, npz["cls"].shape)
        return img, npz, (fx, fy, cx, cy), row

    def __getitem__(self, i: int) -> dict:
        img, npz, (fx, fy, cx, cy), row = self.load_raw(i)
        cont, valid, cls = T.encode(npz)
        if self.max_range_m is not None:                      # do not supervise P/range on far, noisy hits
            far = np.nan_to_num(npz["range"], nan=np.inf) > self.max_range_m
            valid[T.HIT_CHANNELS][:, far] = False
            cont[T.HIT_CHANNELS][:, far] = 0.0
        pixel = np.ones(cls.shape, bool)
        if self.pad:
            img, cont, valid, cls, pixel = T.pad(img), T.pad(cont), T.pad(valid), T.pad(cls, T.CLS_IGNORE), T.pad(pixel)
            fx, fy, cx, cy = T.pad_intrinsics(fx, fy, cx, cy)
        if self.crop:
            H, W = cls.shape
            ch, cw = self.crop
            y0 = np.random.randint(0, H - ch + 1)
            x0 = np.random.randint(0, W - cw + 1)
            sl = (slice(y0, y0 + ch), slice(x0, x0 + cw))
            img, cont, valid, cls, pixel = img[:, sl[0], sl[1]], cont[:, sl[0], sl[1]], valid[:, sl[0], sl[1]], cls[sl], pixel[sl]
            cx, cy = cx - x0, cy - y0
        image = torch.from_numpy(np.ascontiguousarray(img)).float() / 255.0
        if self.jitter is not None:
            image = self.jitter(image)
        return {
            "image": image,
            "target": torch.from_numpy(np.ascontiguousarray(cont)),
            "valid_mask": torch.from_numpy(np.ascontiguousarray(valid)),
            "cls": torch.from_numpy(np.ascontiguousarray(cls)),
            "pixel_mask": torch.from_numpy(np.ascontiguousarray(pixel)),
            "sky_mask": torch.zeros(cls.shape, dtype=torch.bool),
            "pallet_P_cam": torch.from_numpy(np.asarray(npz["pallet_P_cam"], np.float32)),
            "cam_in_pallet": torch.from_numpy(np.asarray(npz["cam_in_pallet"], np.float32)),
            "intrinsics": torch.tensor([fx, fy, cx, cy], dtype=torch.float32),
            "face_dist_m": torch.tensor(float(row["face_dist_m"])),
            "key": str(row["key"]), "scan": int(row["scan"]), "dataset_name": "pallet",
        }


def decode_sample(s: dict) -> dict:
    """Sample -> un-padded numpy {"cls","range","P","theta","phi","image"} comparable with the npz (NaN where invalid)."""
    cont, valid, cls = s["target"].numpy(), s["valid_mask"].numpy(), s["cls"].numpy()
    img = (s["image"].numpy() * 255).round().astype(np.uint8)
    if cls.shape == (T.PAD_H, T.PAD_W):
        cont, valid, cls, img = T.unpad(cont), T.unpad(valid), T.unpad(cls), T.unpad(img)
    out = T.decode(cont, valid)
    out["cls"] = cls.astype(np.uint8)
    out["image"] = img.transpose(1, 2, 0)
    return out
