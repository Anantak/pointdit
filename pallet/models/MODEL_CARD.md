> **2026-10-24: no checkpoint committed yet.** The export described below was withdrawn: the runs up to prior2 were trained with a
> frozen RANDOM DINOv3 encoder (PointDiT's `initialize_weights` wiped the hub-loaded weights; fixed in `PalletNet.reload_dinov3`),
> so an encoder-stripped export does not reproduce them. Re-export with `--keep-encoder`, or wait for the corrected series. See the
> handover note of 2026-10-24.

# PalletDiT model files

Inference-only checkpoints for other units. Each `.pth` (Git LFS) holds `{"model": state_dict, "meta": {...}}` with the EMA
weights of a `pallet.model.PalletNet`, **without the DINOv3 encoder** (gated upstream weights; every unit already has
`dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` under `TensorRT-Models/models/dinov3/weights`, pointed to by `DINOV3_WEIGHTS_DIR`).
The `.json` next to it repeats the metadata, the sha256 and the byte size. Full training checkpoints (model + EMA + optimizer,
2.85 GB each) stay off git: `runs/<run>/checkpoint-last.pth` on ana-ml-1 and ana-ml-02.

## palletdit-b256-prior2-nodinov3-fp16

| | |
|---|---|
| architecture | PointDiT-B/16 (12 blocks, width 768, patch 16), 6 input channels, 6 + 4 output channels; frozen DINOv3-B/16 conditioning (4 layers) |
| input | upright forkview RGB 432x240 in [0,1] (424x240 rotated upright by `rot_code 2`, 4 black rows padded top and bottom; cy += 4) |
| output | per pixel: encoded (px, py, pz, range, theta, phi) and 4 class logits (pallet, load, ground, background) |
| decoding | raw = enc * (2, 2, 4, 4, 0.5, 0.5) + (0, 0, 0, 0, pi/2, pi/2); P in metres in the pallet frame, range in metres, angles in radians |
| pose | R by Kabsch from (theta, phi) over all real pixels; camera centre = median over pallet/load pixels of P - range * R d; pallet origin in the camera frame = -R^T c (`pallet/pose.py`) |
| training | from the public PointDiT-B/256 weights through runs 2-6 (data up to 2,030 picks), then prior-conditioned fine-tune with pose corruption 2 deg / 8 cm and the on-ray consistency loss (0.5), 3 epochs; EMA weights at step 15,521 |
| accuracy (201 picks never trained on, medians) | lateral offset from the centreline 3.3 / 4.2 / 6.7 / 9.8 / 12.2 / 17.3 cm at 0-2 / 2-3 / 3-4.5 / 4.5-6 / 6-8 / 8-12 m; depth 1.6 / 2.1 / 2.8 / 4.4 / 6.5 / 11.5 cm; yaw ~0.9 deg (label-limited, see the paper); pallet centre in the camera frame 1.5-3.3 cm sideways, bearing 0.2-0.5 deg; class accuracy 0.98 |
| sequential use | `pallet.sequence` recipe: warm start from the previous frame's own maps and fused pose, t0 0.8 fading from 3 m to 0 at 6 m, pose weight 0.2, one Euler step |
| provenance | exported by `pallet/export_checkpoint.py` from `runs/b256_prior2` (git `a7fff92` tree, 2026-10-24); sha256 and size in the `.json` |
| precision | fp16 weights; on one image the decoded P differs from the fp32 EMA by < 1 mm and the pose by < 0.1 cm |

### Loading

```python
import os; os.environ["DINOV3_REPO"] = ".../third_party/dinov3"; os.environ["DINOV3_WEIGHTS_DIR"] = ".../pretrained/dinov3"
from pallet.model import PalletNet
from pallet import targets as T
net, meta = PalletNet.load_inference("pallet/models/palletdit-b256-prior2-nodinov3-fp16.pth", device="cuda")
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    x, logits = net.generate(image[None].cuda(), steps=1)          # image: [3,432,240] float in [0,1]
dec = T.decode(x[0].float().cpu().numpy(), np.ones((6, 432, 240), bool))   # P [3,H,W], range, theta, phi
```

Then `pallet.pose.estimate_pose(dec["P"], dec["range"], dec["theta"], dec["phi"], T.pixel_rays(fx, fy, cx, cy + 4, 432, 240),
vote_mask=(logits.argmax(1)[0] <= 1), ray_mask=real_pixels)`. The full pipeline, metrics and the sequential estimator are in
`pallet/` and documented in `docs/PALLET_CODE_GUIDE.md`; results and limits in `docs/PALLET_POINTDIT_PAPER.md`.

### Known limits

* The pallet heading (yaw) is accurate to ~1 deg and this is set by a per-pick bias of the training labels, not by the network;
  lateral error at distance is this heading error times the distance. The pallet centre's bearing and lateral position in the
  camera frame do not depend on it and are accurate to 2-3 cm at all distances.
* Trained on one site (Van Buren), two forklifts (249, 250), daytime indoor lighting; no evaluation elsewhere yet.
* 1-step inference only; multi-step sampling degrades the angles.
