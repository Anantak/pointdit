# Pallet pose from forkview images with PointDiT — handover to start the work

Written for: Claude Code (and Manuj) in a fresh session, probably opened in this repo. State as of 2026-10-15.
The data side of this project lives in `~/Anantak/SensorUnit` (branch `THD`); read its `docs/PALLET_DATA_BUILDING_PROCESS.md`
sections "Per-pixel ray tensors" and "Training manifest" when you need the data details beyond what is here.

## Goal

Train a PointDiT-style network that predicts, for every pixel of a forklift's forkview image, **7 numbers**:

| channel | meaning | valid where |
|---|---|---|
| `cls` (uint8 → 4-way) | 0 pallet, 1 load, 2 ground, 3 background | everywhere |
| `range` (m) | ray length from the camera centre to what the ray hits | hit with known range (else NaN) |
| `P` (3 floats, m) | the hit point in the **pallet frame** | hit (else NaN) |
| `theta` (rad) | angle between the ray and gravity-down | everywhere |
| `phi` (rad) | angle between the pallet x axis and the ray projected on the ground plane, `atan2(d_z, d_x)` in the pallet frame (a ray straight into the face has phi = +pi/2) | everywhere |

Pallet frame: origin at the centre of the pallet's front face on the ground, x lateral, **y down**, z into the pallet
(gravity-down is exactly +y). Purpose: every pallet/load pixel then votes for the pallet pose (predicted `P` + the pixel's known
ray → camera-to-pallet translation), votes are fused over the approach sequence so the unit knows where the pallet is from far
away. Later channels may be added (load type, pallet type, quality, expected weight). Model architecture: PointDiT (this repo).

## Decisions already taken with the user (2026-10-14)

- Do the work **in this fork, on a branch `pallet-pose`**, with a `pallet/` package (dataset adapter, targets, metrics, pose
  voting, export). SensorUnit stays C++ and only produces data + manifest. (A separate repo was considered and rejected for now.)
- Targets are metric, in the pallet frame: **no per-frame centring/scaling** (the STUDY_GUIDE already flags that normalisation
  as wrong for metric use), **no horizontal flip** (it would break the pallet-frame labels), random crop only.
- 6 continuous channels (`P` 3, `range`, `theta`, `phi`) under the masked flow/MSE loss; the **class as a separate 4-way
  cross-entropy head** (an integer under MSE is wrong). `P`/`range` masked to hit pixels; angles valid everywhere.
- Image size 240x424 must become **240x432** (patch 16): pad 8 px (decide left/right or symmetric, keep it consistent).
- Start from the public **B/256 checkpoint**, frozen DINOv3-B, **1-step mode** (effectively a regressor), fine-tune on our data.
  Baseline first, as the study guide advises; diffusion multi-step and bigger models later if the baseline is limited.
- The real metric is the **pallet origin per image from pixel voting** vs ground truth, as a function of distance, not the
  per-pixel MSE.

## The data

Produced by `~/Anantak/SensorUnit/scripts/utils/pallet_rays.py` (rendered from the lidar patch cloud, the XT32 submap and the
ground plane; box reprojection verified to 0.05 px against the pipeline's own portrait corners). Per pick (model dir
`PalletModels/<robot>/<GGGG>/`):
- `rays/cameras.json`: `camera_index` (2 = forkview, the only calibrated camera), `rot_code` (2: raw 424x240 PNG → upright
  240x424 by `np.rot90(raw, 1)`), `image_width/height`, pinhole `f_px, cx, cy` (f ≈ 322, no distortion), `cam_P_jt`,
  `W_P_pallet`, and `images[]`: `{scan, image ("02/color_NNNN.png", relative to the raw episode dir), ts_img_us, face_dist_m,
  cam_in_pallet [x,y,z], cam_P_W {R row-major 9, p 3}, pallet_P_cam {R, p}}`. `pallet_P_cam` maps camera → pallet
  (X_pallet = R X_cam + p) and is the pose target.
- `rays/cam2_NNNN.npz`: `cls` u8 [424,240], `range` f32, `P` f32 [3,424,240], `theta` f32, `phi` f32, `pallet_P_cam` f32[12]
  (R then p), `cam_in_pallet` f32[3], `scan`, `ts_img_us`. Invalid = NaN. Pixel (u,v) ray in the camera: ((u+0.5-cx)/f,
  (v+0.5-cy)/f, 1) normalised, camera x right, y down, z forward.
- `rays/background_points.npz`, `rays/panels/` (inspection PNGs; `pallet_rays.py --panels` makes them: photo + pallet-frame
  position colour on a fixed metre scale R = x -0.75..0.75, G = height 0..2, B = depth 0..1.3, range, theta, phi, class masks).
- Raw images: `PicksCopies/<robot>/<GGGG>/02/color_NNNN.png` (424x240 RGBA, landscape) + `color_NNNN.txt` (timestamp).
- Manifest: `~/Anantak/SensorUnit/scripts/utils/pallet_manifest.py` → `PalletModels/manifest/images.parquet` (one row per
  image: `key, robot, pick, scan, image, image_abs, image_on_disk, rot_code, ts_img_us, face_dist_m, cam_in_pallet_*,
  pallet_P_cam_R0..R8, pallet_P_cam_p0..p2, tensor, kind_id, not_a_pick, px_pallet, px_load, px_ground, px_bg_range,
  px_bg_none, range_known_frac, min_range_m, split`) and `picks.parquet` (pallet geometry, kind, flags, paths). `split` is empty:
  split by pick, stratified by kind, is the plan.

Where the data is:
- `ana-ml-1` (this laptop, 16 GB RTX): 24 picks with tensors (~100 MB each) under `~/Anantak/Data/PalletModels/{249,250}/<GGGG>/rays/`
  — e.g. 249/1697 (50 images, 10.6 → 1.45 m), 250/1248 (salt bags), 250/1224 (boxes), 250/1246 (tall drums), 249/1720,
  249/1740, 250/1271; manifest at `~/Anantak/Data/PalletModels/manifest/` (2,861 picks, 892 images with tensors on 10-14).
- `ana-ml-02` (ubuntu@192.168.5.193, 8 GB RTX 5060, ssh with `-o IdentitiesOnly=yes -i ~/.ssh/id_rsa`): the full history is
  being rebuilt on the external drive `/media/ubuntu/ana-back-01/Work/Data/` (≈ 1,750 + 1,275 picks, ~45 images each,
  2–3 days from 2026-10-14 evening); copy what you need from there (read-only), ~100 MB per pick.
- Kinds (77 visual classes, human-reviewed): `~/Anantak/Data/Posters/classes/classes.json` on ana-ml-1.

## PointDiT facts that matter (this repo, verified in the code on 2026-10-14)

- Image in [0,1] from the loader, rescaled to [-1,1] in `engine.py:97`, then to ImageNet mean/std for DINOv3 in `model.py:574-578`.
- Conditioning: frozen DINOv3 (`model.py:325-328`; ViT-B/16, L/16 or H+/16), 4 intermediate layers concatenated
  (`model.py:333-338, 587-592`); image tokens and point tokens concatenated along channels and projected (`concat_proj`, ~`model.py:615`).
- Patch 16; H and W must be divisible by 16 (`model.py:67`). Output at full resolution (`FinalLayer`, `model.py:216, 533`).
- Target: 3-channel XYZ in the camera frame, back-projected from z-depth (`dataloader/img_depth_intrinsics.py:~3830`),
  centred on the valid mean and divided by the mean distance (`:3933-3975`, flags in `main.py:153-160`). Intrinsics are only
  used by the loader.
- Rectified flow, x-prediction, logit-normal t (`denoiser.py:56-59`), v-space loss (`:97-103`); deterministic Euler from zero
  noise (`generate()` `:159`, `generate_noise_scale=0`, `main.py:85`); 1 step for B/256, 3 for L/512 and H/512; DINOv3
  features cached across steps (`:196-200`).
- Loss: masked per-pixel MSE over the channels (`denoiser.py:139`), sky weight 0.01, `rel_point_loss` weight 0.1
  (`engine.py:121-140`, `loss.py`). Invalid → 0 (`:3858`), `valid_mask` = z > 1e-6 (`:3924`), `--noise_fill_invalid`
  (`denoiser.py:89-93`). Augmentations: ColorJitter (`:616`), blur/noise, horizontal flip negating X (`:3852-3855`), random crop.
- Datasets: RGB + depth `.npy` (float32 m) + `.npz` with `intrinsics`; add a dataset by a `dataset_name` branch in `__getitem__`
  (copy the `eden` branch at `:3758`) and an entry in `dataloader/configs/*.yaml` (`mixed_dataset.py:211`).
- Checkpoints: B/L/H at 256 and 512 on Hugging Face `haofeixu/pointdit` (25 GB for all six); DINOv3 weights are gated and not
  included — we have `dinov3_vitb16_pretrain_lvd1689m` and `vits16` in `~/Anantak/TensorRT-Models/models/dinov3/weights/`
  (put them in `pretrained/dinov3` or `DINOV3_WEIGHTS_DIR`). Depth/width: B 12/768, L 24/1024, H 32/1280 (`model.py:667-676`).
- To change for 7 channels: `in_channels=3` (`denoiser.py:30`) → `out_channels` (`model.py:277`); `randn(bsz,3,…)`
  (`denoiser.py:166`); mask `.repeat(1,3,…)` (`:115,116,139,142`); per-channel weights (`:103`); `rel_point_loss` (`loss.py:18`);
  the final layer. Pretrained `x_embedder`/`final_layer` will not load with C ≠ 3 → `--no_strict_load` (`main.py:150`).
- Environment: Python 3.12, torch 2.7.0 + cu128 (driver 580 on both machines is fine), `pip install -r requirements.txt`,
  DINOv3 code via torch.hub from a local checkout (`third_party/dinov3`; `~/Anantak/TensorRT-Models/third_party/dinov3`
  already exists). Make a venv separate from the python3.11 one used by SensorUnit scripts.
- Original training: 16–64 GPUs, bf16; we fine-tune. On the 16 GB laptop B/256 at 240x432 should take batch ~8 in bf16.

## Plan (agreed), each step with a visible result

1. Branch `pallet-pose`; `pallet/targets.py`: the 7-channel target definition above, padding 240x424 → 240x432, masks.
2. `pallet/dataset.py`: reads `images.parquet` (+ npz + raw image with the upright rotation), returns image, target, masks,
   `pallet_P_cam`, intrinsics; unit test on one pick (decode → re-encode == npz); visualise with the same colour scheme as the
   panels (port `colourise()` / `panel()` from `pallet_rays.py`).
3. Model changes (6 continuous channels + class head), baseline fine-tune from B/256 on the picks on this laptop, then on the
   full set from ml-02. Metrics: class accuracy, `P` error (cm) on pallet/load pixels, and **pallet-origin error from pixel
   voting vs `cam_in_pallet`, binned by face distance**. Keep `theta`/`phi` as auxiliary outputs (they are pure camera-pose
   geometry; a sanity check of what the net learns).
4. Sequence fusion of the per-image votes along the approach (with inter-frame odometry), then TensorRT export for the unit
   (the DINOv3 encoder already runs there via `TensorRT-Models`; export conventions in its `docs/ADDING_A_MODEL.md`).

## Working rules the user holds to

Propose code changes and wait for the go-ahead before editing; `python3.11` for SensorUnit scripts, a 3.12 venv for this repo;
never kill processes by pattern; long jobs with `setsid nohup ... &`; keep notes for the next session in this file (newest first).

## Prompt to start the new session

> Read `docs/PALLET_POSE_HANDOVER.md` in this repo (`~/Libraries/GoogleResearch/pointdit`), then `docs/STUDY_GUIDE.md`.
> We are adapting PointDiT to predict the 7 per-pixel pallet channels described there, on the branch `pallet-pose`. Start with
> steps 1 and 2 of the plan on the 24 picks under `~/Anantak/Data/PalletModels` on this laptop: set up the Python 3.12 venv
> per the README (torch 2.7 cu128, DINOv3 weights from `~/Anantak/TensorRT-Models`), create the branch and the `pallet/`
> package, write the target definition and the dataset adapter against `~/Anantak/Data/PalletModels/manifest/images.parquet`,
> and show me decoded targets rendered with the pallet-frame colour scheme before touching the model. Propose each code change
> before making it.
