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

## Session notes (newest first)

### 2026-10-20 19:00 — the model is named **PalletDiT**; work continues on ml-02

- User: name = PalletDiT. Docs renamed (paper title, code guide), paper section VIII "Status and current work" added, PDFs regenerated.
- Running on ml-02: yaw fusion along the approach (`pallet.sequence --sources pred --pose_alpha 0.2 / 0.1 / 0.05`, 60 clean picks,
  prior2), `runs/yawfuse_a0*` + `runs/yawfuse_chain.log` there. New metrics: lateral/depth/yaw (+ fused_*) in `metrics.py`.
- Laptop GPU reserved for the user's map building.

### 2026-10-20 14:05 — PAUSED (user needs the GPU for map building); how to resume

- **Resume prior2** (3rd of 3 epochs, ~1 h at 19 img/s; same flags as the original launch, plus `--resume`):
  ```
  cd ~/Libraries/GoogleResearch/pointdit && export DINOV3_REPO=$PWD/third_party/dinov3 DINOV3_WEIGHTS_DIR=$PWD/pretrained/dinov3
  setsid nohup .venv/bin/python -m pallet.train --out runs/b256_prior2 --resume runs/b256_prior2/checkpoint-last.pth \
    --pretrained runs/b256_run6/checkpoint-last.pth --batch_size 16 --lr 3e-5 --warmup_steps 300 --epochs 3 --eval_every 1 \
    --prior_p 0.5 --prior_rot_deg 2.0 --prior_trans_cm 8.0 --consist_weight 0.5 --num_workers 8 >> runs/b256_prior2.log 2>&1 < /dev/null &
  ```
  Checkpoint verified: keys model/ema/optimizer/step/epoch, epoch index 1 saved (step 7,760), resume continues at epoch 2.
- **Then run, in order** (each one command; see `pallet/scripts/day2_2026-10-20.sh` and `day3_2026-10-20.sh` for the exact lines):
  1. `pallet.sequence --run runs/b256_prior2 --picks 30 --sources pred pred_gtpose oracle --t0 0.2 0.35 0.5 0.8 --steps 1
     --pose_alpha 0.3 --consistency_cm 0 --splat 3 --prior_fill noise --out runs/seq_eval_prior2`
  2. `pallet.evaluate --run runs/b256_prior{1,2} --steps 1 --split-from runs/fixed_val --val_max_images 1500`
  3. the independent-lineage model (`day3` script without its wait line): `runs/b256_indep` from the public B/256 weights,
     2 epochs, then the per-pick yaw correlation with run 6 (label-bias test).
- **State of results**: best single-frame = prior2 after 2 epochs (8.16 cm overall on its split; run 6 10.2); best sequence =
  prior1 own maps + own pose 9.5 cm (GT pose 5.5; oracle 4.1); label yaw bias ~1.3 deg/pick established (lineage caveat pending
  the indep model). Paper section VII and `runs/yaw_bias_per_pick.csv` are current; PDFs regenerated 13:59.
- Running on CPU only: tensorboard :6006, `plot_runs --loop` (runs/fit_stats.png), pull_loop (ml-02 -> store), compact_loop.
- **18:36 LABEL YAW BIAS CONFIRMED (ml-02 chain done)**: `runs/b256_indep` (public B/256 weights, no lineage with runs 3-6, 2 epochs,
  only 16.0 cm overall) makes the SAME per-pick yaw errors as run 6: **r = 0.89 over 177 clean picks** (std 1.52 / 1.34 deg).
  Initialisation lineage is excluded; what the models share is the training labels. => the pipeline's pallet heading is off by
  ~1.3 deg (std) per pick; lateral error at distance is this bias times distance. Top action (data side): verify/correct the
  heading in the SensorUnit pallet fit, regenerate poses, re-evaluate; the model is likely better laterally than measured.
  Final numbers of the day on the clean split (origin all / lateral 0-2, 2-3, 3-4.5 m / yaw): run 6 10.5 / 3.5, 5.2, 6.8 / 1.07;
  prior2 ep3 8.0 / 3.0, 4.3, 6.7 / 1.05; indep 16.0 / - / 1.28. Checkpoints: prior2 ep3 on ml-02 (copied to the laptop 18:40).
- **17:20 prior2 epoch 3 done on ml-02 (16:25)**: clean split origin 8.0 cm overall (ep2 8.4); lateral per bin 3.0 / 4.3 / 6.7 / 9.2 /
  13.4 / 15.6 cm; yaw 1.05 deg; P 7.6. Final prior2 checkpoint lives on ml-02 (`runs/b256_prior2/checkpoint-last.pth` there;
  copy back). indep model training (epoch 1 of 2 at 17:16, 20 img/s), done ~18:40 + eval + yaw test.
- **15:10 user priority: LATERAL alignment from afar (10 m) matters most; depth can lean on lidar; final assessment by lidar.**
  Clean-split medians (prior2): lateral |x| 3.0 / 4.4 / 6.9 / 9.7 / 13.1 / 13.5 cm per bin, depth |z| 1.7 / 2.2 / 2.8 / 4.6 / 7.2 /
  12.4 cm, |yaw| ~0.9 deg everywhere. **Signed lateral error = -(distance x signed yaw error) with r = -0.98 (3-12 m)**: the lateral
  offset from the centreline is entirely the yaw estimate; depth is already small. => yaw is THE target for the product metric:
  resolve the label-heading question (indep model tonight), then yaw-focused training (phi weight, multi-frame yaw consistency
  within a pick), yaw fusion over the approach (removes the 0.64 deg scatter, not the bias), far-field levers. New headline
  metrics to add to evaluate/aggregate: lateral error, depth error, yaw error by distance (currently derivable from
  origin_err_xyz_cm / rot_err_vec_deg in the JSONs).
- **14:45 results from ml-02 (clean fixed_val split, 201 picks, 1,499 images; origin cm 0-2 / 2-3 / 3-4.5 / 4.5-6 / 6-8 / 8-12 / all):**
  run 6 5.8 / 7.9 / 11.1 / 15.4 / 19.7 / 30.6 / 10.5 | prior1 5.6 / 7.3 / 10.2 / 14.2 / 21.0 / 30.5 / 9.9 |
  **prior2 (2 epochs) 4.3 / 5.8 / 8.5 / 12.3 / 16.8 / 24.8 / 8.4**; P 8.3 / 7.6 / 7.7; GT-mask floor 6.8 / 6.2 / 4.7; rot 1.07 / 1.06 / 1.04.
  The consistency loss (+ stronger prior corruption) cut the origin error by 20 % at every distance without changing P or the
  rotation: the votes became self-consistent. **prior2 is the new best single-frame model.** prior2 sequence eval (30 picks,
  `runs/seq_eval_prior2`): single 8.4; own maps + own pose 8.0-8.3 (jitter 4.1 -> 2.8); + GT pose 4.4-5.0; oracle 3.2-4.0.
- **14:25 deferred work moved to ml-02** (its GPU freed; build done at 3,296 picks): `pallet/scripts/ml02_chain_2026-10-20.sh`
  running there (log `~/Libraries/GoogleResearch/pointdit/runs/ml02_chain_2026-10-20.log` on ml-02): manifest refresh ->
  `pallet/make_split.py` builds `runs/fixed_val/split.csv` and `runs/b256_prior2/split.csv` from the pick lists in
  `pallet/splits/` (same picks as the laptop, ml-02 paths) -> prior2 sequence eval -> fixed_val evals (prior1/prior2/run6) ->
  prior2's 3rd epoch resumed at batch 8 / lr 1.5e-5 (`--split_csv`, `runs/b256_prior2_ml02.log`) -> `runs/b256_indep` (public
  weights, 2 epochs, batch 8) -> indep eval + yaw correlation with run 6. Results land on ml-02; copy `runs/*` back
  (rsync) when done. `train.py --split_csv` and `pallet/make_split.py` are new (uncommitted).

### 2026-10-20 — sequential estimation (phase 1: warm-started sampling from a propagated prior)

- **Sweep 1 result (30 picks, 839 frames, origin cm overall; `runs/seq_eval/summary.csv`):** single 10.9 | oracle (GT maps + GT pose)
  5.3 / 4.2 / 3.8 / 3.6 at t0 0.2 / 0.35 / 0.5 / 0.8, rot 0.2-0.4 deg, far bins 4-7 cm | gtcur 5.1 -> 3.1 | oracle maps + own pose
  10.7-13.8 (no gain: the net copies the 1-deg pose error through the angle prior) | own maps + GT pose 12.5-16.5 with P error
  40-110 cm (my rendering splatted every predicted pixel incl. background/ground with poor P in front of the pallet) | own maps +
  own pose 21-65 (both feedbacks). 3 steps worse than 1 everywhere (far bins explode). => mechanism works given a good prior;
  the own-prediction prior needs (a) class + P-range consistency filtering of what is splatted, (b) a smoothed pose.
- Sweep 2 launched 08:20 (`runs/seq_eval2`): consistency filter 5 cm, pallet/load only + analytic ground, pose EMA alpha 0.3,
  pose-only (angles) priors, t0 {0.2, 0.35, 0.5}, 1 step.
- Sweep 2/3 (own maps + GT pose still P 70-136 cm, far bins better, near worse; angles-only prior catastrophic with zero fill):
  two faults found. (1) `generate` filled unknown prior entries with 0 = "point at the pallet origin"; now `prior_fill=noise`
  (training distribution; `mixed`/`zeros` kept). (2) The 5 cm P-range consistency filter kept 0 % of predicted pallet pixels:
  **predicted P and predicted range disagree along the ray by ~15 cm median** (a finding: the two channels are not mutually
  consistent; the vote uses both). With no pallet pixels, the analytic ground plane filled the pallet region ("ground 1 m
  behind") and the net copied it (P displaced along the ray, origin error unaffected). Fixes: consistency filter off, splat 3 px,
  ground drawn only where visible (z < 0 in front of the face or |x| > 0.9 m). Own-map prior now covers 95 % of the next
  frame's pallet region at 6.7 cm median P error (diagnostic on 249/109). Sweep 4 launched 09:05 (`runs/seq_eval4`).
- Smoothed pose alone (EMA alpha 0.3, motion-compensated, no prior): 9.8 vs 10.9 cm single -> the pose-level filter is worth ~1 cm by itself.
- **Sweep 3 (noise fill, consistency filter still on; `runs/seq_eval3`)**: oracle 6.2 / 4.7 / 4.2 cm at t0 0.2 / 0.35 / 0.5 (zero fill
  gave 5.3 / 4.2 / 3.8: the zero-noise mean is slightly better for KNOWN entries -> `prior_fill=mixed` is the candidate);
  angles-only prior still catastrophic (57-122 cm) even with noise-filled P/range. Reading: in training, noise in z at t > 0 only
  ever marked UNSUPERVISED pixels, so the net never learned to predict P from the image where z is noise at intermediate t;
  partial priors therefore wreck the uncovered pixels (pred_gtpose P 86-146 cm). Exactly what prior-conditioned training
  (loss on all valid pixels while z carries a partial prior) should fix.
- **Sweep 4 (class-only filter, splat 3, visible ground, noise fill; `runs/seq_eval4`) — the maps prior works:** own maps + GT pose,
  1 step: 8.2 / 6.8 / **6.0 cm overall at t0 0.2 / 0.35 / 0.5** (single 10.9); per bin at t0 0.5: 4.8 / 6.4 / 7.3 / 7.1 / 5.4 / 6.6
  (single 5.6 / 8.2 / 12.2 / 15.4 / 21.6 / 26.8); P 6.6 (8.0); rot 0.41 deg (from the GT-pose angle prior). Own maps + own
  smoothed pose: 11.5-13.3 (no gain, worse with t0): **the pose prior is the bottleneck** (the net copies the ~1.2 deg own
  rotation through theta/phi). Angles-only priors catastrophic as before. Sweep 5 launched 09:15 (`runs/seq_eval5{a,b,c}`):
  maps-only prior (theta/phi left to the image) and `mixed` fill, own pose vs GT pose, alongside the prior1 training.
- **Sweep 5 (`runs/seq_eval5{a,b,c}`)**: maps-only prior (angles left to the image) is WORSE than the full prior: own maps + GT pose
  9.7-10.8 (full 6.0), own maps + own pose 15-17 (single 10.9), rotation 0.93-1.46 deg -> with noise in the angle channels at t0
  the net's angle output degrades (same gap: noise in z was never "predict from the image" in training). `mixed` fill is
  slightly better than noise fill (own maps + GT pose 5.7 vs 6.0; + own pose 12.8 vs 13.3). Pose smoothing (EMA) does not reduce
  the rotation error (stays ~1.2 deg): the per-frame rotation error is systematic per pick, not noise. => Everything now hinges
  on the prior-conditioned training making the net correct, rather than copy, a prior with ~1 deg / ~5 cm pose error.
- **prior1 result (`runs/b256_prior1`, 3 epochs, 11:40; `runs/seq_eval_prior1`)**: single-frame on the 30 picks 10.2 cm (run 6: 10.9;
  P 7.3 vs 8.0) -> the prior training also helped single-frame. **Own maps + own smoothed pose: 10.0 / 9.7 / 9.5 / 9.5 cm at t0
  0.2 / 0.35 / 0.5 / 0.8 (run 6: 11.5-13.3)**: the net no longer copies the pose error, but the gain is small because the rotation
  stays 1.1-1.2 deg; jitter 5.4 -> 3.7. Own maps + GT pose 7.9 / 6.5 / 5.9 / 5.5; oracle 6.3 / 5.0 / 4.6 / 4.1. Single-frame
  training val per epoch 10.6 / 9.8 / 9.8. => The remaining lever is the ROTATION: ~1.1 deg systematic per pick.
- **Rotation error anatomy (run 6, fixed_val, 1,499 images, 177 picks; `metrics` now stores `rot_err_vec_deg`)**: per-pick |mean
  rotation error| median 0.95 deg vs within-pick frame-to-frame std 0.64 deg; global mean ~0 (0.16 deg); per-pick mean vector
  std across picks (x, y, z) = (0.27, 1.35, 0.43) deg -> **a per-pick YAW bias about the pallet y (vertical) axis of ~1 deg**,
  62 % of picks above 0.7 deg. 1 deg yaw at 3 m = 5 cm lateral origin error (the lateral-dominated residual seen before).
  Label (pipeline pallet heading) or model? Next experiment: fit one yaw per pick from trajectory/odometry consistency
  (`pallet/yawfit.py`); if the fit recovers the GT-measured bias, it is model-side and sequence fusion removes it.
- **Yaw fit: two null results (`pallet/yawfit.py`, `runs/yawfit_run6{,_spread}`)**. (1) Trajectory/odometry consistency is INVARIANT
  to a global yaw (it rotates the voted and the propagated trajectory together): fit correlation 0.08 with the GT-measured bias,
  values at the grid bounds. (2) Vote dispersion: correlation 0.05, no change. => The per-pick yaw bias is self-consistent inside
  the prediction (P and angles agree with each other); it is either a per-pick LABEL bias of the pallet heading (zero global
  mean, per-pick constant: exactly what a network trained across picks would show against biased labels) or a perception bias
  that moves P and angles together. Discriminating needs an independent heading reference (raw lidar re-fit near the face, or
  the unit's lidar pallet detector). Consequence: sequence fusion alone cannot remove it; with an external yaw the warm start
  reaches 5.5 cm overall and 8 cm at 6-8 m (GT-pose rows).
- **LABEL YAW BIAS, established (14:10)**: per-pick mean yaw error on fixed_val (177 picks) is the SAME across runs 3/4/5/6/prior1
  (pairwise r = 0.92-0.99, std ~1.35-1.55 deg) and the same near (<3 m) and far (3-8 m) within a pick (r = 0.80). Different
  models would not share a perception bias this exactly: **the pallet heading in the labels (pipeline pallet fit) has a per-pick
  yaw error of ~1.3 deg std**. Consequences: (1) the measured near-field floor (4-5 cm with GT mask + GT R) is largely label error;
  the network is probably more accurate w.r.t. the physical pallet than the tables say; (2) the "GT pose" sequence gains are
  measured against the same biased heading; (3) top priority for the data side: fix/verify the pallet heading in the SensorUnit
  pallet fit (re-fit the face plane from the near-frame lidar, check against the patch cloud), then regenerate targets
  (cheap: compact records only need the pose; `targets.reconstruct` does the rest) and re-evaluate. Per-pick yaw errors are in
  `runs/*/eval_steps1_ema_split-fixed_val.json` (`rot_err_vec_deg[1]`), usable to find the worst picks for inspection.
- Caveat on the label-bias inference: runs 3->4->5->6->prior1 are one LINEAGE (each initialised from the previous), so a shared
  perception bias inherited through initialisation is not excluded. Third chain `pallet/scripts/day3_2026-10-20.sh` (after chain 2):
  `runs/b256_indep` from the PUBLIC B/256 weights (2 epochs, split seed 7), then per-pick yaw correlation with run 6 on fixed_val
  (printed in `runs/day3_2026-10-20.log`). r ~0.9+ confirms the labels; much lower points at the lineage.
  Per-pick table for the data side: `runs/yaw_bias_per_pick.csv` (177 picks, worst first; 68 picks > 1 deg, 21 > 2 deg, top ones
  4-5 deg: 250/1046, 250/617, 250/747, 250/485, 249/1281 ...).
- **13:58 GPU handed to the user (map building).** prior2 stopped after 2 of 3 epochs with its checkpoint saved
  (`runs/b256_prior2/checkpoint-last.pth`, EMA): single-frame val epoch 1 / 2 = 8.88 / **8.16 cm** overall (prior1 10.6 / 9.8,
  run 6 10.2 on this split), P 7.6, consistency term 15 -> 5 cm. Chains 2 and 3 killed before their GPU stages: STILL TO RUN when
  the GPU is free: `pallet.sequence --run runs/b256_prior2 ...` (same flags as prior1's), `pallet.evaluate --split-from
  runs/fixed_val` for prior1/prior2, and the independent-lineage model (`pallet/scripts/day3_2026-10-20.sh`, remove its wait line).
  tensorboard (:6006) and the plot loop keep running (CPU only).
- **Phase 3 code (prior-conditioned training)**: `PalletDataset(prior=PriorConfig(...))` renders, per sample, a prior from a
  previous frame (1-3 back) of the same pick: GT pallet/load pixels + visible ground, pose corrupted (rot ~N(0,1 deg),
  trans ~N(0,5 cm)), P noise 4 cm + per-frame bias 4 cm, 16-px block dropout 20 %; `PalletNet.forward(batch, prior_p)` noises the
  prior instead of the target for a fraction prior_p of samples (loss unchanged); `train.py --prior_p ... --prior_*`;
  `tests/test_prior_training.py`. Smoke: loss settles in ~30 steps, ~9 img/s at batch 8 while sharing the GPU.
- **On-ray consistency loss** (`PalletNet.consistency`, `--consist_weight`): mean |P_pred - (c_gt + r_pred * R_gt d)| over
  supervised hit pixels, metres; test `tests/test_consistency.py` (GT gives < 2 mm). Motivated by the 15 cm P-vs-range
  disagreement; not yet used by prior1.
- Second chain `pallet/scripts/day2_2026-10-20.sh` (log `runs/day2_2026-10-20.log`): after prior1's eval -> `runs/b256_prior2`
  (from run 6, prior_p 0.5, pose corruption 2 deg / 8 cm, consist_weight 0.5, 3 epochs) -> `runs/seq_eval_prior2` -> fixed_val
  single-frame evals of prior1 and prior2 (`runs/b256_prior*/eval_fixed_val.log`). Paper: draft section VII written (sweeps 1-5).
- Day chain `pallet/scripts/day_2026-10-20.sh` (log `runs/day_2026-10-20.log`): after sweeps 3/4 -> `runs/b256_prior1`
  (from run 6, prior_p 0.5, 3 epochs, lr 3e-5, ~4-5 h) -> `runs/seq_eval_prior1` (sources pred / pred_gtpose / oracle, t0
  0.2-0.8, 1 step, consistency off, splat 3, noise fill). Compare with `runs/seq_eval4` (run 6, same settings).

- Idea (user): along an approach, start each frame's denoising from the previous frame's estimate carried by the known camera
  motion instead of from zero. Possible because the flow objective trained the net at every t. Code: `PalletNet.generate(...,
  x_prior, prior_mask, t0, prior_noise)` (z0 = t0 * prior, zero-noise mean by default; unknown entries 0), `pallet/sequence.py`
  (pose propagation R' = R A^T, c' = c - R' b; exact theta/phi prior from the propagated pose; P/range prior by z-buffer
  re-rendering of the previous frame's hit pixels + analytic ground plane; sources pred / pred_gtpose / oracle_predpose / oracle /
  gtcur; metrics + frame-to-frame jitter), `tests/test_sequence.py` (oracle prior matches the next frame's GT: angles 1e-4, P < 3 cm).
- Smoke (2 picks, t0 0.5, 1 step): single 12.7 cm overall; oracle prior 4.8 cm (far bins 6 cm!); GT-current-at-t0 5.1; **own
  prediction as prior 37 cm (P 102 cm): error feedback / drift lock-in**. Sweep launched 07:30 on 30 fixed_val picks (839 frames):
  sources x t0 {0.2, 0.35, 0.5, 0.8} x steps {1, 3 at 0.35/0.8} -> `runs/seq_eval/summary.csv`, log `runs/seq_eval.log`.
- ml-02: repo + venv + B/256 + run 6 checkpoint being set up by `scratchpad/setup_ml02.sh` (log in the scratchpad) at
  `~/Libraries/GoogleResearch/pointdit` on ml-02; its GPU (8 GB) is still used by the build pipeline.

### 2026-10-19 15:45 — clean held-out comparison, documents regenerated

- Cross-run comparisons on another run's split are biased (later runs trained on earlier val picks; the other session noted it on
  10-17). Built `runs/fixed_val/split.csv`: 201 picks (5,854 images; 84 % robot 250) never in any run's train set. `evaluate.py
  --split-from runs/fixed_val`, 600 images, origin cm per bin (0-2 / 2-3 / 3-4.5 / 4.5-6 / 6-8 / 8-12 / all), P, GT-mask floor:
  run 3 6.9/10.4/12.8/19.5/25.1/38.2/12.5, P 10.2, 9.3 | run 4 6.6/9.5/12.1/18.2/22.2/35.8/12.0, P 9.5, 8.2 |
  run 5 6.1/8.9/11.4/15.7/21.7/28.6/11.1, P 8.6, 7.3 | **run 6 5.5/7.7/11.2/16.5/21.2/33.6/11.1, P 8.3, 6.7**. Near field improves
  steadily with data (not flat: the earlier "saturation" was split drift); mid field flat after run 5; far bins too few to rank.
  Use `runs/fixed_val` for every future comparison (and keep those 201 picks out of training).
- Paper corrected (abstract, results, discussion), code guide refreshed; PDFs regenerated with `pallet/scripts/md2pdf.js`
  (reproduces the Markdown PDF plugin's output: `node pallet/scripts/md2pdf.js docs/X.md docs/X.pdf`).

### 2026-10-17 07:45 — run 6 result (all data): saturation confirmed

- `runs/b256_run6` (8 epochs, 2,030 train / 468 val picks, 62,090 train images, 7.0 h, 19.8 img/s), EMA 1 step on its own 600-image
  val: origin 5.7 / 7.0 / 10.5 / 15.4 cm (0-2 / 2-3 / 3-4.5 / 4.5-6 m), **overall 10.18**; P 8.35; rot 0.96; pallet IoU 0.71.
  Per-epoch EMA overall: 10.81 (ep 2) -> 10.64 -> 10.55 -> 10.18. Hybrid-3: P 7.05 but origin 12.02 (as before: keep 1 step).
- Cross-evaluation (approximate, each model saw part of the other's val picks): run 5 on run 6's split 10.43 (vs run 6's 10.18);
  run 6 on run 5's split 9.16 (vs run 5's 10.49). => run 6 is at best ~0.3 cm better than run 5 with 1.4x the training picks:
  the data lever is exhausted for this model/resolution/voting. Results table in `PALLET_POINTDIT_PAPER.md` not updated (owner:
  the pallet-pose session). Files: `runs/b256_run6/{eval.log, eval_on_run5_split.log}`, `runs/b256_run5/eval_on_run6_split.log`,
  comparison plot `runs/fit_stats_runs4-6.png` (a second plot loop refreshes it every 5 min).
- Nothing on the GPU now; pull_loop, compact_loop, tensorboard (:6006), both plot loops still running.

### 2026-10-17 00:10 — run 6 launched on all data (from the SensorUnit session, user's request)

- `pallet/scripts/night3_2026-10-17.sh` (log `runs/night3.log`): compact + manifest, then `runs/b256_run6` from run 5 EMA on the
  whole store (2,614 picks with tensors at 00:10; the 23:51 pull found nothing new on ml-02, which has 2,862 picks with rays),
  batch 16, lr 3e-5, 8 epochs, eval every 2, `--val_raw` (~9 h), then `evaluate.py` 1 step + hybrid 3, plus run 6 on run 5's split
  (`eval_on_run5_split.log`) and run 5 on run 6's split (`runs/b256_run5/eval_on_run6_split.log`). train.py has no `--split-from`, so
  neither cross-evaluation is clean (each model has seen part of the other's val picks); read them as bounds. No ray loop relaunch
  (ml-02 builds the history). pull_loop, compact_loop, tensorboard, plot_runs left running.

### 2026-10-17 14:05 — run 6 result (launched from another session by the user's night3 chain)

- `runs/b256_run6` (from run 5, 2,030 train / 468 val picks, 62k images, 8 epochs, lr 3e-5, finished 07:12): own split 1 step
  origin 5.7 / 7.0 / 10.5 / 15.4 / 16.8 / 33.3 cm, overall 10.2; P 8.3; rot 0.96; pallet IoU 0.71.
  **Like-for-like on run 5's split: run 4 10.5, run 5 10.5, run 6 9.2 cm overall**; near field unchanged (5.7 / 6.0), far bins
  clearly better (3-4.5 m 12.2 -> 9.8, 6-8 m 21.6 -> 16.6, 8-12 m 27.0 -> 19.6). Hybrid-3 again: P better, origin worse far out.
- Reading: data diversity (run 6 added the oldest third of the history) still helps beyond 3 m; the near-field floor (~5.7 cm, ~4.3 cm
  with GT mask + R) is systematic and needs the bias diagnosis (paper section VI, step 1) rather than more picks.
- Paper table and discussion updated accordingly. Store 2,620 picks; disk ~68 GB.

### 2026-10-16 12:35 — run 5 result: data saturation

- `runs/b256_run5` (8 epochs, 1,453 train / 359 val picks), EMA 1 step: origin 6.1 / 6.9 / 11.5 / 13.6 / 20.2 / 26.2 cm per bin,
  overall 10.5; P 8.5; rot 0.93; pallet IoU 0.71. **Run 4 evaluated on the same split (`evaluate.py --split-from runs/b256_run5`):
  overall 10.55** -> no gain from 816 -> 1,453 picks. Hybrid-3 again: P 8.5 -> 7.2 but origin unchanged. rot-from-obj worse.
- Paper results table updated (run 5 row + like-for-like row). Ray loop relaunched 12:28 with the GPU-memory wait; store 2,142 picks;
  pull_loop and compact_loop still running; disk 110 GB.
- Next levers (user to choose): PointDiT-L/256 fine-tune, 2x input resolution, voting improvements (confidence weights / inlier
  selection / fuse over the sequence), then TensorRT export.

### 2026-10-16 07:35 — run 4 hybrid evaluation, run 5 launched

- Docs written 2026-10-16 ~08:00: `docs/PALLET_POINTDIT_PAPER.md` (paper-style description: targets, masking, model changes,
  objective, voting, data, results of runs 1-4) and `docs/PALLET_CODE_GUIDE.md` (file-by-file code guide with shapes, masks,
  training loop, metrics, tooling, pitfalls). Update the results table there when run 5 finishes.

- Run 4 EMA, 600 val images, origin error per bin (0-2 / 2-3 / 3-4.5 / 4.5-6 m): 1 step 5.2 / 7.5 / 11.3 / 14.5 cm;
  hybrid (angles 1 step, P/range 3 steps) 5.5 / 6.7 / 8.9 / 18.5 cm, P error 6.3->5.1, 7.3->5.7, 8.5->6.6, vote spread 4.9->3.3 cm.
  Hybrid helps at 2-4.5 m and for P everywhere, hurts at 4.5-6 m, neutral near; overall 9.7 either way. Rotation from predicted
  pallet/load pixels only: worse (1.26 vs 0.91 deg). Keep 1 step as the default; revisit with a confidence-weighted vote.
- Run 5 started 07:30 (`runs/b256_run5`): 1,453 train / 359 val picks (44,340 train images), from run 4 EMA, batch 16, lr 4e-5,
  8 epochs (~22k steps, ~5.5 h -> ~13:00), 17.8 img/s. night2 chain evaluates it and relaunches the ray loop afterwards.

### 2026-10-16 07:30 — night stall (again), fix

- The night2 chain did nothing from 00:32 to 07:25: the ray loop (relaunched by overnight.sh) waited on `pgrep -f pallet.train`,
  which matched the Bash-tool wrapper shell that had launched night2.sh (its command line held the heredoc text of the script,
  including "pallet.train"); night2 in turn waited for the ray loop to exit. GPU idle ~7 h, no run 5. Killed the wrapper 07:25;
  `rays_from_ml02.sh` now waits on GPU memory use (nvidia-smi > 2 GB) + `pgrep -x test_pallet_model_app`, no `-f` patterns.
  Rule for all scripts: never wait on `pgrep -f <pattern>` (shell wrappers and heredocs carry the pattern); use PID files, -x, or GPU memory.
- Store at 07:20: 1,820 picks (pull_loop kept working all night). Chain now continues: evaluate run 4 -> run 5 -> evaluate -> ray loop.

### 2026-10-16 01:05 — run 4 result

- `runs/b256_run4` (10 epochs, 816 train / 204 val picks), EMA medians: origin error 5.2 cm (0-2 m), 7.5 (2-3), 11.3 (3-4.5),
  14.5 (4.5-6), 21.7 (6-8), 29.1 (8-12); overall 9.7 (run 3: 11.35); P error 6.3-18 cm (overall 8.9); rot 0.9 deg; class acc 0.99,
  pallet IoU 0.72, load IoU 0.92; GT mask + GT rotation 4.1 / 5.6 / 8.2 cm; vote spread 3.9-4.9 cm near. Flat from epoch 6 on.
- Store at 01:05: 1,551 picks. night2 chain proceeding (evaluate run 4 -> run 5 -> evaluate -> ray loop).

### 2026-10-15 23:15 — night plan (user away until ~07:00)

- `pallet/scripts/night2_2026-10-15.sh` (log `runs/night2.log`) waits for run 4 to end (~01:00), stops the ray loop, evaluates run 4
  (`runs/b256_run4/eval.log`: 1 step + hybrid-3; `eval_rotobj.log`: same with rotation from predicted pallet/load pixels),
  rebuilds the manifest, trains run 5 (`runs/b256_run5`, from run 4 EMA, 8 epochs, lr 4e-5, ~5 h on the store as of ~01:00,
  expected 1,500+ picks thanks to `pull_loop.sh`), evaluates it the same way, then relaunches the ray loop.
- `pallet/evaluate.py` gained `--hybrid k` (angles/logits from 1 step, P/range from k steps) and `--rot-from obj`;
  `metrics.image_metrics(rot_from=...)`. Test suite: 9 tests (one needs GPU + checkpoint).
- Still running: pull_loop (every 30 min), compact_loop (10 min), tensorboard :6006, plot_runs (fit_stats.png every 5 min).

### 2026-10-15 20:56 — multi-step sampling of run 3, run 4 launched

- 21:55: `pallet/scripts/pull_loop.sh` started (re-runs the ml-02 puller every 30 min; `touch _logs/pull_loop.STOP` to end).
  ml-02 had 1,258 built picks at 21:55 (~70/h). Run 4 epoch-2 validation: origin 9.8 cm overall, P 9.1 cm, pallet IoU 0.72.

- `pallet/evaluate.py` on run 3 EMA (600 val images): steps 1 / 3 / 5 -> P error 9.5 / 7.7 / 7.8 cm, vote spread 5.6 / 3.7 / 3.7,
  origin error 0-2 m 6.4 / 5.7 / -, 2-3 m 8.9 / 8.2 / -, but theta error 0.22 / 0.33 / 0.34 deg, rotation 1.00 / 1.13 / 1.16 deg,
  overall origin 11.35 / 12.2 / 12.7 cm. => 3 steps sharpen P/range, 1 step is better for the angles. Cheap win to implement:
  hybrid inference (angles + rotation from the 1-step pass, P/range from a 3-step pass), or rotation from pallet/load pixels only.
- Run 4 launched 20:56 by `pallet/scripts/overnight_2026-10-15.sh`: store 1,020 picks / 31,274 images, from run 3 EMA, batch 16,
  lr 5e-5, 10 epochs (~4 h), `runs/b256_run4`; the ray loop is relaunched by the chain when run 4 ends.

### 2026-10-15 20:50 — run 3 result

- `runs/b256_run3` (12 epochs, 655 train / 164 val picks), EMA medians: origin error 6.4 cm (0-2 m), 8.9 (2-3), 11.7 (3-4.5),
  19.2 (4.5-6), 21.9 (6-8), 29.1 (8-12); overall 11.35 (run 2: 12.2); P error 6.7-20 cm (overall 9.5, run 2: 11.2); rot 1.0 deg;
  class acc 0.98, pallet IoU 0.69, load IoU 0.91. GT mask + GT rotation: 4.9 / 6.5 / 8.1 cm; vote spread 4.4-5.5 cm near.
  Doubling the data again gave ~1 cm: the remaining error is in the voting/class mask, not in fit.
- Live monitoring: tensorboard on :6006 (`runs/tensorboard.log`), `runs/fit_stats.png` refreshed every 5 min by
  `python -m pallet.plot_runs --loop 300`.
- Next (agreed 20:30): evaluate run 3 with 1/3/5 sampling steps (`pallet/evaluate.py`), then run 4 overnight on the full store
  (1,016 picks) from the run 3 weights, ray loop resumes after. Tomorrow: robust voting / rotation from pallet+load pixels /
  confidence weights, then step 4 (sequence fusion, TensorRT export).

### 2026-10-15 17:22 — run 3 launched

- `runs/b256_run3`: init from run 2 EMA, store 819 picks / 25,017 images, 655 train / 164 val picks (600-image cap), batch 16,
  lr 7e-5, 300 warmup, 12 epochs (~15k steps, ~3.5 h at 19.5 img/s), eval every 2 epochs, `--val_raw`.
- At 17:19: ray loop 283 picks built (0 failures); third puller pass slow (~50 s/pick, shares the LAN with the ray loop's
  per-pick rsync), still running; ml-02 had 822 built picks at 15:42.

### 2026-10-15 12:35 — run 2 result

- `runs/b256_run2` (12 epochs, 318 train / 80 val picks, val capped 600 images), EMA medians: origin error 7.7 cm (0-2 m), 9.3 (2-3),
  12.5 (3-4.5), 21 (4.5-6), 27 (6-8), 34 (8-12); P error 8-24 cm; rotation ~1.0 deg; class acc 0.98, pallet IoU 0.68, load 0.90.
  With GT class mask + GT rotation the origin error is 5.8 / 6.5 / 8.1 cm in the first three bins: class mask and rotation cost
  ~2-4 cm; the vote spread (median |vote - median|) is 5-7 cm. Gains flattened after epoch 7 (12.5 -> 12.2 cm overall).
- Store at 12:34: 585 picks. Second puller pass running (ml-02 had 509 built picks at 11:04). Ray loop resumed after run 2.
- Plan: run 3 at ~16:00 on the full store (longer, maybe batch 16 x 20 epochs), ray loop pauses meanwhile.

### 2026-10-15 10:37 — run 2 launched on the store

- `runs/b256_run2` (log `runs/b256_run2.log`): init from `runs/b256_baseline/checkpoint-last.pth` (EMA weights), 398 picks /
  12,237 images in the store (249: 10,268 images, 250: 1,969), split 318 train picks / 80 val picks (val capped to 600 images),
  batch 16, lr 1e-4, 500 warmup, 12 epochs (~7.4k steps, ~1.75 h at 18.6 img/s, 8.6 GB), EMA warmup, `--val_raw`, eval every 2 epochs.
  The ray loop pauses itself while this runs; the ml-02 puller and the compactor keep going.

### 2026-10-15 09:30-10:20 — baseline result, training store, three data feeders

- **Baseline (`runs/b256_baseline`, 60 epochs, 30 train / 8 val picks), EMA, medians:** origin error from voting 7 cm (0-2 m),
  11 cm (2-3 m), 16 cm (3-4.5 m), 25 cm (4.5-6 m), 34 cm (6-8 m); P error on pallet/load 9-19 cm; rotation < 1.1 deg; class acc 0.98,
  pallet IoU 0.65, load IoU 0.90. Validation table per 5 epochs in the log and `val_*.json`.
- **Lost 6 h (03:30-09:30):** the ray loop waits on `pgrep -f pallet.train`; my own background wait shell had that string in its
  command line, so the loop never started. Rule: never put a process-name pattern you wait on into a `bash -c` command line.
- **Training store = `~/Anantak/Data/PalletTrainImages/<robot>/<pick>/{02/, rays/}`** with COMPACT records (`pallet/compact.py`:
  cls, range, pose only, ~250 KB/image vs 1.4 MB; `targets.reconstruct` rebuilds P/theta/phi on load, verified < 1 mm /
  1e-4 rad; background P becomes on-ray). Manifest from the store: `python -m pallet.manifest` (kinds from SensorUnit's
  picks.parquet). The dataset reads full or compact npz transparently.
- Feeders (all `setsid nohup`, logs in `~/Anantak/Data/PicksCopies/_logs/`): `pallet/scripts/rays_from_ml02.sh` (GPU, ~30 s/pick
  when staged, pauses while a `pallet.train` process exists), `prefetch_from_ml02.sh` (stopped 10:10 to save disk; the ray loop
  fetches per pick itself), `pull_ml02_rays.sh` (ml-02 had 401 built picks at 09:40, ~10 s/pick, no GPU), `compact_loop.sh`
  (copy mode every 10 min). Deleting the full `PalletModels/*/*/rays/cam2_*.npz` after compaction needs the user's OK
  (classifier blocked `--delete-src`); until then ~65 MB/pick of disk per locally built pick. Caches: ~/.cache/uv 19 GB,
  ~/.cache/pip 11 GB, ~/.cache/torch 14 GB (not touched).
- `pallet/train.py`: EMA warmup, `--val_raw`, `--val_max_images 600`. `pose.rotation_error_deg` now Frobenius-based (arccos was
  ill-conditioned: float32 R gave 0.05 deg "errors").

### 2026-10-15 (night) — autonomous data building while training

- User instruction 02:20: keep going for several hours, build more data from picks. GPU is time-shared: the SensorUnit pipeline app
  is CUDA and OOMs next to a training run, so `pallet/scripts/rays_from_ml02.sh` waits for `pallet.train` to exit; the prefetcher
  stages raw episodes meanwhile. See `pallet/scripts/README.md`. ~2,790 local pallet models lack rays; ml-02 holds all raw episodes.
- Mistake, self-healed: the first loop version deleted 5 raw folders of the local rolling window (249/1747-1749, 250/1274-1275);
  they were re-fetched from ml-02. The current scripts only delete what they fetched.
- Images of the 18 image-less picks were pulled from ml-02 (`PicksCopies/<r>/<p>/02/`), manifest regenerated: 56 picks / 1,831 images.
- `pallet/train.py` now has EMA warmup (decay ramps (1+s)/(10+s) -> ema_decay) and `--val_raw`; the running baseline predates both.

### 2026-10-15 (later) — step 3 code in, baseline launched, data moved under us

- **Raw images are a rolling window.** The SensorUnit rebuild job (`rounds_*.log` under `PicksCopies/_logs`) keeps the 60 newest raw
  pick folders per robot and deletes the oldest when disk is tight (they stay in GCS); ray tensors under `PalletModels` stay. 18 of
  the original 25 tensor picks lost their images today. Protective copy of the images of every pick with tensors:
  `~/Anantak/Data/PalletTrainImages/<robot>/<pick>/02/` (38 picks, 1,230 images, 259 MB; `cp -n`, rerun to add new picks), and a
  manifest built against it: `python3.11 scripts/utils/pallet_manifest.py --episodes ~/Anantak/Data/PalletTrainImages --out
  ~/Anantak/Data/PalletTrainImages/manifest` (SensorUnit). `pallet/dataset.py` defaults to that manifest and drops rows whose
  files are missing. 56 picks have tensors now; only 38 have images. Most new picks have no kind yet (NaN): `assign_splits`
  treats unknown kind as its own group and tops up to round(val_frac * n_picks) picks.
- **Step 3 code** (`pallet/model.py`, `pose.py`, `metrics.py`, `train.py`, tests): PointDiT untouched, built with 6 input channels and
  the final linear widened to 6 + 4 (class logits); upstream flow loss per channel through the 6-channel mask + cross-entropy
  (ignore 255); B/256 weights loaded minus `x_embedder.proj1` and `final_layer.linear`. Target encoding fixed on 2026-10-15:
  OFFSET (0,0,0,0,pi/2,pi/2), SCALE (2,2,4,4,0.5,0.5); P/range supervised up to 12 m. Pose: R by Kabsch from predicted
  theta/phi, camera centre = median over pallet/load pixel votes `P - range * R d`; metrics binned by face distance.
- **Baseline run**: `runs/b256_baseline` (log `runs/b256_baseline.log`, tensorboard `runs/b256_baseline/tb`, panels, `val_*.json`),
  `python -m pallet.train --out runs/b256_baseline` defaults: batch 8, lr 1e-4, 300 warmup, cosine, 60 epochs (~7.9k steps),
  EMA .999, 1061 train / 169 val images (8 val picks, split.csv in the run dir), ~15 img/s, 6 GB. B/256 checkpoint:
  `pretrained/pointditb-256-scenenet-nodinov3-58f0b231.pth` (1.07 GB).
- Smoke training from the proposal dir needs `cd` to that dir (the repo's `pallet/` shadows it otherwise); from the repo root
  `python -m pallet.train` just works. Tests: `.venv/bin/python -m pytest pallet/tests -q` (8 tests, one needs the GPU + checkpoint).

### 2026-10-15 — steps 1 and 2 done on the laptop

- Environment: no system Python 3.12 on ana-ml-1, so the venv is uv-managed: `.venv` in the repo root (`uv venv --python 3.12 .venv`),
  torch 2.7.0+cu128 / torchvision 0.22.0, `requirements.txt` + `pallet/requirements.txt` (pandas, pyarrow, pytest). Run things with
  `.venv/bin/python`. `third_party/dinov3` and `pretrained/dinov3` are symlinks into `~/Anantak/TensorRT-Models` (checkout and weights).
  Venv is 7.2 GB; disk had 46 GB free afterwards.
- Correction to "Decisions": the upright tensors are **424 high x 240 wide**, so the patch-16 padding is on the **height**, 424 -> 432:
  4 rows top + 4 bottom (`pallet/targets.py` PAD_TOP/PAD_BOTTOM), cy += 4, padded rows black, class 255 (ignore), all masks false.
  Width 240 is already a multiple of 16.
- 25 picks with tensors on this laptop (21 on robot 249, 4 on 250), 892 images. f_px is 317.0 on 249 and 319.0 on 250 (per-pick cameras.json).
- Verified on the data: for pallet/load/ground pixels P == cam_in_pallet + range * R d to < 1 mm and theta/phi are the ray angles to 1e-6 rad.
  Background pixels are splatted lidar points, up to ~16 cm off the ray: vote with pallet/load pixels only.
- `pallet/` package: `targets.py` (channel order px,py,pz,range,theta,phi; class map; fixed per-channel TARGET_OFFSET/SCALE, identity for now,
  to be chosen in step 3; pad/unpad; pixel rays; pose identity), `dataset.py` (`PalletDataset` over images.parquet, per-channel
  `valid_mask` [6,H,W], `cls` [H,W], `pixel_mask`, pose, intrinsics; optional crop in multiples of 16, colour jitter, max_range_m;
  `assign_splits` by pick stratified by kind), `visualize.py` (CLI, same colours as the pipeline panels), `tests/test_dataset.py` (5 tests).
  Default training size is the full 432x240, no crop. Sample key names deliberately differ from the stock loader (`target` not
  `pointcloud`); the engine/denoiser changes for 6 channels + class head are step 3.
- Commands: `.venv/bin/python -m pytest pallet/tests -q`; `.venv/bin/python -m pallet.visualize --picks 249/1697 --out /tmp/panels`.

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
