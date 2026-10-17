::: title
## Pallet Pose from Forkview Images with a Pixel-Space Diffusion Transformer
:::

::: authors
Manuj Naman, with Claude (Anthropic) as implementation assistant<br>
*Anantak Robotics*<br>
manujnaman@anantak.com
:::

:::: content

*Adapting PointDiT to predict metric, pallet-frame geometry and a per-pixel class from a single forklift camera image, and recovering the pallet pose by pixel voting. Working paper, 2026-10-16. Code: branch pallet-pose, package pallet/.*



### Abstract

A forklift approaching a pallet needs the pallet's pose relative to the camera from several metres away, continuously, from a
single low-resolution RGB camera. We adapt PointDiT, a pixel-space diffusion transformer for monocular point maps, to this task.
Instead of a per-frame normalised point map in the camera frame, the network predicts for every pixel seven quantities in a
metric, pallet-anchored frame: a four-way class (pallet, load, ground, background), the ray length, the hit point in the pallet
frame, and two ray angles. The targets come from a lidar reconstruction of each recorded pick, rendered into the camera. The
backbone, the frozen DINOv3 conditioning and the rectified-flow objective of PointDiT are kept; only the input/output channel
count, the per-channel masking and an added classification head change. Each pallet or load pixel then casts a vote for the
camera centre in the pallet frame; the median vote is the pose estimate. Fine-tuned from the public PointDiT-B/256 weights on
1,453 picks (44k images), the model reaches a median pallet-origin error of 5.2 cm within 2 m, 7.5 cm at 2–3 m and 11.3 cm at
3–4.5 m on 359 unseen picks, with a rotation error of 0.9 degrees, in a single network evaluation at 240x432 pixels.

### I. Problem

The camera is the forkview camera of a forklift (424x240 raw frames, rotated upright to 432x240 after padding; pinhole, f ~317–319
px, no distortion). During an approach the robot records 20–50 images from ~11 m down to ~1.2 m. For each image we want the
rigid transform `pallet_P_cam` (X_pallet = R X_cam + p) between the camera and a frame anchored on the pallet: origin at the
centre of the pallet's front face on the ground, x lateral, **y down** (gravity is exactly +y), z into the pallet.

Why not regress the pose directly? A dense, per-pixel formulation (i) uses the whole visible pallet and load surface instead of a
few keypoints, (ii) degrades gracefully with occlusion and distance, (iii) gives a segmentation and a metric depth map for free,
and (iv) lets later stages fuse evidence over the approach sequence. PointDiT already produces dense geometry from one image with
strong priors from DINOv3; the question was whether it transfers to a metric, object-anchored frame with a small dataset.

### II. Background: PointDiT in one paragraph

PointDiT (Xu et al., 2026) is a plain ViT that denoises raw 3D point-map patches in pixel space, conditioned on image tokens
from a frozen DINOv3 encoder (four intermediate layers concatenated), with image and point tokens concatenated along channels and
projected to the model width. It is trained with rectified flow, x-prediction and a logit-normal timestep distribution; sampling is
deterministic Euler from zero noise (1 step for the 256-px models). Targets are camera-frame XYZ maps centred on their valid mean
and divided by the mean distance, i.e. per-frame affine normalisation; the loss is a masked per-pixel MSE in v-space. The B/16
model has 12 blocks, width 768, patch 16, a bottleneck patch embedding (conv 16x16 to 128 then linear to 768) and a final
adaLN-modulated linear that emits `16*16*C` values per token, unpatchified to full resolution.

### III. Method

#### 3.1 Targets: seven channels in the pallet frame

For each pixel of the upright image, from the lidar reconstruction (SensorUnit pipeline, `pallet_rays.py`):

| channel | symbol | definition | where defined |
|---|---|---|---|
| class | c | 0 pallet, 1 load, 2 ground, 3 background | every pixel |
| range | r | length of the ray from the camera centre to what it hits (m) | hit with known range |
| point | P = (px, py, pz) | the hit point in the pallet frame (m) | hit |
| theta | θ | angle between the ray and gravity-down (+y of the pallet frame) | every pixel |
| phi | φ | atan2(d_z, d_x) of the ray projected on the ground plane; a ray straight into the face has φ = +π/2 | every pixel |

With the unit ray d_cam of a pixel (from the intrinsics), R the camera-to-pallet rotation and c the camera centre in the pallet
frame, the pallet-frame ray is d = R d_cam and

    P = c + r d,    θ = arccos(d_y),    φ = atan2(d_z, d_x).

θ and φ depend only on the camera pose, not on the scene; they are included as a dense, image-independent handle on the
rotation. On the real data, P reconstructed from (r, d, pose) matches the stored P to under 1 mm for pallet, load and ground
pixels; background pixels are splatted lidar points up to ~16 cm off the ray and are never used for pose.

**Fixed encoding.** PointDiT's per-frame centring and scaling destroy metric meaning, so they are removed. Instead each channel
gets one fixed, invertible affine chosen once from the data: offsets (0, 0, 0, 0, π/2, π/2) and scales (2, 2, 4, 4, 0.5, 0.5)
metres or radians, so the network sees O(1) numbers (section 5 of the code guide). Range and P are supervised only up to 12 m.

**Padding.** 424 is not a multiple of the 16-px patch; 4 black rows are added above and below (432x240), the principal point
shifts by 4 px, padded pixels carry class 255 (ignore) and all-false masks. Width 240 is already a multiple of 16.

#### 3.2 Masking

Three masks travel with every sample:

* `pixel_mask` [H,W]: real (non-padded) pixels.
* `valid_mask` [6,H,W]: per channel. P and range are valid where the range is known (and ≤ 12 m); θ and φ are valid on every real
  pixel. This replaces PointDiT's single [H,W] mask repeated over three channels.
* `cls` [H,W] with 255 on padding: the cross-entropy ignore index.

The flow loss is averaged over the true entries of `valid_mask` only; invalid target entries are replaced by noise before being
mixed into the diffusion input (PointDiT's `--noise_fill_invalid`), so the network cannot read "zero means invalid".

No horizontal flip is used (it would mirror the pallet frame); no per-frame normalisation; colour jitter and random crop are
available but the baseline trains at the full 432x240 without augmentation.

#### 3.3 Model changes

The PointDiT module is used unchanged. It is constructed with `in_channels = 6` (the continuous targets) and its final linear is
widened after construction to emit `6 + 4` channels: the first six are the x-prediction of the continuous targets, the last four
are class logits produced by the same final layer. The patch-embedding conv (`[128, 3, 16, 16]` pretrained) and the final linear
are the only weights that do not load from the public checkpoint; everything else (all transformer blocks, the concat projection,
positional embeddings, the adaLN of the final layer) is initialised from PointDiT-B/256. DINOv3-B/16 stays frozen. The positional
embedding is bicubically resized from the 16x16 pretraining grid to 27x15 patches on the fly.

#### 3.4 Objective

Per sample, with x the encoded 6-channel target and e ~ N(0, I):

    t ~ sigmoid(N(-0.8, 0.8)), with probability 0.1 set t = 0
    z = t * x_filled + (1 - t) * e            (x_filled = x where valid, e elsewhere)
    x_hat, logits = net(z, t, image)
    L_flow = mean over valid entries of  (x_hat - x)^2 / max(1 - t, 0.05)^2
    L_cls  = cross-entropy(logits, c, ignore 255)
    L = L_flow + L_cls

The weighting `1/(1-t)^2` is exactly PointDiT's v-space MSE rewritten for x-prediction, so the objective and the pretrained
weights agree. The 10 % of samples at t = 0 train the network under the exact condition used at inference. The class logits are
supervised at every t.

#### 3.5 Inference and pose voting

Inference is PointDiT's deterministic Euler integration from z = 0; with one step the network is a plain regressor
`x_hat = net(0, t=0, image)`, costing one forward pass with cached DINOv3 features.

Pose from the dense prediction:

1. **Rotation.** Every real pixel's predicted (θ, φ) gives the ray in the pallet frame, d_pal = (sin θ cos φ, cos θ, sin θ sin φ);
   its camera ray d_cam is known. R is the Kabsch solution of min Σ |d_pal − R d_cam|² over all real pixels (~100k pairs).
2. **Translation.** Each pixel classified pallet or load votes for the camera centre: c_i = P_i − r_i R d_i. The estimate is the
   coordinate-wise median of the votes; the median absolute deviation of the votes ("spread") is a confidence proxy.

The pallet origin in the camera frame is −Rᵀ c. The metric reported throughout is |c_est − c_gt| in the pallet frame, binned by
the distance of the camera to the pallet face.

### IV. Data

* **Source.** Recorded picks of two forklifts (robots 249 and 250) at one site. The SensorUnit pipeline reconstructs each pick from
  the XT32 lidar and the carriage odometry, fits the pallet and load as 5 cm oriented patches, calibrates the camera per pick
  (focal length and rotation by edge reprojection), and renders the per-pixel targets by exact ray–patch intersection plus a splat
  of the remaining lidar points and the analytic ground plane. Box reprojection agrees with the pipeline's own portrait corners to
  0.05 px.
* **Scale.** 2,895 picks exist; by 2026-10-16 07:30, 1,820 had rendered targets (about 56k images), produced on two machines in
  parallel: ana-ml-02 builds the full history in ascending order, the laptop builds newest-first from the same raw episodes.
* **Training store.** Images and *compact* target records (class map, range, pose; P, θ, φ are rebuilt exactly on load) live in
  `~/Anantak/Data/PalletTrainImages/<robot>/<pick>/`, about 250 KB per image instead of 1.4 MB. A manifest (one row per image)
  is built from the store.
* **Splits.** By pick, never by image (an approach sequence is highly correlated), stratified by the human-reviewed pallet kind
  where known, 20 % of picks held out; validation uses an even 600-image subsample of the held-out images.

### V. Experiments

All runs: PointDiT-B/16 at 432x240, frozen DINOv3-B/16, bf16, AdamW (0.9, 0.95), cosine schedule, EMA, grad-clip 1.0, 1-step
inference, one 16 GB laptop GPU (RTX 3080 Ti). Medians over the validation images per face-distance bin, pallet-origin error in cm.

| run | init | train picks / images | epochs, batch, lr | val picks | 0–2 m | 2–3 m | 3–4.5 m | 4.5–6 m | 6–8 m | all | P err | rot (deg) | pallet IoU |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 baseline | B/256 public | 30 / 1,061 | 60, 8, 1e-4 | 8 | 7.2 | 11.0 | 16.0 | 25.4 | 34.4 | 17.1 | 14.0 | 0.9 | 0.65 |
| 2 | run 1 | 318 / 9,809 | 12, 16, 1e-4 | 80 | 7.7 | 9.3 | 12.5 | 21.2 | 27.1 | 12.2 | 11.2 | 1.0 | 0.68 |
| 3 | run 2 | 655 / 20,151 | 12, 16, 7e-5 | 164 | 6.4 | 8.9 | 11.7 | 19.2 | 21.9 | 11.4 | 9.5 | 1.0 | 0.69 |
| 4 | run 3 | 816 / 25,049 | 10, 16, 5e-5 | 204 | 5.2 | 7.5 | 11.3 | 14.5 | 21.7 | 9.7 | 8.9 | 0.9 | 0.72 |
| 5 | run 4 | 1,453 / 44,340 | 8, 16, 4e-5 | 359 | 6.1 | 6.9 | 11.5 | 13.6 | 20.2 | 10.5 | 8.5 | 0.9 | 0.71 |
| 4 on run-5 split | | | | 359 | 5.7 | 7.4 | 12.2 | 13.4 | 21.6 | 10.6 | 8.8 | 0.9 | 0.72 |
| 6 | run 5 | 2,030 / 62,090 | 8, 16, 3e-5 | 468 | 5.7 | 7.0 | 10.5 | 15.4 | 16.8 | 10.2 | 8.3 | 1.0 | 0.71 |
| 6 on run-5 split | | | | 359 | 5.7 | 6.0 | 9.8 | 13.1 | 16.6 | 9.2 | 7.9 | 0.9 | 0.72 |

Class accuracy is 0.98–0.99 throughout; load IoU 0.90–0.92; the ground and background classes are easy. The validation set
changes with every run (more held-out picks), so rows are indicative rather than strictly comparable; the trend with data is
nevertheless monotone, and within each run the error curve flattens after 6–8 epochs. **Like-for-like** on run 5's 600 validation
images: run 4 10.5, run 5 10.5, run 6 9.2 cm overall. Run 5 (1,453 picks) did not improve on run 4 (816), but run 6 (2,030 picks,
including the oldest third of the history) did, almost entirely beyond 3 m (3–4.5 m 12.2 to 9.8, 6–8 m 21.6 to 16.6, 8–12 m 27 to
19.6 cm), while the near field stayed at 5.7–6.0 cm. The near-field floor is therefore not a data problem; the far field still
responds to data diversity.

**Where the remaining error is.** With the ground-truth class mask and ground-truth rotation but the *predicted* P and range, the
origin error of run 4 drops to 4.1 / 5.6 / 8.2 cm in the first three bins: mask and rotation together cost 1–3 cm. The vote spread
is 4–5 cm near the pallet. Within 3 m the 90th percentile is ~16 cm and 3 % of images exceed 20 cm; the error is mostly lateral and
depth (~4.5 cm each), height is 1.3 cm. Far bins are limited by pixel resolution: at 8 m a pallet spans ~45 px.

**Sampling steps.** Evaluating the run-4 weights with 3 Euler steps instead of 1 lowers the P error from 8.9 to 7.4 cm and the vote
spread from 4.9 to 3.3 cm, but the angles get worse (θ error 0.22 to 0.33 deg), and so does the rotation; the origin error improves
at 2–4.5 m (7.5 to 6.7, 11.3 to 8.9 cm) and worsens beyond 4.5 m (14.5 to 18.5 cm), a wash overall. A hybrid (angles from 1 step,
P and range from 3 steps) behaves the same. Rotation from pallet/load pixels only is worse (1.26 vs 0.91 deg): the ground and
walls carry most of the rotation signal.

**Cost.** Training throughput 18–20 images/s at batch 16 (8.6 GB); inference is one forward pass. Fine-tuning from the public
checkpoint converges in a few thousand steps; training from scratch was not attempted.

### VI. Discussion

* The per-frame normalisation of PointDiT is not needed for transfer: fixed per-channel scales and the pretrained blocks suffice,
  and the model keeps metric accuracy at the few-centimetre level near the pallet.
* The classification head shares the final layer with the regression; this cost nothing in accuracy and avoids a second decoder.
* Data helps the far field, not the near field. Runs 2 to 4 gained about 1 cm per doubling; run 5 did not improve on run 4, run 6
  did (9.2 vs 10.5 cm on the same images) but only beyond 3 m. The near-field error (5.7–6.0 cm within 2 m) has been flat since
  run 4 across a 2.5x increase in training picks: it is not a matter of more picks of the same site with this model and resolution.

#### Error budget

Reading the run-4/5 validation columns together gives a rough decomposition of the near-field origin error (0–3 m, ~6–7 cm):

| component | evidence | size |
|---|---|---|
| vote dispersion of P and range | median |vote − median| ("spread") | 3.7–3.9 cm per vote, largely averaged out by the median over 10^4 votes |
| class mask (which pixels vote) and rotation | origin error with GT mask + GT R vs predicted | 1.5–2 cm |
| pallet-frame bias of P | residual with GT mask + GT R | 4.3–5.4 cm |
| rotation alone | θ error 0.2 deg, φ 0.8 deg, R error 0.9 deg | at 2 m a 0.9 deg error moves the centre by ~3 cm |

So the floor is set by a *systematic* P/range bias per image rather than by noise: the median of 20k votes is 4–5 cm off even with
a perfect mask and rotation. Candidate causes, in the order they should be checked: (a) the camera calibration per pick (focal
length and rotation are fitted per pick from edge reprojection; a 1 % focal error at 2 m is 2 cm of depth), (b) the timestamp
interpolation of the carriage pose to the image time during motion, (c) label bias from the 5 cm patch quantisation, (d) a true
model bias (e.g. systematically shrinking the pallet). (a)–(c) are target errors and would also appear as a residual when voting
with the *ground-truth* P; that single experiment separates label noise from model error and should be run first.

Beyond 4.5 m the error grows roughly linearly with distance (14 cm at 5 m, 21 cm at 7 m, 27 cm at 10 m): a pallet spans ~90 px at
4 m and ~45 px at 8 m, so a one-pixel error in P is 2–4 cm there, and the class mask erodes (pallet IoU 0.75 near, 0.65 far).

#### Next steps, with expected gain and cost

1. **Diagnose the bias (no GPU, hours).** Vote with GT P/range/angles and the *predicted* mask, and with predicted P and the GT
   mask, on the saved validation predictions; plot the signed per-axis error against face distance and against the pick's
   camera-calibration residual. Outcome: whether 4–5 cm is label or model. If it is label, better calibration or a calibration
   channel in the network (predict a per-image focal correction) is the fix, not a bigger model.
2. **Voting and confidence (CPU, a day).** Inlier selection (e.g. iterate the median with a 3-sigma cut on |vote − c|), a
   per-pixel confidence from the class softmax or from the agreement between the 1-step and 3-step predictions (the hybrid
   showed the two differ most where P is unreliable), and weighting votes by predicted range (near pixels are more accurate).
   Expected gain: 1–2 cm near, more far, by removing the mask and dispersion terms; cheap to evaluate on the same JSON/predictions.
3. **Sequence fusion (CPU, days; the original step 4).** Chain the per-image votes with the carriage odometry into a filter over
   the approach; the pose is then estimated from 20–50 images rather than one. Noise terms average out; the systematic bias of
   (1) does not, which is why (1) comes first. Also yields the velocity and a confidence for the unit's planner.
4. **Model capacity: PointDiT-L/256 (GPU, ~1 day).** Same code path, public weights, 24 blocks / width 1024, ~3x compute, batch 8
   at 432x240 in 16 GB. Expected: sharper P and masks, mainly in the middle bins; the paper's own scaling (B to L) is worth ~10 %
   relative on its benchmarks. Cheap to try because the data and the pipeline are fixed.
5. **Resolution (GPU, 4x cost).** Upsample the input 2x (864x480, 54x30 patches); DINOv3 and the ViT scale quadratically in
   tokens, so training throughput drops to ~5 img/s and inference to ~4x the current cost on the unit. Targets the far bins only;
   run after (4) and only if the far field matters for the product.
6. **Multi-step inference** is already measured: 3 steps help P (−1.5 cm) and the 2–4.5 m band but hurt the angles; it becomes
   useful once the rotation is taken from a better source (sequence fusion or odometry), at 3x inference cost.
7. **Not yet addressed:** other sites and robots (domain shift), lighting extremes, overhanging loads, and the TensorRT export
   (the DINOv3 encoder already runs on the unit; the ViT is plain attention and should export directly).

The recommended order is 1 → 2 → 4 in parallel with 3, then decide on 5 from the product requirement for the far field.

::::

::: references
1. PointDiT: H. Xu et al., *PointDiT: Pixel-space Diffusion Transformers for Monocular Geometry Estimation*, 2026 (arXiv:2607.02515).
2. DINOv3: Meta AI, 2025.
3. Data pipeline: `~/Anantak/SensorUnit/docs/PALLET_DATA_BUILDING_PROCESS.md` (SensorUnit, branch THD).
:::
