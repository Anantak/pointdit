# PointDiT Study Guide

A study plan for this repo, oriented toward the end goal of pallet pose estimation
(SE(3) pose from camera, complementing the existing CUDA LiDAR pipeline, Kalman
tracker with π-ambiguity folding, and calibration stack).

**Repo:** [github.com/GoogleResearch/pointdit](https://github.com/GoogleResearch/pointdit) ·
**Paper:** [arXiv:2607.02515](https://arxiv.org/abs/2607.02515) ·
**Project page:** [haofeixu.github.io/pointdit](https://haofeixu.github.io/pointdit/) ·
**Checkpoints:** [huggingface.co/haofeixu/pointdit](https://huggingface.co/haofeixu/pointdit)

---

## Why study this repo

PointDiT is **not a pose estimator**. It produces a dense, affine-invariant point
map (`out_channels=3` in [model.py](../model.py); evaluation is entirely under
affine / scale-shift alignment in [metrics/alignment.py](../metrics/alignment.py)).
It has no notion of an object. To get pallet pose from its output you would still
run RANSAC / fork-pocket fitting downstream — useful, but it doesn't touch the hard
part of the problem.

The real reason to study it: it is the cleanest complete reference implementation of
**conditional flow matching with frozen-encoder conditioning**, in a domain where
every intermediate tensor is visually debuggable. That machinery is exactly what
SE(3)-PoseFlow is built on. Learn it here, where you can look at a point cloud and
see what went wrong, then transfer to SE(3) where you can't.

The specific payoff: **π-ambiguity is the canonical case where a generative pose
model beats a regressor.** A stringer pallet face viewed head-on has a genuinely
bimodal pose posterior. An L2 regressor averages the two modes and outputs garbage
in between; that is currently patched downstream with folding logic in the tracker.
A flow model represents the bimodality natively — sample it and you get two
clusters, and the tracker's job becomes mode *selection* instead of ambiguity
*repair*.

---

## Phase 1 — Flow matching theory (1–3 weeks)

Read in this order; the first is the single best entry point. If you're already
comfortable with diffusion-model basics, this compresses to ~1 week: treat the
Guide as primary and skim the rest for the specific ideas noted below. Read
[denoiser.py](../denoiser.py) *alongside* this phase, not after it — 236 lines of
concrete code is the fastest way to make the theory stick.

| Paper | Link | Why |
|---|---|---|
| Lipman et al., *Flow Matching Guide and Code* | [arXiv:2412.06264](https://arxiv.org/abs/2412.06264) | Textbook-length, has the code. **Do this first.** |
| Lipman et al., *Flow Matching for Generative Modeling* | [arXiv:2210.02747](https://arxiv.org/abs/2210.02747) | The conditional/marginal vector-field identity — the one idea everything else rests on. |
| Liu et al., *Flow Straight and Fast* (Rectified Flow) | [arXiv:2209.03003](https://arxiv.org/abs/2209.03003) | Linear interpolant, path straightening. Explains why 1–3 Euler steps suffice. |
| Esser et al., *Scaling Rectified Flow Transformers* (SD3) | [arXiv:2403.03206](https://arxiv.org/abs/2403.03206) | Source of the logit-normal t schedule. PointDiT's `P_mean=-0.8, P_std=0.8` comes from here via JiT. |
| Li & He, *Back to Basics: Let Denoising Generative Models Denoise* (JiT) | [arXiv:2511.13720](https://arxiv.org/abs/2511.13720) | x-prediction vs v/ε-prediction in high dimensions. The load-bearing choice — the PointDiT paper reports v-prediction failing catastrophically. |
| Karras et al., *Elucidating the Design Space…* (EDM) | [arXiv:2206.00364](https://arxiv.org/abs/2206.00364) | The preconditioning / noise-schedule vocabulary everyone else assumes. |

## Phase 2 — Architecture (1 week)

| Paper | Link | Why |
|---|---|---|
| Peebles & Xie, *Scalable Diffusion Models with Transformers* (DiT) | [arXiv:2212.09748](https://arxiv.org/abs/2212.09748) | adaLN-zero conditioning. |
| Su et al., *RoFormer* | [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) | RoPE; PointDiT uses a 2D-axial variant ([model.py](../model.py) ~L489–530). |
| *DINOv3* (Meta AI) | [arXiv:2508.10104](https://arxiv.org/abs/2508.10104) | The frozen conditioning encoder. |
| Ranftl et al., *Vision Transformers for Dense Prediction* (DPT) | [arXiv:2103.13413](https://arxiv.org/abs/2103.13413) | Why mid-layers carry geometry that the final layer has abstracted away. Note: PointDiT computes its 4 tap indices equally spaced, not DPT's hardcoded list — the analogy is conceptual. |

## Phase 3 — Read the repo (1–2 weeks)

Order matters. Go small-to-large:

1. **[loss.py](../loss.py)** (38 lines) — trivial, read it first. Note it's an
   *auxiliary* relative-point loss (weight 0.1, applied in
   [engine.py](../engine.py) ~L125–140), not the main objective.
2. **[denoiser.py](../denoiser.py)** (236 lines) — the entire flow-matching
   contract. Target and prediction share the same clamped denominator
   (`v = (x − z) / (1−t).clamp_min(t_eps)` at ~L97, same for `v_pred` at ~L100),
   so the loss algebraically reduces to `(x − x_pred)² / clamp(1−t)²` — an
   **x-space loss with a time-dependent weight**. Confirm that on paper; it's the
   clearest window into what the model actually optimizes.
3. **[dataloader/point_transforms.py](../dataloader/point_transforms.py)** (200
   lines) + the normalization block in
   [dataloader/img_depth_intrinsics.py](../dataloader/img_depth_intrinsics.py)
   (~L3933–3996) — centroid shift, **median**-distance scale (mean only with
   `--normalize_point_by_mean`), sky dome at radius 3.0. This is where a pallet
   adaptation will succeed or fail: per-frame affine normalization is right for
   zero-shot monocular geometry and wrong for a metric use case — you want metric,
   and you have LiDAR to anchor it.
4. **[model.py](../model.py)** — read `forward()` (~L549) first, then
   `extract_y_embedding()` (~L625), then work outward. Frozen DINOv3, four
   equally-spaced intermediate-layer taps, adaLN-zero blocks,
   `BottleneckPatchEmbed`.
5. **[engine.py](../engine.py)** (1650 lines) — skim; it's training scaffolding.
6. **[metrics/alignment.py](../metrics/alignment.py)** — the affine alignment
   protocol: exactly what the reported numbers do and don't claim.

Verified code details that aren't obvious from the paper:

- `--generate_noise_scale` defaults to **0.0** (deterministic, zeros init) —
  [main.py](../main.py) ~L85, [denoiser.py](../denoiser.py) ~L166.
- Sampling steps at eval: **3 steps for L/512 and H/512 only**; B/512 and all 256
  evals use 1 step; online eval during training uses 1 step (see
  [scripts/](../scripts)).
- `--noise_scale` is 1.0 at 256 and **2.0 at 512** (sqrt-area Simple-Diffusion
  rule, commented in `train_stage2_512_*.sh`).
- `--sky_loss_weight 0.01` appears only in the stage-2 (512) training scripts;
  default 0.0.
- `BottleneckPatchEmbed` ([model.py](../model.py) ~L49–70) is used even at patch
  16: conv to `bottleneck_dim` (128 for B/L, 256 for H), then 1×1 to hidden.
- Logit-normal t: `P_mean=-0.8, P_std=0.8` ([main.py](../main.py) ~L82–83),
  `t = sigmoid(z)` ([denoiser.py](../denoiser.py) ~L56–59).
- L/512 script-header numbers to reproduce: Rel_p 4.85 / d1_p 97.54 / Rel_d 3.09 /
  BF1 10.36 (`eval_l_512.sh`); H/512 is Rel_p 4.39.

## Phase 4 — Pose estimation literature (2 weeks)

This is the part that actually targets the goal, and it's largely disjoint from
PointDiT:

| Paper | Link | Why |
|---|---|---|
| *SE(3)-PoseFlow* (Jin et al.) | [arXiv:2511.01501](https://arxiv.org/abs/2511.01501) | 6D pose *distributions* via flow matching — the direct target architecture. |
| *Flose: Generative 6D Pose Estimation via Conditional Flow Matching* | [arXiv:2602.19719](https://arxiv.org/abs/2602.19719) · [project page](https://tev-fbk.github.io/Flose/) | Same family; second reference point. |
| Chen & Lipman, *Flow Matching on General Geometries* (Riemannian FM) | [arXiv:2302.03660](https://arxiv.org/abs/2302.03660) | Required for SE(3): geodesic interpolation on SO(3) replaces the linear interpolant. |
| Wen et al., *FoundationPose* (CVPR 2024) | [arXiv:2312.08344](https://arxiv.org/abs/2312.08344) | Render-and-compare with a known CAD model. |
| Labbé et al., *MegaPose* | [arXiv:2212.06870](https://arxiv.org/abs/2212.06870) | Same regime. Be honest here: pallets are a small set of known geometries (GMA stringer, CHEP block) — the regime where render-and-compare often beats learned generative pose with a fraction of the data. |
| Chen et al., *EPro-PnP* (CVPR 2022) | [arXiv:2203.13254](https://arxiv.org/abs/2203.13254) | Differentiable PnP with an explicit pose distribution. Directly relevant to π-ambiguity, much cheaper to train than a flow model. |
| BOP Challenge | [bop.felk.cvut.cz](https://bop.felk.cvut.cz) · 2024 report: [arXiv:2504.02812](https://arxiv.org/abs/2504.02812) | Evaluation protocol and metrics (ADD-S, VSD, MSSD) rather than method. |

---

## Suggested build order

Rather than jumping to a from-scratch pallet flow model:

1. **Reproduce.** Get the official L/512 checkpoint running and reproduce the eval
   numbers in the script headers. Establishes ground truth for everything after.
2. **Integrate.** Feed its point map into the existing fork-pocket detector,
   metric-anchored with LiDAR. A working camera-only detector fast, and a baseline
   to beat.
3. **Export.** Push it through `trt_base` — MoGe-2 is already done, so the delta is
   small, and it gives the real Orin latency budget for any DINOv3-conditioned
   approach.
4. **Decide.** Flow-based SE(3) head on frozen DINOv3 patch tokens (reusing
   PointDiT's conditioning design verbatim) vs. EPro-PnP vs. render-and-compare.

**The honest caveat on step 4:** the flow route needs pose-labeled pallet data at
scale, which means synthetic rendering (Isaac Sim / BlenderProc) with the GMA/CHEP
models plus a sim-to-real story. That is the dominant cost of the whole program —
larger than the modeling work. Scope it before committing.
