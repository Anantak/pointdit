#!/bin/bash
# Third chain 2026-10-20: an INDEPENDENT model (from the public B/256 weights, not the run-3..6 lineage) on the same store, to test
# whether the per-pick yaw bias is shared because of the labels (expected r ~0.95 with run 6) or inherited through initialisation.
cd /home/ubuntu/Libraries/GoogleResearch/pointdit
export DINOV3_REPO=$PWD/third_party/dinov3 DINOV3_WEIGHTS_DIR=$PWD/pretrained/dinov3
until grep -q "fixed_val evals done" runs/day2_2026-10-20.log; do sleep 120; done
echo "indep start $(date +%H:%M)"
.venv/bin/python -m pallet.train --out runs/b256_indep --pretrained pretrained/pointditb-256-scenenet-nodinov3-58f0b231.pth --batch_size 16 --lr 1e-4 \
  --warmup_steps 500 --epochs 2 --eval_every 1 --num_workers 8 --split_seed 7 > runs/b256_indep.log 2>&1
echo "indep training finished $(date +%F' '%H:%M) rc $?"
.venv/bin/python -m pallet.evaluate --run runs/b256_indep --steps 1 --split-from runs/fixed_val --val_max_images 1500 > runs/b256_indep/eval_fixed_val.log 2>&1
.venv/bin/python - <<'PY'
import json, numpy as np, collections
def per(r):
    d = json.load(open(f"runs/{r}/eval_steps1_ema_split-fixed_val.json"))["per_image"]; by = collections.defaultdict(list)
    for x in d:
        if x["face_dist_m"] < 8: by[x["key"]].append(x["rot_err_vec_deg"][1])
    return {k: float(np.mean(v)) for k, v in by.items() if len(v) >= 5}
a, b = per("b256_indep"), per("b256_run6"); ks = sorted(set(a) & set(b))
x, y = np.array([a[k] for k in ks]), np.array([b[k] for k in ks])
print(f"INDEP vs run6 per-pick yaw error over {len(ks)} picks: r = {np.corrcoef(x, y)[0,1]:.2f} (std {x.std():.2f} / {y.std():.2f} deg)")
PY
echo "indep done $(date +%H:%M)"
