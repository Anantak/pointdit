#!/bin/bash
# Second day chain 2026-10-20: after prior1 is evaluated -> prior2 (stronger pose corruption 2 deg / 8 cm + on-ray consistency
# loss 0.5, from run 6) -> sequence eval -> single-frame eval of both on the fixed split.
cd /home/ubuntu/Libraries/GoogleResearch/pointdit
export DINOV3_REPO=$PWD/third_party/dinov3 DINOV3_WEIGHTS_DIR=$PWD/pretrained/dinov3
until grep -q "prior1 sequence eval done" runs/day_2026-10-20.log; do sleep 120; done
echo "prior2 start $(date +%H:%M)"
.venv/bin/python -m pallet.train --out runs/b256_prior2 --pretrained runs/b256_run6/checkpoint-last.pth --batch_size 16 --lr 3e-5 \
  --warmup_steps 300 --epochs 3 --eval_every 1 --prior_p 0.5 --prior_rot_deg 2.0 --prior_trans_cm 8.0 --consist_weight 0.5 \
  --num_workers 8 > runs/b256_prior2.log 2>&1
echo "prior2 training finished $(date +%F' '%H:%M) rc $?"
.venv/bin/python -m pallet.sequence --run runs/b256_prior2 --picks 30 --sources pred pred_gtpose oracle --t0 0.2 0.35 0.5 0.8 --steps 1 \
  --pose_alpha 0.3 --consistency_cm 0 --splat 3 --prior_fill noise --out runs/seq_eval_prior2 > runs/seq_eval_prior2.log 2>&1
echo "prior2 sequence eval done $(date +%H:%M)"
for r in b256_prior1 b256_prior2; do .venv/bin/python -m pallet.evaluate --run runs/$r --steps 1 --split-from runs/fixed_val > runs/$r/eval_fixed_val.log 2>&1; done
echo "fixed_val evals done $(date +%H:%M)"
