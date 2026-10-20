#!/bin/bash
# Day chain 2026-10-20: wait for sweeps 3+4 -> prior-conditioned fine-tune of run 6 (runs/b256_prior1) -> sequence evaluation of
# the new checkpoint (own maps + own pose, own maps + GT pose, oracle) and of run 6 with the same settings for reference.
cd /home/ubuntu/Libraries/GoogleResearch/pointdit
export DINOV3_REPO=$PWD/third_party/dinov3 DINOV3_WEIGHTS_DIR=$PWD/pretrained/dinov3
until [ -f runs/seq_eval3/summary.csv ] && [ -f runs/seq_eval4/summary.csv ]; do sleep 60; done
echo "sweeps done $(date +%H:%M)"
.venv/bin/python -m pallet.train --out runs/b256_prior1 --pretrained runs/b256_run6/checkpoint-last.pth --batch_size 16 --lr 3e-5 \
  --warmup_steps 300 --epochs 3 --eval_every 1 --prior_p 0.5 --num_workers 8 > runs/b256_prior1.log 2>&1
echo "prior1 training finished $(date +%F' '%H:%M) rc $?"
.venv/bin/python -m pallet.sequence --run runs/b256_prior1 --picks 30 --sources pred pred_gtpose oracle --t0 0.2 0.35 0.5 0.8 --steps 1 \
  --pose_alpha 0.3 --consistency_cm 0 --splat 3 --prior_fill noise --out runs/seq_eval_prior1 > runs/seq_eval_prior1.log 2>&1
echo "prior1 sequence eval done $(date +%H:%M)"
