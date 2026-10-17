#!/bin/bash
# Night chain 3 (2026-10-17 00:15, launched from the SensorUnit session at the user's request): one more round of training on
# ALL the data in the store (2,969 picks at 00:10; ml-02 has finished most of the history, the store already holds everything it
# built). Compact + manifest, train run 6 from run 5's EMA, evaluate run 6, and evaluate run 5 vs run 6 on each other's split
# (train.py has no --split-from, so the like-for-like numbers are approximate: see the handover note). No ray loop afterwards.
cd /home/ubuntu/Libraries/GoogleResearch/pointdit
export DINOV3_REPO=$PWD/third_party/dinov3 DINOV3_WEIGHTS_DIR=$PWD/pretrained/dinov3
echo "night3 start $(date +%F' '%H:%M): store $(ls -d /home/ubuntu/Anantak/Data/PalletTrainImages/*/*/rays | wc -l) picks"
.venv/bin/python -m pallet.compact --src-root ~/Anantak/Data/PalletModels 2>&1 | tail -1
.venv/bin/python -m pallet.manifest 2>&1 | tail -1
echo "manifest rebuilt $(date +%H:%M)"
.venv/bin/python -m pallet.train --out runs/b256_run6 --pretrained runs/b256_run5/checkpoint-last.pth --batch_size 16 --lr 3e-5 --warmup_steps 300 --epochs 8 --eval_every 2 --val_raw --num_workers 8 > runs/b256_run6.log 2>&1
echo "run 6 finished $(date +%F' '%H:%M) (rc $?)"
.venv/bin/python -m pallet.evaluate --run runs/b256_run6 --steps 1 --hybrid 3 > runs/b256_run6/eval.log 2>&1
.venv/bin/python -m pallet.evaluate --run runs/b256_run6 --steps 1 --split-from runs/b256_run5 > runs/b256_run6/eval_on_run5_split.log 2>&1
.venv/bin/python -m pallet.evaluate --run runs/b256_run5 --steps 1 --split-from runs/b256_run6 > runs/b256_run5/eval_on_run6_split.log 2>&1
echo "run 6 evaluated $(date +%H:%M)"
