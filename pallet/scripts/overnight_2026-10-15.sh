#!/bin/bash
# Overnight chain (2026-10-15): wait for the ray loop to stop -> evaluate run 3 with 1/3/5 steps -> run 4 on the full store -> resume the ray loop.
cd /home/ubuntu/Libraries/GoogleResearch/pointdit
export DINOV3_REPO=$PWD/third_party/dinov3 DINOV3_WEIGHTS_DIR=$PWD/pretrained/dinov3
L=/home/ubuntu/Anantak/Data/PicksCopies/_logs
while pgrep -f "rays_from_ml02.sh" > /dev/null; do sleep 15; done
echo "ray loop stopped $(date +%H:%M)"; rm -f $L/rays_from_ml02.STOP
.venv/bin/python -m pallet.evaluate --run runs/b256_run3 --steps 1 3 5 > runs/b256_run3/eval_steps.log 2>&1
echo "eval done $(date +%H:%M)"
.venv/bin/python -m pallet.compact --src-root ~/Anantak/Data/PalletModels 2>&1 | tail -1
.venv/bin/python -m pallet.manifest 2>&1 | tail -1
.venv/bin/python -m pallet.train --out runs/b256_run4 --pretrained runs/b256_run3/checkpoint-last.pth --batch_size 16 --lr 5e-5 --warmup_steps 300 --epochs 10 --eval_every 2 --val_raw --num_workers 8 > runs/b256_run4.log 2>&1
echo "run 4 finished $(date +%F' '%H:%M)"
setsid nohup pallet/scripts/rays_from_ml02.sh > $L/rays_from_ml02_$(date +%Y%m%d_%H%M).out 2>&1 < /dev/null &
echo "ray loop relaunched $(date +%H:%M)"
