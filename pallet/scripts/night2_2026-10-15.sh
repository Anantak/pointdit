#!/bin/bash
# Night chain 2 (2026-10-15 23:10): after overnight.sh relaunches the ray loop (run 4 done), stop it again, evaluate run 4
# (1 step, hybrid 3, rotation from object pixels), train run 5 on the whole store, evaluate it, then resume the ray loop.
cd /home/ubuntu/Libraries/GoogleResearch/pointdit
export DINOV3_REPO=$PWD/third_party/dinov3 DINOV3_WEIGHTS_DIR=$PWD/pretrained/dinov3
L=/home/ubuntu/Anantak/Data/PicksCopies/_logs
until grep -q "ray loop relaunched" runs/overnight.log; do sleep 60; done
touch $L/rays_from_ml02.STOP
while pgrep -f "rays_from_ml02.sh" > /dev/null; do sleep 15; done
rm -f $L/rays_from_ml02.STOP; echo "ray loop stopped $(date +%H:%M)"
.venv/bin/python -m pallet.evaluate --run runs/b256_run4 --steps 1 --hybrid 3 > runs/b256_run4/eval.log 2>&1
.venv/bin/python -m pallet.evaluate --run runs/b256_run4 --steps 1 --hybrid 3 --rot-from obj > runs/b256_run4/eval_rotobj.log 2>&1
echo "run 4 evaluated $(date +%H:%M)"
.venv/bin/python -m pallet.compact --src-root ~/Anantak/Data/PalletModels 2>&1 | tail -1
.venv/bin/python -m pallet.manifest 2>&1 | tail -1
.venv/bin/python -m pallet.train --out runs/b256_run5 --pretrained runs/b256_run4/checkpoint-last.pth --batch_size 16 --lr 4e-5 --warmup_steps 300 --epochs 8 --eval_every 2 --val_raw --num_workers 8 > runs/b256_run5.log 2>&1
echo "run 5 finished $(date +%F' '%H:%M)"
.venv/bin/python -m pallet.evaluate --run runs/b256_run5 --steps 1 --hybrid 3 > runs/b256_run5/eval.log 2>&1
.venv/bin/python -m pallet.evaluate --run runs/b256_run5 --steps 1 --hybrid 3 --rot-from obj > runs/b256_run5/eval_rotobj.log 2>&1
echo "run 5 evaluated $(date +%H:%M)"
setsid nohup pallet/scripts/rays_from_ml02.sh > $L/rays_from_ml02_$(date +%Y%m%d_%H%M).out 2>&1 < /dev/null &
echo "ray loop relaunched $(date +%H:%M)"
