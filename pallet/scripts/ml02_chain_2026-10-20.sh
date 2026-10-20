#!/bin/bash
# ml-02 chain 2026-10-20: refresh manifest -> splits (same picks as the laptop) -> prior2 sequence eval -> fixed_val evals ->
# resume prior2's last epoch at batch 8 -> independent-lineage model -> its eval + yaw correlation.
cd /home/ubuntu/Libraries/GoogleResearch/pointdit
export DINOV3_REPO=$PWD/third_party/dinov3 DINOV3_WEIGHTS_DIR=$PWD/pretrained/dinov3
PY=.venv/bin/python; MAN=/home/ubuntu/Anantak/Data/PalletModels/manifest/images.parquet
echo "start $(date +%F' '%H:%M)"
python3.11 /home/ubuntu/Anantak/SensorUnit/scripts/utils/pallet_manifest.py --no-tensor-stats 2>&1 | tail -1
$PY -m pallet.make_split --manifest $MAN --picks pallet/splits/fixed_val_picks_2026-10-19.csv --default train --out runs/fixed_val/split.csv
$PY -m pallet.make_split --manifest $MAN --picks pallet/splits/prior2_picks_2026-10-20.csv --default drop --out runs/b256_prior2/split.csv
echo "splits done $(date +%H:%M)"
$PY -m pallet.sequence --run runs/b256_prior2 --split runs/fixed_val --picks 30 --sources pred pred_gtpose oracle --t0 0.2 0.35 0.5 0.8 --steps 1 \
  --pose_alpha 0.3 --consistency_cm 0 --splat 3 --prior_fill noise --out runs/seq_eval_prior2 > runs/seq_eval_prior2.log 2>&1
echo "prior2 sequence eval done $(date +%H:%M)"
for r in b256_prior1 b256_prior2 b256_run6; do $PY -m pallet.evaluate --run runs/$r --steps 1 --split-from runs/fixed_val --val_max_images 1500 > runs/$r/eval_fixed_val.log 2>&1; done
echo "fixed_val evals done $(date +%H:%M)"
$PY -m pallet.train --out runs/b256_prior2 --resume runs/b256_prior2/checkpoint-last.pth --split_csv runs/b256_prior2/split.csv \
  --pretrained runs/b256_run6/checkpoint-last.pth --batch_size 8 --lr 1.5e-5 --warmup_steps 300 --epochs 3 --eval_every 1 \
  --prior_p 0.5 --prior_rot_deg 2.0 --prior_trans_cm 8.0 --consist_weight 0.5 --num_workers 8 > runs/b256_prior2_ml02.log 2>&1
echo "prior2 epoch 3 finished $(date +%F' '%H:%M) rc $?"
$PY -m pallet.evaluate --run runs/b256_prior2 --steps 1 --split-from runs/fixed_val --val_max_images 1500 > runs/b256_prior2/eval_fixed_val_ep3.log 2>&1
$PY -m pallet.train --out runs/b256_indep --pretrained pretrained/pointditb-256-scenenet-nodinov3-58f0b231.pth --split_csv runs/fixed_val/split.csv \
  --batch_size 8 --lr 5e-5 --warmup_steps 500 --epochs 2 --eval_every 1 --num_workers 8 > runs/b256_indep.log 2>&1
echo "indep training finished $(date +%F' '%H:%M) rc $?"
$PY -m pallet.evaluate --run runs/b256_indep --steps 1 --split-from runs/fixed_val --val_max_images 1500 > runs/b256_indep/eval_fixed_val.log 2>&1
$PY - <<'PYEOF'
import json, numpy as np, collections
def per(r):
    d = json.load(open(f"runs/{r}/eval_steps1_ema_split-fixed_val.json"))["per_image"]; by = collections.defaultdict(list)
    for x in d:
        if x["face_dist_m"] < 8: by[x["key"]].append(x["rot_err_vec_deg"][1])
    return {k: float(np.mean(v)) for k, v in by.items() if len(v) >= 5}
a, b = per("b256_indep"), per("b256_run6"); ks = sorted(set(a) & set(b)); x, y = np.array([a[k] for k in ks]), np.array([b[k] for k in ks])
print(f"INDEP vs run6 per-pick yaw error over {len(ks)} picks: r = {np.corrcoef(x, y)[0,1]:.2f} (std {x.std():.2f} / {y.std():.2f} deg)")
PYEOF
echo "ML02_CHAIN_DONE $(date +%F' '%H:%M)"
