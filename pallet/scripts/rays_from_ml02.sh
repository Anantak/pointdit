#!/bin/bash
# Ray tensors for picks that have a local pallet model (pallet_pose.json + patch_cloud/patches.json, not NOT_A_PICK) but no rays,
# whose raw episode lives on ana-ml-02's drive. Newest first, robots alternating. Per pick: rsync raw episode -> xt32 step --rebuild
# (fused cloud) -> patch-cloud step (rays/cameras.json) -> pallet_rays.py -> copy 02 images to PalletTrainImages -> delete the raw
# copy, strip big clouds. Same per-pick logic as SensorUnit's build_rays_local.sh. touch $STOP to end after the current pick.
SU=/home/ubuntu/Anantak/SensorUnit; D=/home/ubuntu/Anantak/Data; M=$D/PalletModels; L=$D/PicksCopies/_logs; TI=$D/PalletTrainImages
R=ubuntu@192.168.5.193; RD=/media/ubuntu/ana-back-01/Work/Data; SSH="ssh -o IdentitiesOnly=yes -i /home/ubuntu/.ssh/id_rsa -o BatchMode=yes"
MINFREE=${MINFREE:-18}; MAXPICKS=${MAXPICKS:-2000}; STOP=$L/rays_from_ml02.STOP
free_gb() { df -BG --output=avail / | tail -1 | tr -dc '0-9'; }
$SSH $R "ls $RD/PicksCopies/249 | grep -E '^[0-9]{4,5}$'" | sort -rn > $L/ml02_raw_249.txt
$SSH $R "ls $RD/PicksCopies/250 | grep -E '^[0-9]{4,5}$'" | sort -rn > $L/ml02_raw_250.txt
python3 - "$L" <<'PY' > $L/rays_todo.txt
import sys, os, itertools
L = sys.argv[1]; M = '/home/ubuntu/Anantak/Data/PalletModels'
lists = {}
for r in ('249', '250'):
    todo = []
    for d in open(f'{L}/ml02_raw_{r}.txt').read().split():
        m = f'{M}/{r}/{d}'
        if not (os.path.exists(f'{m}/pallet_pose.json') and os.path.exists(f'{m}/patch_cloud/patches.json')): continue
        if os.path.exists(f'{m}/NOT_A_PICK') or os.path.exists(f'{m}/rays/cameras.json') or os.path.exists(f'{m}/RAYS_FAILED'): continue
        todo.append(f'{r}/{d}')
    lists[r] = todo
for a, b in itertools.zip_longest(lists['249'], lists['250']):
    for x in (a, b):
        if x: print(x)
PY
echo "start $(date +%F' '%H:%M): todo $(wc -l < $L/rays_todo.txt) picks, $(free_gb) GB free"
n=0
for rp in $(cat $L/rays_todo.txt); do
  [ -f $STOP ] && { echo "STOP file found, exiting"; exit 0; }
  [ $n -ge $MAXPICKS ] && { echo "MAXPICKS reached"; exit 0; }
  [ "$(free_gb)" -lt $MINFREE ] && { echo "SPACE: only $(free_gb) GB free, stopping"; exit 2; }
  r=${rp%%/*}; d=${rp#*/}; pk=$((10#$d)); ep=$D/PicksCopies/$r/$d; md=$M/$r/$d; t0=$(date +%s)
  [ -f $md/rays/cameras.json ] || [ -f $md/RAYS_FAILED ] && continue
  fetched=0; grep -qx "$rp" $L/ml02_fetched.txt 2>/dev/null && fetched=1
  if [ ! -d $ep ]; then
    mkdir -p $ep; fetched=1; echo "$rp" >> $L/ml02_fetched.txt
    rsync -a -e "$SSH" $R:$RD/PicksCopies/$r/$d/ $ep/ > $L/ml02_rsync_${r}_$d.log 2>&1 || { echo "robot $r pick $pk: RSYNC FAILED"; rm -rf $ep; continue; }
  fi
  t1=$(date +%s); nimg=$(ls $ep/02/color_*.png 2>/dev/null | wc -l)
  [ -d $ep/41 ] && [ $nimg -gt 0 ] || { echo "robot $r pick $pk: no XT32 slot or no forkview images ($nimg), skipped"; echo "no-xt32-or-images $(date +%FT%T)" > $md/RAYS_FAILED; [ $fetched = 1 ] && rm -rf $ep; continue; }
  # the pipeline app needs the GPU: wait while any other process holds > 2 GB of it (a training run); no process-name patterns
  # (a shell whose command line merely CONTAINS a pattern matched pgrep -f twice on 2026-10-15/16 and stalled this loop for hours)
  gpu_busy() { nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '$1 > 2048 {b=1} END {exit !b}'; }
  while gpu_busy || pgrep -x test_pallet_model_app > /dev/null; do sleep 30; done
  cd $SU/_install/bin; st=""
  if [ ! -f $md/xt32_fused.ply ] && [ ! -f $md/rays/background_points.npz ]; then
    timeout 900 ./test_pallet_model_app --robot=$r --pick=$pk --step=xt32 --rebuild --exit > $L/ml02_xt32_${r}_$d.log 2>&1 || st="xt32-failed"
    grep -q "out of memory" $L/ml02_xt32_${r}_$d.log && { echo "robot $r pick $pk: GPU out of memory, will retry later; sleeping 5 min"; sleep 300; continue; }; fi
  if [ -z "$st" ] && [ ! -f $md/rays/cameras.json ]; then
    timeout 900 ./test_pallet_model_app --robot=$r --pick=$pk --step=patch-cloud --exit > $L/ml02_pc_${r}_$d.log 2>&1 || st="patch-cloud-failed"; fi
  rays=""
  if [ -z "$st" ] && [ -f $md/rays/cameras.json ]; then
    python3.11 $SU/scripts/utils/pallet_rays.py --robot $r --pick $pk > $L/ml02_rays_${r}_$d.log 2>&1 && rays=$(grep -o 'wrote [0-9]* ray' $L/ml02_rays_${r}_$d.log | grep -o '[0-9]*') || st="rays-failed"
  else [ -z "$st" ] && st="no-cameras.json"; fi
  ncp=0; if [ -n "$rays" ] && [ "$rays" -gt 0 ]; then mkdir -p $TI/$r/$d/02; cp -n $ep/02/color_*.png $ep/02/color_*.txt $TI/$r/$d/02/ 2>/dev/null; ncp=$(ls $TI/$r/$d/02/*.png 2>/dev/null | wc -l); fi
  [ -n "$st" ] && echo "$st $(date +%FT%T)" > $md/RAYS_FAILED
  [ $fetched = 1 ] && rm -rf $ep
  for f in xt32_fused.ply jt128_fused.ply jt128_implied_fused.ply jt128_in_xt32_frame.ply; do rm -f $md/$f; done; rm -rf $md/camera_calib_overlays; rm -f $md/cam2_mono_cloud_*.ply
  echo "robot $r pick $pk: rsync $((t1-t0))s steps $(( $(date +%s)-t1 ))s images $nimg${st:+ | $st} | rays ${rays:-0} | copied $ncp | $(free_gb) GB free $(date +%H:%M)"
  n=$((n+1))
done
echo "=== DONE: $n picks $(date +%F' '%H:%M)"
