#!/bin/bash
# Pull ray tensors + forkview images of picks already built on ana-ml-02 into the training store as compact records.
# Per pick: rsync rays/ to <store>/<robot>/<pick>/rays_full/, rsync 02/color_*, compact (verified) into rays/, delete rays_full.
# Skips picks already in the store. touch $STOP to end after the current pick.
ST=/home/ubuntu/Anantak/Data/PalletTrainImages; L=/home/ubuntu/Anantak/Data/PicksCopies/_logs; PD=/home/ubuntu/Libraries/GoogleResearch/pointdit
R=ubuntu@192.168.5.193; RD=/media/ubuntu/ana-back-01/Work/Data; SSH="ssh -o IdentitiesOnly=yes -i /home/ubuntu/.ssh/id_rsa -o BatchMode=yes"
STOP=$L/pull_ml02_rays.STOP; MINFREE=${MINFREE:-15}
free_gb() { df -BG --output=avail / | tail -1 | tr -dc '0-9'; }
$SSH $R "ls -d $RD/PalletModels/*/*/rays" | awk -F/ '{print $(NF-2)"/"$(NF-1)}' | sort -r > $L/ml02_rays_list.txt
echo "pull start $(date +%F' '%H:%M): ml-02 has $(wc -l < $L/ml02_rays_list.txt) picks with rays, $(free_gb) GB free"
n=0
for rp in $(cat $L/ml02_rays_list.txt); do
  [ -f $STOP ] && { echo "STOP file found"; exit 0; }
  [ "$(free_gb)" -lt $MINFREE ] && { echo "SPACE: $(free_gb) GB free, stopping"; exit 2; }
  r=${rp%%/*}; p=${rp#*/}; d=$ST/$r/$p
  [ -f $d/rays/cameras.json ] && [ "$(ls $d/rays/cam2_*.npz 2>/dev/null | wc -l)" -gt 0 ] && continue
  [ -f $d/PULL_FAILED ] && continue
  t0=$(date +%s); mkdir -p $d/rays_full $d/02
  rsync -a -e "$SSH" --include='cameras.json' --include='cam2_*.npz' --exclude='*' $R:$RD/PalletModels/$r/$p/rays/ $d/rays_full/ > /dev/null 2>&1 || { echo "$rp: rays rsync failed"; rm -rf $d/rays_full; continue; }
  rsync -a -e "$SSH" --include='color_*.png' --include='color_*.txt' --exclude='*' $R:$RD/PicksCopies/$r/$p/02/ $d/02/ > /dev/null 2>&1 || echo "$rp: image rsync failed"
  nimg=$(ls $d/02/color_*.png 2>/dev/null | wc -l); nfull=$(ls $d/rays_full/cam2_*.npz 2>/dev/null | wc -l)
  if [ $nfull -eq 0 ] || [ $nimg -eq 0 ]; then echo "$rp: $nfull tensors, $nimg images -> skipped"; echo "$nfull tensors $nimg images $(date +%FT%T)" > $d/PULL_FAILED; rm -rf $d/rays_full; continue; fi
  mkdir -p $d/tmp_src/$r/$p && mv $d/rays_full $d/tmp_src/$r/$p/rays
  out=$(cd $PD && .venv/bin/python -m pallet.compact --src-root $d/tmp_src --store $ST --picks $r/$p --min-age-min 0 2>&1 | tail -1)
  rm -rf $d/tmp_src
  echo "$rp: $nfull tensors, $nimg images, $(( $(date +%s)-t0 ))s | $out | $(free_gb) GB free $(date +%H:%M)"; n=$((n+1))
done
echo "=== DONE: $n picks $(date +%F' '%H:%M)"
