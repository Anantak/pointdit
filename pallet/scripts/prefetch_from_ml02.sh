#!/bin/bash
# Keep up to AHEAD raw episodes (from ana-ml-02) staged locally for rays_from_ml02.sh, in its todo order. Records what it fetched
# in ml02_fetched.txt so only those get deleted afterwards. Stops at the free-space guard. touch $STOP to end.
D=/home/ubuntu/Anantak/Data; L=$D/PicksCopies/_logs; R=ubuntu@192.168.5.193; RD=/media/ubuntu/ana-back-01/Work/Data
SSH="ssh -o IdentitiesOnly=yes -i /home/ubuntu/.ssh/id_rsa -o BatchMode=yes"; AHEAD=${AHEAD:-40}; MINFREE=${MINFREE:-26}; STOP=$L/prefetch_from_ml02.STOP
free_gb() { df -BG --output=avail / | tail -1 | tr -dc '0-9'; }
touch $L/ml02_fetched.txt
echo "prefetch start $(date +%F' '%H:%M), $(free_gb) GB free"
while true; do
  [ -f $STOP ] && { echo "STOP file found, exiting"; exit 0; }
  staged=$(grep -c . $L/ml02_fetched.txt 2>/dev/null); live=0
  for rp in $(cat $L/ml02_fetched.txt); do [ -d $D/PicksCopies/$rp ] && live=$((live+1)); done
  if [ $live -ge $AHEAD ] || [ "$(free_gb)" -lt $MINFREE ]; then sleep 60; continue; fi
  next=""
  for rp in $(cat $L/rays_todo.txt); do
    r=${rp%%/*}; d=${rp#*/}
    [ -f $D/PalletModels/$rp/rays/cameras.json ] || [ -f $D/PalletModels/$rp/RAYS_FAILED ] && continue
    [ -d $D/PicksCopies/$rp ] && continue
    grep -qx "$rp" $L/ml02_fetched.txt && continue
    next=$rp; break
  done
  [ -z "$next" ] && { echo "nothing left to prefetch $(date +%H:%M)"; exit 0; }
  t0=$(date +%s); tmp=$D/PicksCopies/${next}.part; mkdir -p $tmp
  if rsync -a -e "$SSH" $R:$RD/PicksCopies/$next/ $tmp/ > /dev/null 2>&1; then mv $tmp $D/PicksCopies/$next; echo "$next" >> $L/ml02_fetched.txt
    echo "fetched $next in $(( $(date +%s)-t0 ))s ($(du -sm $D/PicksCopies/$next | cut -f1) MB), staged $((live+1)), $(free_gb) GB free $(date +%H:%M)"
  else echo "FETCH FAILED $next"; rm -rf $tmp; sleep 30; fi
done
