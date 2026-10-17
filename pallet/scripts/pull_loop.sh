#!/bin/bash
# Re-run pull_ml02_rays.sh every 30 min so picks built on ana-ml-02 keep flowing into the store. touch $STOP to end.
L=/home/ubuntu/Anantak/Data/PicksCopies/_logs; STOP=$L/pull_loop.STOP; PD=/home/ubuntu/Libraries/GoogleResearch/pointdit
while [ ! -f $STOP ]; do
  $PD/pallet/scripts/pull_ml02_rays.sh > $L/pull_ml02_rays_$(date +%Y%m%d_%H%M).out 2>&1
  echo "$(date +%H:%M) $(tail -1 $(ls -t $L/pull_ml02_rays_*.out | head -1) | cut -c1-80); store $(ls -d /home/ubuntu/Anantak/Data/PalletTrainImages/*/*/rays | wc -l) picks"
  sleep 1800
done
echo "STOP file found"
