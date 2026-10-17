#!/bin/bash
# Every 10 min: compact finished local ray tensors (PalletModels/*/*/rays, older than 2 min) into the store (copy mode: originals kept until the user approves deletion).
PD=/home/ubuntu/Libraries/GoogleResearch/pointdit; L=/home/ubuntu/Anantak/Data/PicksCopies/_logs; STOP=$L/compact_loop.STOP
while [ ! -f $STOP ]; do
  cd $PD && .venv/bin/python -m pallet.compact --src-root ~/Anantak/Data/PalletModels 2>&1 | tail -1 | sed "s/^/$(date +%H:%M) /"
  sleep 600
done
echo "STOP file found"
