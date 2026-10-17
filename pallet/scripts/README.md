# Data-building helpers (laptop ana-ml-1)

Ray tensors for picks that already have a pallet model under `~/Anantak/Data/PalletModels` but no `rays/`, using the raw episodes
kept on ana-ml-02's drive (`/media/ubuntu/ana-back-01/Work/Data/PicksCopies`). Newest picks first, robots alternating.

- `prefetch_from_ml02.sh` — stages up to 40 raw episodes (~400 MB each) under `~/Anantak/Data/PicksCopies`, in the order of
  `_logs/rays_todo.txt`; records them in `_logs/ml02_fetched.txt`. Free-space guard 26 GB. `touch _logs/prefetch_from_ml02.STOP` to end.
- `rays_from_ml02.sh` — per pick: xt32 step `--rebuild` -> patch-cloud step -> `pallet_rays.py` -> copies the `02/` images to
  `~/Anantak/Data/PalletTrainImages/<robot>/<pick>/02/` -> deletes the raw copy (only if it fetched it) and strips the big clouds.
  The pipeline app needs the GPU (CUDA out of memory next to a training run), so it waits while a `pallet.train` process exists.
  Writes `RAYS_FAILED` into the model dir on failure (delete to retry). `touch _logs/rays_from_ml02.STOP` to end after the current pick.
- After new picks: `python3.11 ~/Anantak/SensorUnit/scripts/utils/pallet_manifest.py --episodes ~/Anantak/Data/PalletTrainImages
  --out ~/Anantak/Data/PalletTrainImages/manifest`, then train.

Logs: `~/Anantak/Data/PicksCopies/_logs/{rays,prefetch}_from_ml02_*.out`. Both scripts run detached with `setsid nohup`.
