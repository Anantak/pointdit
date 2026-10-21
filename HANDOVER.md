# HANDOVER (pointdit fork, branch `pallet-pose`) — 2026-10-21

This repo hosts **PalletDiT**: PointDiT adapted to predict 7 per-pixel pallet-frame channels from forklift forkview images, with
pixel voting for the pallet pose and a sequential (warm-started) estimator. The living handover is
**`docs/PALLET_POSE_HANDOVER.md`** (newest note first: state, resume options, machines, rules). Read its top note, then
`docs/PALLET_POINTDIT_PAPER.md` section VIII (status) and `docs/PALLET_CODE_GUIDE.md` when touching code.

Quick facts: best model `runs/b256_prior2`; evaluate only on `runs/fixed_val` (201 clean picks); the pallet-heading labels carry a
~1.3 deg per-pick yaw bias that limits lateral accuracy (critical path = fix the heading in the SensorUnit pallet fit);
ml-02 (ubuntu@192.168.5.193) has a synced repo + free 8 GB GPU; the laptop GPU is reserved for the user's map building.
