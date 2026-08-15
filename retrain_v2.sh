#!/usr/bin/env bash
# retrain_v2.sh — Wait for any running dataset downloads, rebuild splits, then
# retrain the experts with improved augmentation (RandAugment, MixUp, weighted
# loss).
#
# Run from repo root:
#   bash retrain_v2.sh
#
# Safe to start at any time: if no downloads are running it goes straight to the
# retrain; if some are, it blocks until they finish.

set -euo pipefail
export CUDA_VISIBLE_DEVICES=1

echo "========================================================"
echo "  retrain_v2.sh — waiting for downloads to finish..."
echo "========================================================"

# Wait on WHAT IS RUNNING, not on PID literals. The previous version was:
#
#     for pid in 1238285 1238290 1238583; do
#         if kill -0 "$pid"; then wait "$pid" || true; fi
#     done
#
# broken two independent ways. Those PIDs went stale at the first reboot, after
# which `kill -0` fails, the guard falls through, and training starts instantly
# on a half-downloaded dataset while looking like it worked. And even while the
# PIDs were alive, `wait` only works on children of the calling shell — those
# downloads were started separately — so it errored out immediately and
# `|| true` swallowed it. That wait never waited, on any run.
#
# `ops.status --wait-for` polls the same live-command-line detector the status
# readout and the watchdog use, so it needs no PIDs and also catches downloads
# started after this script did.
python -m ops.status --wait-for download --poll 60

echo ""
echo "Downloads complete. Rebuilding dataset splits..."

# Psychedelics split (only — berry and highvalue unaffected unless you re-run those too)
python training/scripts/rebuild_dataset_splits.py --only psychedelics
python training/scripts/rebuild_dataset_splits.py --only berry

echo ""
echo "========================================================"
echo "  Retraining psychedelics_expert  (v2)"
echo "========================================================"
python training/scripts/train_efficientnet_specialist.py \
    --dataset psychedelics_dataset_split \
    --name psychedelics_expert \
    --epochs 60

echo ""
echo "========================================================"
echo "  Retraining berry_expert  (v2)"
echo "========================================================"
python training/scripts/train_efficientnet_specialist.py \
    --dataset berry_dataset_split \
    --name berry_expert \
    --epochs 60

echo ""
echo "========================================================"
echo "  highvalue_expert already strong — skipping retrain"
echo "========================================================"

echo ""
echo "========================================================"
echo "  Running benchmarks"
echo "========================================================"
for expert in psychedelics berry; do
    python training/scripts/benchmark_expert.py \
        --checkpoint "runs/efficientnet/${expert}_expert/best.pt" \
        --dataset "${expert}_dataset_split"
done

echo ""
echo "========================================================"
echo "  retrain_v2.sh complete"
echo "========================================================"
