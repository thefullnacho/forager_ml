#!/usr/bin/env bash
# retrain_v2.sh — Wait for downloads, rebuild splits, retrain all three experts
# with improved augmentation (RandAugment, MixUp, weighted loss).
#
# Run from repo root:
#   bash retrain_v2.sh
#
# Downloads (PIDs 1238285, 1238290, 1238583) should already be running.
# This script blocks until they finish, then proceeds.

set -euo pipefail
export CUDA_VISIBLE_DEVICES=1

echo "========================================================"
echo "  retrain_v2.sh — waiting for downloads to finish..."
echo "========================================================"

# Wait for all download processes to finish
# 1238285 = berry_pull_inat (poison_ivy +3k, virginia_creeper +3k)
# 1238290 = psychedelic_pull_inat (azurescens + caerulipes needs_id)
# 1238583 = mushroom_observer_pull (azurescens + caerulipes MO images)
for pid in 1238285 1238290 1238583; do
    if kill -0 "$pid" 2>/dev/null; then
        echo "  Waiting for PID $pid..."
        wait "$pid" || true
    fi
done

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
