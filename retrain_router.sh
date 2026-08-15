#!/usr/bin/env bash
# retrain_router.sh — Wait for "other" class download, rebuild router dataset,
# retrain the domain router with EfficientNet Lite2.
#
# The domain router is the critical first-pass OOD gate in the inference pipeline.
# The current YOLO-based router was trained without the "other" class — this script
# rebuilds it properly with all 4 classes (berry / mushroom / plant / other).
#
# Prerequisites:
#   data/acquisition/other_pull_inat.py must have run.
#
# Run from repo root:
#   bash retrain_router.sh
#
# Safe to start at any time: if no downloads are running it goes straight to the
# rebuild; if some are, it blocks until they finish.

set -euo pipefail
export CUDA_VISIBLE_DEVICES=1

echo "========================================================"
echo "  retrain_router.sh — waiting for 'other' download..."
echo "========================================================"

# Wait on WHAT IS RUNNING, not on PID literals (this previously polled 1258691
# and 1255794, both long dead — after which the loops exit instantly and the
# rebuild runs on whatever happens to be on disk). The `kill -0` polling here
# was at least correct in form, unlike retrain_v2.sh's `wait`, which cannot work
# on a non-child process at all.
#
# `ops.status --wait-for` uses the same live-command-line detector as the status
# readout and the watchdog, so it needs no PIDs and catches any downloader —
# other_pull_inat, medicinals_pull_inat, mushroom_observer_pull — including ones
# started after this script did.
python -m ops.status --wait-for download --poll 30

echo ""
echo "Downloads complete. Rebuilding router dataset..."
python training/scripts/build_router_dataset.py

echo ""
echo "========================================================"
echo "  Training domain router v2 (EfficientNet Lite2)"
echo "========================================================"
python training/scripts/train_domain_router.py \
    --dataset router_dataset \
    --name domain_router_v2 \
    --epochs 60

echo ""
echo "========================================================"
echo "  Benchmarking router v2"
echo "========================================================"
python training/scripts/benchmark_router.py \
    --checkpoint runs/efficientnet/domain_router_v2/best.pt \
    --dataset router_dataset

echo ""
echo "========================================================"
echo "  retrain_router.sh complete"
echo "========================================================"
echo ""
echo "  Next steps:"
echo "  1. Export router to ONNX:    python training/scripts/export_efficientnet_onnx.py \\"
echo "                                   --checkpoint runs/efficientnet/domain_router_v2/best.pt"
echo "  2. Compile HEF:              python training/scripts/compile_efficientnet_hef.py \\"
echo "                                   --onnx inference/onnx_staging/domain_router_v2.onnx"
echo "  3. Copy HEF to inference:    cp inference/onnx_staging/domain_router_v2.hef inference/models/"
echo "  4. Update runner.py:         change router HEF path to domain_router_v2.hef"
