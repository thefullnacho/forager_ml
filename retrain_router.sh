#!/usr/bin/env bash
# retrain_router.sh — Wait for "other" class download, rebuild router dataset,
# retrain the domain router with EfficientNet Lite2.
#
# The domain router is the critical first-pass OOD gate in the inference pipeline.
# The current YOLO-based router was trained without the "other" class — this script
# rebuilds it properly with all 4 classes (berry / mushroom / plant / other).
#
# Prerequisites:
#   data/acquisition/other_pull_inat.py must have run (PID 1258691).
#
# Run from repo root:
#   bash retrain_router.sh

set -euo pipefail
export CUDA_VISIBLE_DEVICES=1

echo "========================================================"
echo "  retrain_router.sh — waiting for 'other' download..."
echo "========================================================"

# Wait for other class download (PID 1258691)
# Note: use kill -0 polling — wait only works on child processes of this shell
OTHER_PID=1258691
while kill -0 "$OTHER_PID" 2>/dev/null; do
    count=$(find inat_dataset/other -name "*.jpg" 2>/dev/null | wc -l)
    echo "  Waiting for other_pull_inat.py (PID $OTHER_PID) — ${count}/19000 images..."
    sleep 30
done
echo "  other_pull_inat.py done."

# Also wait for medicinals download if still running (PID 1255794)
MED_PID=1255794
while kill -0 "$MED_PID" 2>/dev/null; do
    count=$(find medicinals_dataset -name "*.jpg" 2>/dev/null | wc -l)
    echo "  Waiting for medicinals_pull_inat.py (PID $MED_PID) — ${count}/76000 images..."
    sleep 60
done
echo "  medicinals_pull_inat.py done."

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
