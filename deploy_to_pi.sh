#!/usr/bin/env bash
# deploy_to_pi.sh — Sync Forager inference code + models to the Pi.
#
# Usage:
#   bash deploy_to_pi.sh                        # uses PI_HOST default
#   bash deploy_to_pi.sh pi@192.168.1.42        # explicit host
#   bash deploy_to_pi.sh pi@192.168.1.42 --dry  # dry run (shows what would change)
#
# What this syncs:
#   inference/main.py
#   inference/pipeline/          (all .py files)
#   inference/models/            (.hef + _classes.json + _energy.json only)
#   inference/requirements_pi.txt
#
# What it does NOT sync:
#   inference/hailo_dfc/         (large SDK, already on Pi)
#   inference/onnx_staging/      (not needed at runtime)
#   inference/models/*.har       (intermediate, not needed)
#   Any dataset or training artefacts

set -euo pipefail

PI_HOST="${1:-pi@forager.local}"
DRY=""
[[ "${2:-}" == "--dry" ]] && DRY="--dry-run"

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
INFERENCE_DIR="$REPO_ROOT/inference"
REMOTE_DIR="/home/pi/forager"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Forager deploy → $PI_HOST"
echo "  Remote: $REMOTE_DIR"
[[ -n "$DRY" ]] && echo "  Mode: DRY RUN"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Check reachability ────────────────────────────────────────────────────────
if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$PI_HOST" true 2>/dev/null; then
    echo ""
    echo "  ERROR: Cannot reach $PI_HOST"
    echo "  Try:"
    echo "    bash deploy_to_pi.sh pi@<IP_ADDRESS>"
    echo "    bash deploy_to_pi.sh pi@raspberrypi.local"
    exit 1
fi
echo "  ✓ SSH connection OK"

# ── Ensure remote directory exists ───────────────────────────────────────────
ssh "$PI_HOST" "mkdir -p $REMOTE_DIR/models $REMOTE_DIR/pipeline"

# ── Sync Python source ────────────────────────────────────────────────────────
echo ""
echo "  Syncing pipeline code ..."
rsync -avz --progress $DRY \
    --include="main.py" \
    --include="requirements_pi.txt" \
    --exclude="__pycache__/" \
    --exclude="*.pyc" \
    --exclude="hailo_dfc/" \
    --exclude="onnx_staging/" \
    --exclude="models/" \
    --exclude="*" \
    "$INFERENCE_DIR/" "$PI_HOST:$REMOTE_DIR/"

rsync -avz --progress $DRY \
    --include="*.py" \
    --exclude="__pycache__/" \
    --exclude="*.pyc" \
    --exclude="*" \
    "$INFERENCE_DIR/pipeline/" "$PI_HOST:$REMOTE_DIR/pipeline/"

# ── Sync models (HEFs + manifests only, skip large .har files) ───────────────
echo ""
echo "  Syncing models ..."
rsync -avz --progress $DRY \
    --include="*.hef" \
    --include="*_classes.json" \
    --include="*_energy.json" \
    --exclude="*.har" \
    --exclude="*.onnx" \
    "$INFERENCE_DIR/models/" "$PI_HOST:$REMOTE_DIR/models/"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [[ -n "$DRY" ]]; then
    echo "  DRY RUN complete — no files transferred."
else
    echo "  Deploy complete."
    echo ""
    echo "  Verify on Pi:"
    echo "    ssh $PI_HOST"
    echo "    ls $REMOTE_DIR/models/*.hef"
    echo ""
    echo "  Run (no display/voice for SSH test):"
    echo "    ssh $PI_HOST 'cd $REMOTE_DIR && python main.py --no-display --no-voice --no-tts'"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
