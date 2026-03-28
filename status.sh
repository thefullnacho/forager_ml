#!/usr/bin/env bash
# status.sh — Quick status check for all Forager ML background jobs.
# Works locally or over SSH.
#
# Usage:
#   bash status.sh

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Forager ML — Job Status  ($(date '+%H:%M:%S %Z'))"
echo "═══════════════════════════════════════════════════════"

# ── Active processes ────────────────────────────────────────────────────────
echo ""
echo "  Active Python jobs:"
ps aux | grep "[p]ython" | awk '{print "    PID " $2 ": " substr($0, index($0,$11))}' | head -10

# ── Download progress ───────────────────────────────────────────────────────
echo ""
echo "  Download progress:"

medicinals_count=$(find "$REPO_ROOT/medicinals_dataset" -name "*.jpg" 2>/dev/null | wc -l)
printf "    %-30s %6d / 76000  (%d%%)\n" "medicinals_dataset" \
    "$medicinals_count" "$((medicinals_count * 100 / 76000))"

other_count=$(find "$REPO_ROOT/inat_dataset/other" -name "*.jpg" 2>/dev/null | wc -l)
printf "    %-30s %6d / 19000  (%d%%)\n" "inat_dataset/other" \
    "$other_count" "$((other_count * 100 / 19000))"

# ── Per-class breakdown for medicinals ─────────────────────────────────────
if [[ -d "$REPO_ROOT/medicinals_dataset" ]]; then
    echo ""
    echo "  Medicinals per class:"
    for d in "$REPO_ROOT/medicinals_dataset"/*/; do
        n=$(ls "$d" 2>/dev/null | wc -l)
        printf "    %-35s %d\n" "$(basename "$d")" "$n"
    done
fi

# ── Router retrain log tail ─────────────────────────────────────────────────
echo ""
echo "  retrain_router.sh (last 5 lines):"
if [[ -f /tmp/retrain_router.log ]]; then
    tail -5 /tmp/retrain_router.log | sed 's/^/    /'
else
    echo "    (log not found)"
fi

# ── Monitor log tail ────────────────────────────────────────────────────────
echo ""
echo "  Monitor events:"
if [[ -f /tmp/forager_monitor.log ]]; then
    tail -8 /tmp/forager_monitor.log | sed 's/^/    /'
else
    echo "    (monitor not running)"
fi

# ── Disk ─────────────────────────────────────────────────────────────────────
echo ""
echo "  Disk:"
df -h "$REPO_ROOT" | tail -1 | awk '{printf "    %s used of %s  (%s free)\n", $3, $2, $4}'

echo ""
echo "═══════════════════════════════════════════════════════"
echo ""
