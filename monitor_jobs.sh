#!/usr/bin/env bash
# monitor_jobs.sh — Watch background Forager ML jobs and alert on completion/failure.
#
# Monitors PIDs and log files every 60 seconds.
# Sends desktop notification (notify-send) and writes status to /tmp/forager_monitor.log
#
# Usage:
#   bash monitor_jobs.sh &       # run in background
#   kill %1                      # stop it
#
# Or check status at any time:
#   cat /tmp/forager_status.txt

STATUS_FILE="/tmp/forager_status.txt"
LOG_FILE="/tmp/forager_monitor.log"
CHECK_INTERVAL=60   # seconds

# ── Job definitions ────────────────────────────────────────────────────────────
# Format: "PID:name:logfile"
JOBS=(
    "1271469:medicinals_expert:/tmp/train_medicinals.log"
)

# Dataset paths for progress counting
declare -A DATASET_PATHS=(
    ["medicinals_download"]="medicinals_dataset"
    ["other_download"]="inat_dataset/other"
)
declare -A DATASET_TARGETS=(
    ["medicinals_download"]=76000
    ["other_download"]=19000
)

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Track which jobs we've already notified about
declare -A NOTIFIED

notify() {
    local title="$1"
    local body="$2"
    local urgency="${3:-normal}"
    echo "[$(date '+%H:%M:%S')] NOTIFY: $title — $body" >> "$LOG_FILE"
    # Try desktop notification (works if a display session is active)
    DISPLAY=:0 notify-send --urgency="$urgency" --app-name="ForagerML" "$title" "$body" 2>/dev/null || true
    # Also write to terminal if attached
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $title"
    echo "  $body"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

count_images() {
    local path="$REPO_ROOT/$1"
    if [[ -d "$path" ]]; then
        find "$path" -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" 2>/dev/null | wc -l
    else
        echo 0
    fi
}

check_log_for_errors() {
    local logfile="$1"
    if [[ -f "$logfile" ]]; then
        # Look for Python tracebacks, CUDA errors, OOM, etc.
        if grep -qiE "(traceback|error:|exception:|killed|oom|cuda error|segfault|segmentation fault)" "$logfile" 2>/dev/null; then
            # Get the last error context
            grep -iE "(traceback|error:|exception:|killed|oom|cuda error)" "$logfile" 2>/dev/null | tail -3
            return 1  # errors found
        fi
    fi
    return 0  # no errors
}

write_status() {
    {
        echo "Forager ML Job Monitor — $(date '+%Y-%m-%d %H:%M:%S')"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        for job_spec in "${JOBS[@]}"; do
            IFS=: read -r pid name logfile <<< "$job_spec"
            if kill -0 "$pid" 2>/dev/null; then
                status="RUNNING"
                # Show progress for download jobs
                if [[ -n "${DATASET_PATHS[$name]}" ]]; then
                    count=$(count_images "${DATASET_PATHS[$name]}")
                    target="${DATASET_TARGETS[$name]}"
                    pct=$(( count * 100 / (target > 0 ? target : 1) ))
                    status="RUNNING  ${count}/${target} (${pct}%)"
                fi
            else
                # Check exit code via wait if possible, else check log
                if [[ -n "${NOTIFIED[$name]}" ]]; then
                    status="${NOTIFIED[$name]}"
                else
                    # Check if log has errors
                    if [[ -f "$logfile" ]] && ! check_log_for_errors "$logfile" 2>/dev/null; then
                        status="FAILED (check log)"
                    else
                        status="DONE"
                    fi
                fi
            fi
            printf "  %-25s %s\n" "$name" "$status"
        done
        echo ""
        echo "Log: $LOG_FILE"
        echo "Monitor PID: $$"
    } > "$STATUS_FILE"
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] monitor_jobs.sh started (PID $$)" >> "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Watching jobs: ${JOBS[*]}" >> "$LOG_FILE"
echo "Monitor started. Status at: $STATUS_FILE"
echo "Log at: $LOG_FILE"
echo ""

notify "Forager ML Monitor" "Watching $(echo ${#JOBS[@]}) jobs. Status: cat $STATUS_FILE" "low"

while true; do
    write_status

    for job_spec in "${JOBS[@]}"; do
        IFS=: read -r pid name logfile <<< "$job_spec"

        # Skip already-notified jobs
        [[ -n "${NOTIFIED[$name]}" ]] && continue

        if ! kill -0 "$pid" 2>/dev/null; then
            # Process ended — determine success vs failure
            echo "[$(date '+%H:%M:%S')] PID $pid ($name) no longer running" >> "$LOG_FILE"

            # Check log for errors
            error_lines=""
            if [[ -f "$logfile" ]]; then
                error_lines=$(grep -iE "(traceback|error:|exception:|killed|oom|cuda error|segfault)" "$logfile" 2>/dev/null | tail -3 || true)
            fi

            if [[ -n "$error_lines" ]]; then
                NOTIFIED[$name]="FAILED"
                notify "Forager ML — FAILED" "$name FAILED. Check: tail -50 $logfile" "critical"
                echo "[$(date '+%H:%M:%S')] $name FAILED. Errors: $error_lines" >> "$LOG_FILE"
            else
                # Check if it looks like a success
                NOTIFIED[$name]="DONE"

                # Count final images for download jobs
                if [[ -n "${DATASET_PATHS[$name]}" ]]; then
                    count=$(count_images "${DATASET_PATHS[$name]}")
                    target="${DATASET_TARGETS[$name]}"
                    notify "Forager ML — Done" "$name complete: ${count}/${target} images" "normal"
                    NOTIFIED[$name]="DONE (${count} images)"
                else
                    # For training jobs, look for success signal in log
                    if [[ -f "$logfile" ]] && grep -q "Training complete\|complete\|Best val accuracy" "$logfile" 2>/dev/null; then
                        best=$(grep "Best val accuracy\|best:" "$logfile" 2>/dev/null | tail -1 || echo "check log")
                        notify "Forager ML — Done" "$name complete. $best" "normal"
                        NOTIFIED[$name]="DONE"
                    else
                        notify "Forager ML — Ended" "$name ended (check log to verify success)" "normal"
                        NOTIFIED[$name]="ENDED (verify)"
                    fi
                fi
                echo "[$(date '+%H:%M:%S')] $name finished. Status: ${NOTIFIED[$name]}" >> "$LOG_FILE"
            fi
        fi
    done

    # If all jobs are done, write final status and exit
    all_done=true
    for job_spec in "${JOBS[@]}"; do
        IFS=: read -r pid name logfile <<< "$job_spec"
        kill -0 "$pid" 2>/dev/null && all_done=false
    done

    if $all_done && [[ ${#NOTIFIED[@]} -eq ${#JOBS[@]} ]]; then
        write_status
        notify "Forager ML — All Jobs Complete" "All background jobs finished. Check $STATUS_FILE" "normal"
        echo "[$(date '+%H:%M:%S')] All jobs complete. Monitor exiting." >> "$LOG_FILE"
        exit 0
    fi

    sleep $CHECK_INTERVAL
done
