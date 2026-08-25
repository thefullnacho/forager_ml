#!/usr/bin/env bash
# status.sh — thin shim. The readout lives in ops/status.py so the watchdog can
# consume the SAME snapshot() instead of probing the box a second time.
#
#   bash status.sh              human readout
#   bash status.sh --json       the raw snapshot
#
# The previous version of this file hardcoded dataset targets (76000 / 19000)
# and printed a percentage against them. Both downloads have long since
# finished, so that percentage was a fossil that read as live progress.
exec python3 -m ops.status "$@"
