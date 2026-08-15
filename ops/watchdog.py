"""Edge-triggered watchdog for long-running forager_ml jobs.

Ported from hestia's `deploy/watchdog/hestia-watchdog.sh`, keeping the three
properties that make that one worth having and dropping the one that does not
transfer:

  KEPT  Edge-triggered. One notification per transition, silence otherwise, so
        it never becomes noise you learn to ignore.
  KEPT  State on disk, one file per check, so no check can mask another.
  KEPT  Required config with no baked-in defaults. A scrubbed placeholder
        deploys silently and looks like it is working.
  DROPPED  Running off-site. Hestia's watchdog probes the house from the dedi
        because the failure it catches is "the house is dark", which a process
        inside the house cannot report. forager_ml's jobs are local training
        runs; a local watcher is the right shape, and the box itself is already
        covered by hestia's off-site probe.

It consumes `ops.status.snapshot()` rather than probing again — one collector,
two consumers, the same rule hestia's status tool follows.

What it replaces: `monitor_jobs.sh`, a 60-second `while true` loop over a
hardcoded PID array. That design had three failure modes this one cannot have:
the PIDs went stale at the first reboot, the loop itself was a process that had
to be remembered and could silently die, and it exited once its listed jobs
finished, so it never saw the next one.

    python -m ops.watchdog            # one pass; run it from a timer
    python -m ops.watchdog --dry-run  # print what it would send, send nothing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .status import classify_log, snapshot

STATE_DIR = Path(os.environ.get(
    "FORAGER_WATCHDOG_STATE_DIR",
    Path.home() / ".local/state/forager-watchdog",
))

#: Where job logs are looked for when a run ends. Matched by name substring.
LOG_DIR = Path(os.environ.get("FORAGER_LOG_DIR", "/tmp"))

#: Free-space floor. Long downloads and checkpoint writes die badly on a full
#: disk, and the box holds ~1.8TB of datasets, so this is not hypothetical.
DISK_FLOOR_PCT = float(os.environ.get("FORAGER_DISK_FLOOR_PCT", "10"))

#: A job whose checkpoint has not advanced in this long is treated as stalled.
#: A hung run looks exactly like a healthy one to a process check.
STALL_HOURS = float(os.environ.get("FORAGER_STALL_HOURS", "6"))


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M UTC")


def notify(title: str, body: str, priority: str = "default",
           tags: str = "package", *, dry_run: bool = False) -> None:
    line = f"[{priority}] {title} — {body}"
    if dry_run:
        # Deliberately before the config check, so --dry-run is usable on a box
        # where the unit's environment is not set yet.
        print(f"WOULD SEND {line}")
        return
    url = os.environ.get("FORAGER_NTFY_URL")
    if not url:
        sys.exit("set FORAGER_NTFY_URL (e.g. https://ntfy.sh/<random-topic>) in the unit")
    req = urllib.request.Request(
        url, data=body.encode(),
        headers={"Title": title, "Priority": priority, "Tags": tags},
    )
    try:
        urllib.request.urlopen(req, timeout=10).read()
        print(f"SENT {line}")
    except (urllib.error.URLError, OSError) as e:
        # Never fail the pass because the phone was unreachable: the next tick
        # still has its state, and a crashed watchdog notifies nobody of anything.
        print(f"NOTIFY FAILED ({e}) for: {line}", file=sys.stderr)


def _read_state(name: str) -> dict:
    f = STATE_DIR / f"{name}.json"
    try:
        return json.loads(f.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _write_state(name: str, data: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / f"{name}.json").write_text(json.dumps(data, indent=2))


def _find_log(job_name: str) -> Path | None:
    """Newest log in LOG_DIR whose filename mentions the job. Best effort."""
    if not LOG_DIR.is_dir():
        return None
    hits = [p for p in LOG_DIR.glob("*.log") if job_name in p.name]
    return max(hits, key=lambda p: p.stat().st_mtime, default=None)


# --- checks ----------------------------------------------------------------

def check_jobs(snap: dict, *, dry_run: bool) -> None:
    """Notify when a job that WAS running is no longer running.

    The job set is derived fresh each pass from live command lines, so a job
    started after this watchdog was installed is picked up with no config
    change — the thing the hardcoded JOBS array could never do.
    """
    prev = _read_state("jobs").get("running", {})
    now = {j["name"]: j for j in snap["jobs"]}

    for name, was in prev.items():
        if name in now:
            continue
        log = _find_log(name)
        verdict, evidence = classify_log(log) if log else ("unknown", "no log found")
        ran_h = round(was.get("elapsed_s", 0) / 3600, 1)

        if verdict == "failed":
            notify(f"forager_ml: {name} FAILED",
                   f"After {ran_h}h. {evidence}\nLog: {log}\nChecked {_now()}.",
                   "high", "warning", dry_run=dry_run)
        elif verdict == "ok":
            notify(f"forager_ml: {name} finished",
                   f"Ran {ran_h}h, no errors in the log. {evidence}\nChecked {_now()}.",
                   "default", "white_check_mark", dry_run=dry_run)
        else:
            # Explicitly NOT reported as success. A job that vanished with no
            # readable log is the case most worth a human glance, and calling it
            # "done" is how a failed overnight run gets believed.
            notify(f"forager_ml: {name} ended, unverified",
                   f"Ran {ran_h}h. No log found in {LOG_DIR} — verify before trusting the "
                   f"checkpoint.\nChecked {_now()}.",
                   "default", "grey_question", dry_run=dry_run)

    for name in now:
        if name not in prev:
            print(f"watchdog: {name} started (no alert on start)")

    _write_state("jobs", {"running": now, "ts": snap["ts"]})


def check_stalled(snap: dict, *, dry_run: bool) -> None:
    """A running job whose checkpoint has not advanced in STALL_HOURS.

    A hung run holds the GPU and looks alive to any process check, which is
    exactly why the old monitor could not see it.
    """
    by_name = {r["name"]: r for r in snap["runs"]}
    prev = _read_state("stalled").get("stalled", [])
    stalled = []

    for job in snap["jobs"]:
        run = by_name.get(job["name"])
        if not run or run["checkpoint_age_h"] is None:
            continue
        # Only meaningful once the job has been up longer than the stall window;
        # a fresh run legitimately has an old checkpoint from its last training.
        if job["elapsed_s"] / 3600 < STALL_HOURS:
            continue
        if run["checkpoint_age_h"] >= STALL_HOURS:
            stalled.append(job["name"])

    for name in stalled:
        if name not in prev:
            notify(f"forager_ml: {name} may be stalled",
                   f"Process alive but no checkpoint written in {STALL_HOURS}h. "
                   f"Checked {_now()}.",
                   "high", "warning", dry_run=dry_run)
    for name in prev:
        if name not in stalled and name in {j["name"] for j in snap["jobs"]}:
            notify(f"forager_ml: {name} progressing again",
                   f"Checkpoint advanced. Checked {_now()}.",
                   "default", "white_check_mark", dry_run=dry_run)

    _write_state("stalled", {"stalled": stalled, "ts": snap["ts"]})


def check_disk(snap: dict, *, dry_run: bool) -> None:
    free_pct = snap["disk"]["free_pct"]
    state = "low" if free_pct is not None and free_pct < DISK_FLOOR_PCT else "ok"
    prev = _read_state("disk").get("state", "ok")

    if state == "low" and prev != "low":
        notify("forager_ml: disk is low",
               f"{snap['disk']['free_gb']} GB free ({free_pct}%), below the "
               f"{DISK_FLOOR_PCT}% floor. Training checkpoints and dataset pulls "
               f"will fail badly. Checked {_now()}.",
               "urgent", "rotating_light", dry_run=dry_run)
    elif state == "ok" and prev == "low":
        notify("forager_ml: disk recovered",
               f"{snap['disk']['free_gb']} GB free ({free_pct}%). Checked {_now()}.",
               "default", "white_check_mark", dry_run=dry_run)

    _write_state("disk", {"state": state, "free_pct": free_pct, "ts": snap["ts"]})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="forager_ml job watchdog (one pass)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print notifications instead of sending them; state is still written")
    args = ap.parse_args(argv)

    snap = snapshot()
    check_jobs(snap, dry_run=args.dry_run)
    check_stalled(snap, dry_run=args.dry_run)
    check_disk(snap, dry_run=args.dry_run)

    print(f"watchdog: {len(snap['jobs'])} job(s) running, "
          f"disk {snap['disk']['free_pct']}% free, state in {STATE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
