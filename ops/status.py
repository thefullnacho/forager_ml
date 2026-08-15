"""One health readout for the whole forager_ml box.

Design, ported from hestia's `brain/tools/status.py`: a ``snapshot()`` data layer
returning a plain dict, plus a ``render()`` formatter. The watchdog consumes the
SAME ``snapshot()`` rather than probing again, so "is a job still running?" is
answered by one collector and never implemented twice.

What this replaces: the old ``status.sh`` + ``monitor_jobs.sh`` pair, which
identified jobs by **hardcoded PID literals** written down during one session
(``JOBS=("1271469:medicinals_expert:...")``). After any reboot those PIDs are
either dead or, worse, belong to something else. Nothing here stores a PID
between runs: state is derived every time from live processes matched by command
line, and from artifacts on disk.

    python -m ops.status            # human readout
    python -m ops.status --json     # the same snapshot, machine-readable
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Training entry points, and how to pull a run name out of their command line.
#: Matched against full command lines, so a run is identified by what it IS doing
#: rather than by a PID someone pasted into a script.
_TRAINERS = {
    "train_efficientnet_specialist.py": "specialist",
    "train_domain_router.py": "router",
}

#: Dataset acquisition scripts. Same idea: recognised by name, not by PID.
_DOWNLOADERS = ("_pull_inat.py", "mushroom_observer_pull.py", "fetch_illustrations.py",
                "fetch_line_drawings.py")

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")

#: Log lines that mean a finished run did not finish well. Deliberately broad:
#: a false "check this" costs a glance, a missed CUDA OOM costs a night.
_FAILURE_PAT = re.compile(
    r"traceback|error:|exception:|killed|out of memory|cuda error|segmentation fault",
    re.IGNORECASE,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run(cmd: list[str], timeout: float = 5.0) -> str:
    """Best-effort shell out. A missing tool degrades to empty, never raises."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


# --- processes -------------------------------------------------------------

def _ps_lines() -> list[tuple[int, int, str]]:
    """(pid, elapsed_seconds, command) for every process, or [] if ps is unusable."""
    out = _run(["ps", "-eo", "pid=,etimes=,args="])
    rows = []
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            rows.append((int(parts[0]), int(parts[1]), parts[2]))
        except ValueError:
            continue
    return rows


def _arg_value(cmd: str, flag: str) -> str | None:
    m = re.search(rf"{re.escape(flag)}[= ]+(\S+)", cmd)
    return m.group(1) if m else None


def _runs_script(cmd: str, script: str) -> bool:
    """True only when this really is `python .../<script> ...`.

    A plain substring test matches far too much: the shell running a command
    that mentions the script, a grep for it, an editor with it open, this
    watchdog's own argv. All of those would be counted as a live training run.
    So require both an interpreter in argv[0] and the script as a whole argv
    token, not as a substring of the line.
    """
    tokens = cmd.split()
    if not tokens:
        return False
    if not Path(tokens[0]).name.startswith(("python", "uv")):
        return False
    return any(Path(t).name == script for t in tokens[1:])


def live_jobs() -> list[dict]:
    """Training and download jobs running RIGHT NOW, found by command line.

    This is the whole point of the rewrite: no PID is ever persisted, so the
    readout cannot go stale across a reboot the way the old JOBS array did.
    """
    jobs = []
    for pid, etimes, cmd in _ps_lines():
        for script, kind in _TRAINERS.items():
            if _runs_script(cmd, script):
                jobs.append({
                    "kind": kind,
                    "name": _arg_value(cmd, "--name") or script,
                    "dataset": _arg_value(cmd, "--dataset"),
                    "epochs": _arg_value(cmd, "--epochs"),
                    "pid": pid,
                    "elapsed_s": etimes,
                })
                break
        else:
            script_name = next(
                (t for t in cmd.split()[1:]
                 if t.endswith(".py") and any(d in Path(t).name for d in _DOWNLOADERS)),
                None,
            )
            if script_name and Path(cmd.split()[0]).name.startswith(("python", "uv")):
                jobs.append({
                    "kind": "download",
                    "name": Path(script_name).stem,
                    "dataset": None,
                    "epochs": None,
                    "pid": pid,
                    "elapsed_s": etimes,
                })
    return sorted(jobs, key=lambda j: -j["elapsed_s"])


# --- artifacts -------------------------------------------------------------

def training_runs() -> list[dict]:
    """Every run directory under runs/, with its checkpoint age and best metric.

    A run is 'active' when a live job carries the same --name. That pairing is
    what makes "did the overnight retrain actually finish?" answerable without
    remembering a PID.
    """
    active = {j["name"] for j in live_jobs()}
    runs = []
    for family in sorted((REPO_ROOT / "runs").glob("*")):
        if not family.is_dir():
            continue
        for run_dir in sorted(p for p in family.iterdir() if p.is_dir()):
            ckpt = run_dir / "best.pt"
            entry = {
                "family": family.name,
                "name": run_dir.name,
                "active": run_dir.name in active,
                "checkpoint": None,
                "checkpoint_age_h": None,
                "best_metric": None,
            }
            if ckpt.exists():
                st = ckpt.stat()
                entry["checkpoint"] = str(ckpt.relative_to(REPO_ROOT))
                entry["checkpoint_age_h"] = round((time.time() - st.st_mtime) / 3600, 1)
            for bench in ("benchmark.json", "benchmark_router.json"):
                bp = run_dir / bench
                if bp.exists():
                    try:
                        data = json.loads(bp.read_text())
                    except (OSError, json.JSONDecodeError):
                        continue
                    for key in ("top1", "top1_accuracy", "accuracy", "val_acc", "best_val_acc"):
                        if isinstance(data.get(key), (int, float)):
                            entry["best_metric"] = {key: data[key]}
                            break
            runs.append(entry)
    return runs


def datasets() -> list[dict]:
    """Image counts per dataset directory.

    No target totals are baked in. The old status.sh hardcoded 76000 and 19000
    and printed a percentage against them; both downloads have since completed,
    so the percentages were a fossil that read as progress.
    """
    out = []
    # dict.fromkeys de-dupes while keeping order: inat_dataset matches both globs.
    seen = dict.fromkeys(sorted(REPO_ROOT.glob("*_dataset")) + sorted(REPO_ROOT.glob("inat_dataset")))
    for d in seen:
        if not d.is_dir():
            continue
        n = sum(1 for p in d.rglob("*") if p.suffix.lower() in _IMG_EXTS)
        newest = max((p.stat().st_mtime for p in d.rglob("*") if p.is_file()), default=0)
        out.append({
            "name": d.name,
            "images": n,
            "classes": sum(1 for p in d.iterdir() if p.is_dir()),
            "idle_h": round((time.time() - newest) / 3600, 1) if newest else None,
        })
    return out


# --- machine ---------------------------------------------------------------

def gpus() -> list[dict]:
    raw = _run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits"])
    out = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            used, total = int(parts[2]), int(parts[3])
        except ValueError:
            continue
        out.append({
            "index": int(parts[0]), "name": parts[1],
            "mem_used_mib": used, "mem_total_mib": total,
            "mem_pct": round(100 * used / total, 1) if total else None,
            "util_pct": int(parts[4]) if parts[4].isdigit() else None,
        })
    return out


def gpu_holders() -> list[dict]:
    """Which processes hold VRAM. Answers 'is anything unexpected on the GPUs?'

    Straight from hestia's status tool: the recurring failure on this shared box
    is a forgotten process holding VRAM that a training run then cannot get.
    """
    raw = _run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits"])
    out = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            out.append({"pid": int(parts[0]), "name": parts[1],
                        "mem_mib": int(parts[2])})
        except ValueError:
            continue
    return out


def disk() -> dict:
    total, used, free = shutil.disk_usage(REPO_ROOT)
    gb = 1024 ** 3
    return {
        "path": str(REPO_ROOT),
        "total_gb": round(total / gb, 1),
        "used_gb": round(used / gb, 1),
        "free_gb": round(free / gb, 1),
        "free_pct": round(100 * free / total, 1) if total else None,
    }


# --- the snapshot ----------------------------------------------------------

def snapshot() -> dict:
    """Everything, as plain data. The watchdog reads this; so does --json."""
    return {
        "ts": _now(),
        "repo": str(REPO_ROOT),
        "jobs": live_jobs(),
        "runs": training_runs(),
        "datasets": datasets(),
        "gpus": gpus(),
        "gpu_holders": gpu_holders(),
        "disk": disk(),
    }


def classify_log(log_path: Path) -> tuple[str, str]:
    """(verdict, evidence) for a finished job's log: ok | failed | unknown.

    Kept here rather than in the watchdog so the same judgement is available to
    a human running `status` and to the alert that wakes someone at 3am.
    """
    if not log_path or not log_path.exists():
        return "unknown", "no log file"
    try:
        text = log_path.read_text(errors="replace")
    except OSError as e:
        return "unknown", f"unreadable: {e}"
    hits = [ln.strip() for ln in text.splitlines() if _FAILURE_PAT.search(ln)]
    if hits:
        return "failed", hits[-1][:200]
    tail = text.strip().splitlines()[-1:] or [""]
    return "ok", tail[0][:200]


# --- rendering -------------------------------------------------------------

def _age(hours: float | None) -> str:
    """Hours are unreadable past a couple of days: 3370.3h is not a duration
    anyone parses at a glance."""
    if hours is None:
        return "never"
    if hours < 48:
        return f"{hours:.1f}h"
    days = hours / 24
    return f"{days:.0f}d" if days < 90 else f"{days / 30.44:.0f}mo"


def _bar(pct: float | None, width: int = 16) -> str:
    if pct is None:
        return "?" * width
    filled = int(round(width * min(max(pct, 0), 100) / 100))
    return "#" * filled + "." * (width - filled)


def render(snap: dict) -> str:
    L: list[str] = []
    L.append(f"forager_ml — {snap['ts']}")
    L.append("=" * 62)

    L.append("")
    if snap["jobs"]:
        L.append("RUNNING JOBS")
        for j in snap["jobs"]:
            h, m = divmod(j["elapsed_s"] // 60, 60)
            detail = f" [{j['dataset']}]" if j["dataset"] else ""
            L.append(f"  {j['kind']:<10} {j['name']:<22} {h:>3}h{m:02d}m  pid {j['pid']}{detail}")
    else:
        L.append("RUNNING JOBS   (none)")

    L.append("")
    L.append("TRAINING RUNS")
    for r in snap["runs"]:
        flag = "ACTIVE" if r["active"] else "      "
        age = f"{_age(r['checkpoint_age_h'])} ago" if r["checkpoint_age_h"] is not None else "no checkpoint"
        metric = ""
        if r["best_metric"]:
            k, v = next(iter(r["best_metric"].items()))
            metric = f"  {k}={v}"
        L.append(f"  {flag} {r['family']}/{r['name']:<22} {age:>16}{metric}")

    L.append("")
    L.append("DATASETS")
    for d in snap["datasets"]:
        idle = f"{_age(d['idle_h'])} idle" if d["idle_h"] is not None else ""
        L.append(f"  {d['name']:<24} {d['images']:>7} images  {d['classes']:>3} classes  {idle}")

    L.append("")
    L.append("GPUS")
    for g in snap["gpus"]:
        L.append(f"  [{g['index']}] {g['name']:<24} {_bar(g['mem_pct'])} "
                 f"{g['mem_used_mib']:>6}/{g['mem_total_mib']} MiB  util {g['util_pct']}%")
    for p in snap["gpu_holders"]:
        L.append(f"      holding: pid {p['pid']:<8} {p['name'][:38]:<38} {p['mem_mib']} MiB")

    d = snap["disk"]
    L.append("")
    L.append(f"DISK  {_bar(100 - (d['free_pct'] or 0))} "
             f"{d['used_gb']}/{d['total_gb']} GB used, {d['free_gb']} GB free ({d['free_pct']}%)")
    L.append("")
    return "\n".join(L)


#: --wait-for groups. "training" covers both trainer entry points.
_KIND_GROUPS = {
    "download": ("download",),
    "specialist": ("specialist",),
    "router": ("router",),
    "training": ("specialist", "router"),
    "any": ("download", "specialist", "router"),
}


def wait_for(kind: str, poll: int = 60, timeout: int = 0) -> int:
    """Block until no job of ``kind`` is running. Returns 0, or 124 on timeout.

    Exists so the retrain scripts can wait on *what is running* rather than on
    PID literals. `retrain_v2.sh` previously did:

        for pid in 1238285 1238290 1238583; do
            if kill -0 "$pid"; then wait "$pid" || true; fi
        done

    which had two independent faults. The PIDs went stale at the first reboot,
    after which the guard fails open and training starts immediately. And even
    while they were alive, `wait` only works on children of the calling shell —
    those downloads were not — so it returned an error instantly that `|| true`
    swallowed. That wait never waited, on any run.
    """
    kinds = _KIND_GROUPS[kind]
    started = time.time()
    while True:
        running = [j for j in live_jobs() if j["kind"] in kinds]
        if not running:
            print(f"wait-for {kind}: clear")
            return 0
        if timeout and (time.time() - started) > timeout:
            print(f"wait-for {kind}: TIMED OUT after {timeout}s with "
                  f"{len(running)} still running", file=sys.stderr)
            return 124
        # Dataset counts move even when a downloader logs nothing, so this line
        # is the progress signal the old scripts printed against a fixed target.
        sizes = {d["name"]: d["images"] for d in datasets()}
        detail = ", ".join(f"{j['name']} {j['elapsed_s'] // 60}m" for j in running)
        print(f"wait-for {kind}: {len(running)} running ({detail}); "
              f"images={sum(sizes.values())}", flush=True)
        time.sleep(poll)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="forager_ml status readout")
    ap.add_argument("--json", action="store_true", help="emit the raw snapshot")
    ap.add_argument("--wait-for", choices=sorted(_KIND_GROUPS),
                    help="block until no job of this kind is running, then exit 0")
    ap.add_argument("--poll", type=int, default=60, help="seconds between checks (--wait-for)")
    ap.add_argument("--timeout", type=int, default=0, help="0 = wait forever (--wait-for)")
    args = ap.parse_args(argv)

    if args.wait_for:
        return wait_for(args.wait_for, poll=args.poll, timeout=args.timeout)

    snap = snapshot()
    print(json.dumps(snap, indent=2) if args.json else render(snap))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
