# ops — status readout and job watchdog

Two commands over one collector.

```bash
python -m ops.status            # human readout: jobs, runs, datasets, GPUs, disk
python -m ops.status --json     # the same snapshot, machine-readable
python -m ops.status --wait-for download   # block until no download is running
python -m ops.watchdog          # one pass; notifies only on a transition
python -m ops.watchdog --dry-run  # print what it would send, send nothing
```

`bash status.sh` still works — it is now a shim over `python -m ops.status`.

## The design: one collector, two consumers

`status.snapshot()` returns a plain dict. `status.render()` formats it for a
human; `watchdog` reads the same dict to decide whether to notify. Health is
never implemented twice, so the watchdog and the readout cannot disagree about
whether a job is running. Ported from hestia's `brain/tools/status.py`.

## What this replaced, and why

`monitor_jobs.sh` identified jobs by **hardcoded PID literals** pasted in during
one session:

```bash
JOBS=( "1271469:medicinals_expert:/tmp/train_medicinals.log" )
```

Three failure modes, all of which had already happened:

1. **The PIDs went stale at the first reboot.** After a restart they are dead,
   or worse, belong to something unrelated. `retrain_v2.sh` had the same fossil
   in its `kill -0` guard, where dead PIDs make it skip its wait and start
   training immediately — a guard that fails open and looks like it worked.
2. **The monitor was itself a `while true` process** that had to be remembered,
   could die silently, and left nothing watching.
3. **It exited once its listed jobs finished**, so it never saw the next run.

Nothing here persists a PID. Every pass derives the job set from live command
lines, so a run started long after the watchdog was installed is picked up with
no config change.

## `--wait-for`: the third consumer of the same detector

`retrain_v2.sh` and `retrain_router.sh` used to block on PID literals. Both now
call `python -m ops.status --wait-for download`, which polls the same
live-command-line detector, so they need no PIDs and pick up downloads started
after the script did.

```
--wait-for {download,specialist,router,training,any}
--poll SECONDS     # default 60
--timeout SECONDS  # 0 = forever; exits 124 on timeout
```

`retrain_v2.sh`'s old version was broken two independent ways:

```bash
for pid in 1238285 1238290 1238583; do
    if kill -0 "$pid"; then wait "$pid" || true; fi   # both faults, one line
done
```

1. Stale PIDs make `kill -0` fail, the guard falls through, and training starts
   instantly on a half-downloaded dataset **while looking like it worked**.
2. Even with live PIDs, `wait` only works on children of the calling shell.
   Those downloads were started separately, so `wait` errored immediately and
   `|| true` swallowed it. **That wait never waited, on any run.**
   `retrain_router.sh` knew this — its comment says so, and it polls `kill -0`
   instead — which is how the two scripts came to disagree.

## What the watchdog notifies on

Edge-triggered throughout: one notification per transition, silence otherwise.

| Check | Fires when |
|---|---|
| job finished | a job that was running is gone, and its log is clean |
| job **FAILED** | same, but the log has a traceback / CUDA OOM / kill / segfault |
| job ended, **unverified** | same, but no log was found — never reported as success |
| stalled | a job is alive but its checkpoint has not advanced in `FORAGER_STALL_HOURS` |
| disk low | free space crosses below `FORAGER_DISK_FLOOR_PCT` |

A start never notifies. An unverifiable ending is deliberately not called
success: believing a failed overnight run is the expensive mistake.

## Install

```bash
cp ops/systemd/forager-watchdog.{service,timer} ~/.config/systemd/user/
# edit the service: set FORAGER_NTFY_URL to a long random ntfy topic
systemctl --user daemon-reload
systemctl --user enable --now forager-watchdog.timer
systemctl --user start forager-watchdog.service   # one pass now
journalctl --user -u forager-watchdog -n 20
```

`FORAGER_NTFY_URL` has no default on purpose. A scrubbed placeholder deploys
silently and looks like it is working. ntfy topics are readable by anyone who
guesses the name, so use a long random one.

| Variable | Default | Meaning |
|---|---|---|
| `FORAGER_NTFY_URL` | *(required)* | ntfy topic URL |
| `FORAGER_LOG_DIR` | `/tmp` | where to find a finished job's log, matched by name |
| `FORAGER_DISK_FLOOR_PCT` | `10` | free-space floor |
| `FORAGER_STALL_HOURS` | `6` | no-checkpoint-progress window |
| `FORAGER_WATCHDOG_STATE_DIR` | `~/.local/state/forager-watchdog` | edge-trigger state |

**Point `FORAGER_LOG_DIR` somewhere persistent.** `/tmp` is cleared on reboot,
so a run that ends across a boot reports as "ended, unverified" rather than
done. The unit ships pointing at `forager_ml/logs/` (gitignored); have the
retrain scripts tee there.

## `convergence_drift`: catching the hand-port silently disagreeing

```bash
python -m ops.convergence_drift
```

`convergence.py` is hand-ported between this repo and the Field Station (see
STATUS.md, 2026-08-15) -- two different implementations by design, so a whole-file
diff is permanently red and tells you nothing. What actually matters is the named
safety constants (`DEADLY_VETO_FLOOR`, `EXPERT_CONFIDENCE_THRESHOLD`,
`LOW_CONFIDENCE_THRESHOLD`, ...). This walks both files' ASTs and separates two
cases:

- **DIVERGENCE** -- a constant exists on only one side. Reported, does not fail.
  Known and visible; today that's `DEADLY_VETO_FLOOR`, `EXPERT_CONFIDENCE_THRESHOLD`,
  `CONFIDENCE_THRESHOLD`.
- **DRIFT** -- the *same* constant holds two different values on each side. Fails
  the check. This is the dangerous case: it reads green in both repos and nothing
  says otherwise, the exact failure mode `forager-obs` was extracted to close for
  `toxic_as_edible`.

**Local only, not in GitHub Actions.** It reads a sibling `forager-field-station`
checkout by path (`../forager-field-station`, override with `FORAGER_FS_PATH`),
and that repo has no GitHub remote, so ops.yml/observability.yml can't check it
out. Install as a pre-commit hook instead:

```bash
cp ops/hooks/pre-commit .git/hooks/pre-commit
```

Skips cleanly (exit 0, one line) if no Field Station checkout is found at the
expected path -- never blocks a commit on a machine that doesn't have both repos.

## Not ported: running off-site

Hestia's watchdog runs on the dedi and probes the house, because the failure it
catches is "the house is dark" — which no process inside the house can report.
forager_ml's jobs are local, so a local watcher is the right shape here, and the
box itself is already covered by hestia's off-site probe.

## Tests

```bash
python -m pytest ops/tests/ -q      # 36 tests, no GPU, no DB, no training run
                                     # (one skips without a sibling Field Station checkout)
```

They pin the behaviours the old script got wrong: identifying a job without a
PID (including the false positive where a *shell* whose argv merely mentioned
the script counted as a live training run), staying silent when nothing changed,
and never reporting an unverifiable ending as success.
