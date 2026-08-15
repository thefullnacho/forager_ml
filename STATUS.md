# STATUS

Running log of where forager_ml actually is. Newest entry at the top. Append, do not rewrite.

Cross-repo facts (what this repo shares with forager-field-station, hestia and the site) live
in the Forager wiki at `~/Documents/Forager/forager-wiki/`, not here. This file is for what is
true inside this repo.

---

## 2026-08-15 — shared observability package, and the end of the PID scripts

Two pieces of ops debt closed, both found by a primitives audit across the constellation
rather than by anything breaking.

**`forager-obs` extracted.** The definition of `toxic_as_edible` existed twice: here as a
property on `InferenceOutcome`, in forager-field-station inline in its `SessionWriter`. The
flagship 0.0 claim rested on both. A safety metric defined twice can drift once and still read
green in both dashboards, and `convergence.py` had already been hand-ported the same way.
The rule now lives in `github.com/thefullnacho/forager-obs` (public, `main`), installed as an
editable sibling by `observability/requirements.txt`, along with connect/migrate, the
batch-writer contract, and the ImageFolder val-set walk.

**Deliberately not merged: the schemas.** This repo is image-grained (`inference_runs` +
`expert_predictions`). The Field Station is session-grained (`sessions`, `n_photos` a
first-class dimension). Those answer different questions. Recorded in the wiki as intentional
so a later pass does not "fix" it.

**Two latent bugs, both found by running it rather than reading it.** Both compose files bound
host port 5433 *and* derived the same Compose project name from their parent directory
(`observability` in both repos), so bringing up the second silently recreated the first repo's
container. Now 5433 / `forager-ml-obs` here and 5434 / `forager-fs-obs` there, verified running
side by side for the first time.

**`ops/` replaces `status.sh` + `monitor_jobs.sh`.** `snapshot()` + `render()`, the pattern
from hestia's `brain/tools/status.py`: one collector, three consumers now — the readout, the
watchdog, and the retrain scripts' `--wait-for`. `monitor_jobs.sh` is removed. It identified
jobs by hardcoded PID literals pasted in during one session, was itself a `while true` process
that could die unnoticed, and exited once its listed jobs finished so it never saw the next
run. Nothing in `ops/` persists a PID; the job set is derived each pass from live command
lines.

**`retrain_v2.sh` had been broken since the day it was written.** Beyond the stale PIDs (dead
after any reboot, after which `kill -0` fails, the guard falls open, and training starts on a
half-downloaded dataset while looking like it worked), it called `wait "$pid"` on processes
that were never children of its shell. `wait` errors instantly on a non-child and `|| true`
swallowed it. That wait never waited, on any run. `retrain_router.sh` already knew this — its
own comment says so and it polled `kill -0` instead — so the two scripts had silently
disagreed the whole time. Both now call `ops.status --wait-for download`.

**Both scripts now tee to `logs/<name>.log`**, which is where the watchdog reads to decide
finished-vs-failed. Without it every completed retrain would have reported "ended,
unverified" — which the watchdog does deliberately, because believing a failed overnight run
is the expensive mistake.

**The false positive worth remembering.** The first job collector matched the script name as a
substring anywhere in a command line, so a *shell* whose argv merely mentioned
`train_efficientnet_specialist.py` counted as a live training run — as would any grep or open
editor. It now requires an interpreter in `argv[0]` plus the script as a whole argv token.
Pinned by test.

**Verified:** 36 forager-obs tests, 29 `ops/` tests, 11 observability tests against a live
Postgres, plus an end-to-end write through the shared rule read back out of
`v_safety_regression`, and a full retrain-to-watchdog loop (run started, seen live, log
written, reported finished).

### In flight

- **`observability` branch is 6 commits ahead of `main` and unpushed.** Nothing merged yet.
- The new `ops` workflow runs green anywhere (stdlib only). The `observability` workflow now
  checks out `thefullnacho/forager-obs` beside this repo; that path is verified but has not yet
  run on a real push.

### Next concrete action

Push the `observability` branch and confirm both workflows go green on GitHub, then decide
whether it merges to `main` or `dev`.

### Not done, on purpose

- `convergence.py` (single-expert routing, `DEADLY_VETO_FLOOR`, `EXPERT_CONFIDENCE_THRESHOLD`)
  is still hand-ported between this repo and the Field Station. It sits in the Space's *runtime*
  path, so sharing it the same way would make the Space depend on a pip install to boot. Needs
  a different mechanism.
- `[non-production]` Pick an ntfy topic and enable the watchdog timer
  (`ops/systemd/forager-watchdog.{service,timer}`). `FORAGER_NTFY_URL` has no default on
  purpose. Until this is done, nothing tells you a retrain died.
