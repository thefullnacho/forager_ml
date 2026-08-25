"""Watchdog + status logic. No GPU, no DB, no real training run.

The behaviours pinned here are the ones the old monitor_jobs.sh got wrong:
identifying a job without a PID, staying quiet when nothing changed, and never
reporting an unverifiable ending as success.
"""

import json

import pytest

from ops import status, watchdog


# --- job identification ----------------------------------------------------

@pytest.mark.parametrize("cmd", [
    "python3 training/scripts/train_efficientnet_specialist.py --name berry_expert",
    "/usr/bin/python training/scripts/train_efficientnet_specialist.py --name x",
    "python3.13 /abs/path/train_efficientnet_specialist.py --epochs 60",
    "uv run training/scripts/train_efficientnet_specialist.py --name x",
])
def test_real_invocations_are_jobs(cmd):
    assert status._runs_script(cmd, "train_efficientnet_specialist.py")


@pytest.mark.parametrize("cmd", [
    # The false positive found by running the watchdog: a shell whose own argv
    # mentions the script counted as a live training run.
    "/usr/bin/zsh -c cp /usr/bin/sleep ./train_efficientnet_specialist.py",
    "grep -rn train_efficientnet_specialist.py .",
    "vim training/scripts/train_efficientnet_specialist.py",
    "python3 -m ops.watchdog",
    "",
])
def test_mentions_are_not_jobs(cmd):
    assert not status._runs_script(cmd, "train_efficientnet_specialist.py")


def test_run_name_comes_from_the_command_line():
    cmd = "python3 train_efficientnet_specialist.py --dataset berry_dataset_split --name berry_expert"
    assert status._arg_value(cmd, "--name") == "berry_expert"
    assert status._arg_value(cmd, "--dataset") == "berry_dataset_split"
    assert status._arg_value(cmd, "--missing") is None


# --- wait-for (what the retrain scripts call instead of waiting on PIDs) ----

def test_wait_for_returns_immediately_when_nothing_runs(monkeypatch):
    monkeypatch.setattr(status, "live_jobs", lambda: [])
    assert status.wait_for("download", poll=0) == 0


def test_wait_for_blocks_until_the_job_clears(monkeypatch):
    calls = {"n": 0}

    def fake_jobs():
        calls["n"] += 1
        return [{"kind": "download", "name": "berry_pull_inat", "elapsed_s": 60}] \
            if calls["n"] < 3 else []

    monkeypatch.setattr(status, "live_jobs", fake_jobs)
    monkeypatch.setattr(status, "datasets", lambda: [{"name": "d", "images": 1}])
    monkeypatch.setattr(status.time, "sleep", lambda s: None)
    assert status.wait_for("download", poll=0) == 0
    assert calls["n"] == 3


def test_wait_for_ignores_other_kinds(monkeypatch):
    """A training run must not hold up a script waiting on downloads."""
    monkeypatch.setattr(status, "live_jobs",
                        lambda: [{"kind": "specialist", "name": "berry_expert",
                                  "elapsed_s": 60}])
    assert status.wait_for("download", poll=0) == 0


def test_wait_for_times_out_rather_than_hanging_forever(monkeypatch):
    monkeypatch.setattr(status, "live_jobs",
                        lambda: [{"kind": "download", "name": "stuck", "elapsed_s": 9}])
    monkeypatch.setattr(status, "datasets", lambda: [])
    monkeypatch.setattr(status.time, "sleep", lambda s: None)
    assert status.wait_for("download", poll=0, timeout=-1) == 124


def test_training_group_covers_both_trainers():
    assert set(status._KIND_GROUPS["training"]) == {"specialist", "router"}


# --- log classification ----------------------------------------------------

def test_clean_log_is_ok(tmp_path):
    p = tmp_path / "berry_expert.log"
    p.write_text("epoch 59/60\nBest val accuracy: 0.981\nTraining complete\n")
    verdict, evidence = status.classify_log(p)
    assert verdict == "ok"
    assert "Training complete" in evidence


@pytest.mark.parametrize("line", [
    "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB",
    "Traceback (most recent call last):",
    "Killed",
    "Segmentation fault",
])
def test_failure_lines_are_caught(tmp_path, line):
    p = tmp_path / "x.log"
    p.write_text(f"epoch 1/60\n{line}\n")
    assert status.classify_log(p)[0] == "failed"


def test_missing_log_is_unknown_not_ok(tmp_path):
    """The distinction that matters: an unverifiable ending must never be
    reported as success."""
    assert status.classify_log(tmp_path / "nope.log")[0] == "unknown"


# --- edge-triggered transitions --------------------------------------------

@pytest.fixture
def wd(tmp_path, monkeypatch):
    """Watchdog pointed at a temp state dir, collecting sent notifications."""
    monkeypatch.setattr(watchdog, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(watchdog, "LOG_DIR", tmp_path / "logs")
    (tmp_path / "logs").mkdir()
    sent = []
    monkeypatch.setattr(watchdog, "notify",
                        lambda title, body, *a, **k: sent.append((title, body)))
    return sent, tmp_path


def _snap(jobs=(), free_pct=50.0, runs=()):
    return {
        "ts": "2026-08-15T00:00:00+00:00",
        "jobs": [{"name": n, "kind": "specialist", "pid": 1,
                  "elapsed_s": e, "dataset": None, "epochs": None} for n, e in jobs],
        "runs": list(runs),
        "disk": {"free_pct": free_pct, "free_gb": 100.0},
    }


def test_start_does_not_notify(wd):
    sent, _ = wd
    watchdog.check_jobs(_snap(jobs=[("berry_expert", 60)]), dry_run=False)
    assert sent == []


def test_finish_notifies_once_then_stays_quiet(wd):
    sent, tmp = wd
    (tmp / "logs/berry_expert.log").write_text("Training complete\n")
    watchdog.check_jobs(_snap(jobs=[("berry_expert", 3600)]), dry_run=False)
    watchdog.check_jobs(_snap(jobs=[]), dry_run=False)
    assert len(sent) == 1 and "finished" in sent[0][0]
    watchdog.check_jobs(_snap(jobs=[]), dry_run=False)
    assert len(sent) == 1, "edge-triggered: a steady state must stay silent"


def test_failed_job_is_reported_as_failed(wd):
    sent, tmp = wd
    (tmp / "logs/berry_expert.log").write_text("RuntimeError: CUDA out of memory\n")
    watchdog.check_jobs(_snap(jobs=[("berry_expert", 3600)]), dry_run=False)
    watchdog.check_jobs(_snap(jobs=[]), dry_run=False)
    assert "FAILED" in sent[0][0]


def test_job_without_a_log_is_unverified_not_finished(wd):
    sent, _ = wd
    watchdog.check_jobs(_snap(jobs=[("ghost", 3600)]), dry_run=False)
    watchdog.check_jobs(_snap(jobs=[]), dry_run=False)
    assert "unverified" in sent[0][0]
    assert "finished" not in sent[0][0]


def test_state_survives_between_passes(wd):
    """The whole point vs. monitor_jobs.sh: the watcher is not a long-lived
    process holding state in memory that dies with it."""
    _, tmp = wd
    watchdog.check_jobs(_snap(jobs=[("berry_expert", 60)]), dry_run=False)
    saved = json.loads((tmp / "state/jobs.json").read_text())
    assert "berry_expert" in saved["running"]


# --- disk ------------------------------------------------------------------

def test_disk_fires_once_on_crossing_and_recovers(wd, monkeypatch):
    sent, _ = wd
    monkeypatch.setattr(watchdog, "DISK_FLOOR_PCT", 10.0)
    watchdog.check_disk(_snap(free_pct=50.0), dry_run=False)
    assert sent == []
    watchdog.check_disk(_snap(free_pct=5.0), dry_run=False)
    watchdog.check_disk(_snap(free_pct=4.0), dry_run=False)
    assert len(sent) == 1 and "disk is low" in sent[0][0]
    watchdog.check_disk(_snap(free_pct=50.0), dry_run=False)
    assert len(sent) == 2 and "recovered" in sent[1][0]


# --- stall -----------------------------------------------------------------

def test_young_job_with_an_old_checkpoint_is_not_stalled(wd, monkeypatch):
    """A fresh run legitimately has an old checkpoint from its last training;
    calling that a stall would page on every single start."""
    sent, _ = wd
    monkeypatch.setattr(watchdog, "STALL_HOURS", 6.0)
    snap = _snap(jobs=[("berry_expert", 600)],
                 runs=[{"name": "berry_expert", "checkpoint_age_h": 900.0}])
    watchdog.check_stalled(snap, dry_run=False)
    assert sent == []


def test_long_job_with_a_stale_checkpoint_is_stalled(wd, monkeypatch):
    sent, _ = wd
    monkeypatch.setattr(watchdog, "STALL_HOURS", 6.0)
    snap = _snap(jobs=[("berry_expert", 40 * 3600)],
                 runs=[{"name": "berry_expert", "checkpoint_age_h": 20.0}])
    watchdog.check_stalled(snap, dry_run=False)
    assert len(sent) == 1 and "stalled" in sent[0][0]
    watchdog.check_stalled(snap, dry_run=False)
    assert len(sent) == 1, "edge-triggered"
