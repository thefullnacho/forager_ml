"""Constant-level drift check between forager_ml and Field Station convergence.py.

No live Field Station checkout needed -- writes fixture files to tmp_path so this
runs anywhere, same as the rest of ops/tests/.
"""

import pytest

from ops import convergence_drift as cd


def _write(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(body)
    return path


def test_matching_constants_do_not_drift(tmp_path):
    ml = _write(tmp_path, "ml.py", "LOW_CONFIDENCE_THRESHOLD = 0.50\n")
    fs = _write(tmp_path, "fs.py", "LOW_CONFIDENCE_THRESHOLD = 0.50\n")
    drifted, divergent = cd.check(ml, fs)
    assert drifted == []
    assert divergent == []


def test_same_name_different_value_is_drift(tmp_path):
    ml = _write(tmp_path, "ml.py", "LOW_CONFIDENCE_THRESHOLD = 0.50\n")
    fs = _write(tmp_path, "fs.py", "LOW_CONFIDENCE_THRESHOLD = 0.60\n")
    drifted, divergent = cd.check(ml, fs)
    assert drifted == [("LOW_CONFIDENCE_THRESHOLD", 0.50, 0.60)]
    assert divergent == []


def test_constant_on_only_one_side_is_divergence_not_drift(tmp_path):
    ml = _write(tmp_path, "ml.py", "LOW_CONFIDENCE_THRESHOLD = 0.50\n")
    fs = _write(tmp_path, "fs.py", "LOW_CONFIDENCE_THRESHOLD = 0.50\nDEADLY_VETO_FLOOR = 0.40\n")
    drifted, divergent = cd.check(ml, fs)
    assert drifted == []
    assert divergent == ["DEADLY_VETO_FLOOR"]


@pytest.mark.skipif(
    not cd.field_station_convergence().exists(),
    reason="needs a sibling forager-field-station checkout (local machine only, no remote)",
)
def test_real_files_current_known_state():
    """Pins today's actual divergence so a future silent fix has to touch this test."""
    drifted, divergent = cd.check()
    assert drifted == []
    assert set(divergent) == {"CONFIDENCE_THRESHOLD", "DEADLY_VETO_FLOOR", "EXPERT_CONFIDENCE_THRESHOLD"}


def test_non_upper_and_non_literal_assignments_are_ignored(tmp_path):
    ml = _write(tmp_path, "ml.py", "lower_case = 1\nCOMPUTED = some_func()\nTHRESH = 0.5\n")
    consts = cd.constants_from(ml)
    assert consts == {"THRESH": 0.5}


def test_leading_underscore_names_are_ignored(tmp_path):
    ml = _write(tmp_path, "ml.py", "_ABSTAIN_REASON = {'a': 'b'}\nTHRESH = 0.5\n")
    consts = cd.constants_from(ml)
    assert consts == {"THRESH": 0.5}


def test_missing_field_station_checkout_skips_cleanly(monkeypatch, tmp_path):
    monkeypatch.setenv("FORAGER_FS_PATH", str(tmp_path / "does-not-exist"))
    assert cd.main() == 0
