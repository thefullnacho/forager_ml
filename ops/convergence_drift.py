"""Drift check for convergence.py's safety constants against the Field Station's copy.

convergence.py is hand-ported, not shared (STATUS.md, 2026-08-15: "still hand-ported
between this repo and the Field Station... needs a different mechanism"). The two
copies are legitimately different implementations -- RawPrediction vs. a plain dict,
different call signatures -- so a whole-file diff is useless, it is red forever.

What actually matters is the named safety thresholds. A constant that exists on both
sides with two different values is silent drift: it reads green in both repos and
nothing says otherwise, the same failure mode STATUS.md already names for
`toxic_as_edible`. A constant that exists on only one side is a visible, known
divergence -- worth surfacing, not worth failing a commit over.
"""

import ast
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FORAGER_ML_CONVERGENCE = REPO_ROOT / "inference" / "pipeline" / "convergence.py"


def field_station_convergence() -> Path:
    fs_root = os.environ.get("FORAGER_FS_PATH", str(REPO_ROOT.parent / "forager-field-station"))
    return Path(fs_root) / "pipeline" / "convergence.py"


def constants_from(path: Path) -> dict:
    """Module-level ALL_CAPS constant assignments: name -> literal value.

    Skips leading-underscore names (_ABSTAIN_REASON etc.) -- private lookup
    tables, not the tunable safety thresholds this check is for.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.isupper() and not target.id.startswith("_"):
                try:
                    out[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    return out


def check(ml_path: Path = None, fs_path: Path = None) -> tuple[list, list]:
    """Returns (drifted, divergent). drifted = shared name, different value."""
    ml_path = ml_path or FORAGER_ML_CONVERGENCE
    fs_path = fs_path or field_station_convergence()

    ml_consts = constants_from(ml_path)
    fs_consts = constants_from(fs_path)

    shared = ml_consts.keys() & fs_consts.keys()
    drifted = sorted(
        (name, ml_consts[name], fs_consts[name])
        for name in shared
        if ml_consts[name] != fs_consts[name]
    )
    divergent = sorted(ml_consts.keys() ^ fs_consts.keys())
    return drifted, divergent


def main() -> int:
    fs_file = field_station_convergence()
    if not fs_file.exists():
        print(f"convergence_drift: no Field Station checkout at {fs_file.parent} "
              f"(set FORAGER_FS_PATH) -- skipping, nothing checked")
        return 0

    drifted, divergent = check()

    if divergent:
        print("convergence_drift: constants present on only one side (known, not failing):")
        for name in divergent:
            print(f"  DIVERGENCE: {name}")

    if drifted:
        print("convergence_drift: SAME constant, DIFFERENT value on each side:")
        for name, ml_val, fs_val in drifted:
            print(f"  DRIFT: {name} = {ml_val!r} in forager_ml, {fs_val!r} in Field Station")
        print("\nA safety threshold split like this reads green in both repos until someone")
        print("diffs them by hand. Reconcile the value or confirm the split is intentional.")
        return 1

    print("convergence_drift: no shared constant has drifted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
