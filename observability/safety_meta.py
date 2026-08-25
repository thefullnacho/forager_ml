"""Bridge to the single source of truth for species safety tiers.

The canonical table lives in ``inference/pipeline/safety.py``. We load it by
file path rather than importing ``inference.pipeline`` as a package, because
that package's ``__init__`` chain pulls in ``runner`` and the Hailo runtime,
which is not installed off-device or in CI. safety.py itself is pure stdlib, so
loading just that file is clean and keeps one source of truth (no copied dict).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SAFETY_PATH = Path(__file__).resolve().parents[1] / "inference" / "pipeline" / "safety.py"

_spec = importlib.util.spec_from_file_location("forager_safety", _SAFETY_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

SPECIES_METADATA: dict[str, dict] = _mod.SPECIES_METADATA
UNKNOWN_META: dict = _mod.UNKNOWN_META
safety_of = _mod.safety_of
