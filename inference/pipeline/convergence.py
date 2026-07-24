"""
convergence.py — Build a single ForagerResult from the two-stage router pipeline.

The domain router determines which expert(s) to run. This module takes the
router's domain prediction and the surviving expert predictions and resolves
them into a single ForagerResult with species metadata and safety info.

Safety-first resolution (deadly-vetoes-safe): when a domain is served by more
than one expert, a DEADLY verdict from any expert beats a non-deadly verdict
even if the non-deadly one is more confident. This is the fix for the
lily-of-the-valley failure mode — a confident "ramps" (SAFE) must never
out-vote a cautious "deadly". Max-confidence is used only to break ties within
a safety tier. DEADLY findings are always flagged prominently.
"""

from dataclasses import dataclass

import numpy as np

from .runner import RawPrediction
from .safety import SPECIES_METADATA, UNKNOWN_META, safety_of


# ── Tunable parameters ────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD    = 0.75   # used in log_predictions() threshold marker
LOW_CONFIDENCE_THRESHOLD = 0.50  # below this -> show "LOW CONFIDENCE" in display


# ── Safety metadata ───────────────────────────────────────────────────────────
# SPECIES_METADATA / UNKNOWN_META moved to safety.py (single source of truth,
# importable without the Hailo runtime). Re-imported below.




# ── Result type ──────────────────────────────────────────────────────────────

@dataclass
class ForagerResult:
    """Single identification result from the two-stage pipeline."""
    domain:          str    # "berry" | "mushroom" | "plant" | router raw output
    species:         str    # class key, e.g. "blackberry_common"
    scientific_name: str
    confidence:      float
    safety:          str    # SAFE | CAUTION | DEADLY | UNKNOWN
    lookalike:       str
    key_diff:        str
    low_confidence:  bool   # True when confidence < LOW_CONFIDENCE_THRESHOLD
    expert_model:    str    # which expert produced this result

    @property
    def is_deadly(self) -> bool:
        return self.safety == "DEADLY" and not self.low_confidence

    @property
    def is_unknown(self) -> bool:
        return self.species == "unknown"


# ── Debug logging ─────────────────────────────────────────────────────────────

def log_predictions(domain: str, predictions: list[RawPrediction]):
    """Print raw model output for debugging."""
    print("\n  -- Router + expert prediction --------------------------")
    print(f"  Domain: {domain}")
    if not predictions:
        print("  Expert: (none — router below threshold or OOD-rejected)")
    else:
        for pred in predictions:
            n_probs   = len(pred.probabilities)
            n_classes = len(pred.classes)
            print(f"  [{pred.model}]  output_size={n_probs}  classes={n_classes}", end="")
            if n_probs != n_classes:
                print(f"  MISMATCH", end="")
            print()
            top5_idx = np.argsort(pred.probabilities)[::-1][:5]
            for idx in top5_idx:
                label  = pred.classes[idx] if idx < n_classes else f"<unknown_idx_{idx}>"
                marker = "+" if pred.probabilities[idx] >= CONFIDENCE_THRESHOLD else "-"
                print(f"    {marker} {label:<45} {pred.probabilities[idx]:.1%}")
    print("  -------------------------------------------------------\n")


# ── Safety-aware resolution ────────────────────────────────────────────────────

def _safety_of(prediction: RawPrediction) -> str:
    """Safety tier of a prediction's top class (DEADLY when unknown — fail safe)."""
    return safety_of(prediction.top_class)


def resolve(predictions: list[RawPrediction]) -> RawPrediction:
    """
    Pick the winning expert prediction (deadly-vetoes-safe).

    Any DEADLY verdict beats a non-deadly one even when the non-deadly verdict
    is more confident — this is the lily-of-the-valley fix. Within a tier the
    most confident prediction wins. Assumes `predictions` is non-empty.
    """
    deadly = [p for p in predictions if _safety_of(p) == "DEADLY"]
    pool   = deadly if deadly else predictions
    return max(pool, key=lambda p: p.top_confidence)


# ── Main entry point ──────────────────────────────────────────────────────────

def build_result(domain: str, predictions: list[RawPrediction]) -> ForagerResult:
    """
    Resolve a domain + the surviving expert predictions into one ForagerResult.

    Resolution is deadly-vetoes-safe (see `resolve`). If `predictions` is empty
    (router abstained or every expert was OOD-rejected), returns an UNKNOWN
    result.
    """
    log_predictions(domain, predictions)

    if not predictions:
        return ForagerResult(
            domain=domain,
            species="unknown",
            scientific_name=UNKNOWN_META["scientific"],
            confidence=0.0,
            safety=UNKNOWN_META["safety"],
            lookalike=UNKNOWN_META["lookalike"],
            key_diff=UNKNOWN_META["key_diff"],
            low_confidence=True,
            expert_model="none",
        )

    prediction = resolve(predictions)
    meta = SPECIES_METADATA.get(prediction.top_class, UNKNOWN_META)
    low  = prediction.top_confidence < LOW_CONFIDENCE_THRESHOLD

    return ForagerResult(
        domain=domain,
        species=prediction.top_class,
        scientific_name=meta["scientific"],
        confidence=prediction.top_confidence,
        safety=meta["safety"],
        lookalike=meta["lookalike"],
        key_diff=meta["key_diff"],
        low_confidence=low,
        expert_model=prediction.model,
    )
