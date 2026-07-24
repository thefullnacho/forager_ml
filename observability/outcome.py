"""Structured inference outcomes — the event shape the writer persists.

Deliberately decoupled from the inference *source*: a Hailo on-device run and
an ONNX-on-CPU run both produce these same objects, so the schema, writer, and
analytics don't care where the numbers came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExpertOutcome:
    """One expert's verdict on one image, including the energy/OOD signal that
    the live pipeline currently prints and discards."""
    expert_name: str
    top_class: str
    top_confidence: float
    energy_score: float
    energy_threshold: float
    ood_rejected: bool


@dataclass
class InferenceOutcome:
    """Everything one image produced, across both stages — the row-level event."""
    image_ref: str
    ground_truth: str | None
    model_version: str
    source: str  # 'hailo' | 'onnx_cpu'

    router_domain: str
    router_confidence: float
    router_abstained: bool

    species: str
    safety_tier: str
    confidence: float
    low_confidence: bool
    winning_expert: str
    abstained: bool
    deadly_veto: bool

    total_ms: float | None = None
    experts: list[ExpertOutcome] = field(default_factory=list)

    @property
    def correct(self) -> bool | None:
        if self.ground_truth is None:
            return None
        return self.species == self.ground_truth

    @property
    def toxic_as_edible(self) -> bool:
        """The only error that truly matters: the pipeline called something
        edible/safe when the ground truth is a DEADLY class.

        Ground-truth toxicity is read from the safety metadata (single source of
        truth), not guessed from the label string.
        """
        from .safety_meta import safety_of

        if self.ground_truth is None:
            return False
        truth_is_deadly = safety_of(self.ground_truth) == "DEADLY"
        predicted_edible = self.safety_tier in ("SAFE", "CAUTION")
        return truth_is_deadly and predicted_edible and not self.abstained
