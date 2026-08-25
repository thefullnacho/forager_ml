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

        The RULE lives in ``forager_obs.verdict`` and is shared with the Field
        Station, so the flagship 0.0 claim cannot quietly come to mean two
        different things in two dashboards. What stays local is the
        label -> tier lookup, because the metadata table is this repo's.
        """
        from forager_obs import is_toxic_as_edible

        from .safety_meta import safety_of

        truth_tier = None if self.ground_truth is None else safety_of(self.ground_truth)
        return is_toxic_as_edible(truth_tier, self.safety_tier, self.abstained)
