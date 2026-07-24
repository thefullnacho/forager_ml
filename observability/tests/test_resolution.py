"""Pure-logic tests for the safety resolution — no DB, no models needed.

These lock in the two behaviors the whole build exists to protect:
deadly-vetoes-safe resolution, and the toxic-as-edible detector.
"""

from observability.onnx_pipeline import OnnxPipeline
from observability.outcome import ExpertOutcome, InferenceOutcome


def _expert(name, cls, conf):
    return ExpertOutcome(name, cls, conf, energy_score=-3.0,
                         energy_threshold=-1.5, ood_rejected=False)


def test_deadly_vetoes_more_confident_safe():
    # A confident SAFE (ramps @0.9) must lose to a less-confident DEADLY
    # (amanita @0.4) — the lily-of-the-valley failure mode.
    survivors = [
        _expert("highvalue_expert", "ramps_wild_leek", 0.90),
        _expert("psychedelics_expert", "amanita_phalloides_deadly", 0.40),
    ]
    winner, veto = OnnxPipeline._resolve(survivors)
    assert winner.top_class == "amanita_phalloides_deadly"
    assert veto is True


def test_no_veto_when_most_confident_is_already_deadly():
    survivors = [
        _expert("psychedelics_expert", "amanita_phalloides_deadly", 0.80),
        _expert("highvalue_expert", "chanterelles_edible", 0.30),
    ]
    winner, veto = OnnxPipeline._resolve(survivors)
    assert winner.top_class == "amanita_phalloides_deadly"
    assert veto is False  # max-confidence already picked it; safety didn't change the answer


def test_within_tier_most_confident_wins():
    survivors = [
        _expert("highvalue_expert", "chanterelles_edible", 0.55),
        _expert("medicinals_expert", "mullein", 0.72),
    ]
    winner, veto = OnnxPipeline._resolve(survivors)
    assert winner.top_class == "mullein"
    assert veto is False


def _outcome(ground_truth, species, tier, abstained=False):
    return InferenceOutcome(
        image_ref="x.jpg", ground_truth=ground_truth, model_version="t", source="onnx_cpu",
        router_domain="berry", router_confidence=0.9, router_abstained=False,
        species=species, safety_tier=tier, confidence=0.6, low_confidence=False,
        winning_expert="berry_expert", abstained=abstained, deadly_veto=False,
    )


def test_toxic_as_edible_true_when_deadly_called_safe():
    o = _outcome("canada_moonseed_deadly", "blueberry_highbush", "SAFE")
    assert o.toxic_as_edible is True


def test_toxic_as_edible_true_when_deadly_called_caution():
    # CAUTION still means "you might eat this" — counts as toxic-as-edible.
    o = _outcome("pokeweed_toxic", "elderberry_american", "CAUTION")
    assert o.toxic_as_edible is True


def test_toxic_as_edible_false_when_abstained():
    # Refusing is the safe outcome, never a toxic-as-edible error.
    o = _outcome("canada_moonseed_deadly", "unknown", "UNKNOWN", abstained=True)
    assert o.toxic_as_edible is False


def test_toxic_as_edible_false_when_truth_is_safe():
    o = _outcome("blackberry_common", "blueberry_wild", "SAFE")
    assert o.toxic_as_edible is False


def test_toxic_as_edible_false_when_deadly_stays_deadly():
    o = _outcome("canada_moonseed_deadly", "pokeweed_toxic", "DEADLY")
    assert o.toxic_as_edible is False
