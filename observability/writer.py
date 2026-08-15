"""Persists InferenceOutcome objects to Postgres. Source-agnostic: takes the
same outcome whether it came from Hailo or ONNX-on-CPU."""

from __future__ import annotations

from forager_obs import BaseWriter

from .outcome import InferenceOutcome


class EventWriter(BaseWriter):
    """Writes one inference_runs row + N expert_predictions rows per outcome,
    all tagged with a shared run_id for the batch.

    The connection, the run_id, and commit/rollback come from BaseWriter; the
    SQL stays here because this repo's grain (per image, plus a child row per
    expert) is not the Field Station's (per multi-photo session).
    """

    def write(self, o: InferenceOutcome) -> int:
        row = self.conn.execute(
            """
            INSERT INTO inference_runs (
                run_id, model_version, source, image_ref, ground_truth,
                router_domain, router_confidence, router_abstained,
                species, safety_tier, confidence, low_confidence,
                winning_expert, abstained, deadly_veto, correct,
                toxic_as_edible, total_ms
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s
            ) RETURNING id
            """,
            (
                self.run_id, o.model_version, o.source, o.image_ref, o.ground_truth,
                o.router_domain, o.router_confidence, o.router_abstained,
                o.species, o.safety_tier, o.confidence, o.low_confidence,
                o.winning_expert, o.abstained, o.deadly_veto, o.correct,
                o.toxic_as_edible, o.total_ms,
            ),
        ).fetchone()
        run_pk = row[0]

        for e in o.experts:
            self.conn.execute(
                """
                INSERT INTO expert_predictions (
                    run_id, expert_name, top_class, top_confidence,
                    energy_score, ood_rejected
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (run_pk, e.expert_name, e.top_class, e.top_confidence,
                 e.energy_score, e.ood_rejected),
            )
        return run_pk
