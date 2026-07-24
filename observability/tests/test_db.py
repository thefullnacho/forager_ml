"""DB round-trip + view tests. Skipped automatically if no Postgres is reachable."""

import uuid

import pytest

from observability.outcome import ExpertOutcome, InferenceOutcome

psycopg = pytest.importorskip("psycopg")


@pytest.fixture
def conn():
    from observability.db import connect, migrate
    try:
        c = connect()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no Postgres: {e}")
    migrate(c)
    # isolate: work in a transaction we roll back
    yield c
    c.rollback()
    c.close()


def _outcome(**over):
    base = dict(
        image_ref="berry_dataset_split/val/pokeweed_toxic/1.jpg",
        ground_truth="pokeweed_toxic", model_version="test-abc", source="onnx_cpu",
        router_domain="berry", router_confidence=0.9, router_abstained=False,
        species="pokeweed_toxic", safety_tier="DEADLY", confidence=0.8,
        low_confidence=False, winning_expert="berry_expert", abstained=False,
        deadly_veto=False, total_ms=12.3,
        experts=[ExpertOutcome("berry_expert", "pokeweed_toxic", 0.8, -3.1, -1.5, False)],
    )
    base.update(over)
    return InferenceOutcome(**base)


def test_write_roundtrip_and_expert_child(conn):
    from observability.writer import EventWriter
    w = EventWriter(conn, run_id=uuid.uuid4())
    pk = w.write(_outcome())
    run = conn.execute("SELECT species, safety_tier, correct, toxic_as_edible "
                       "FROM inference_runs WHERE id=%s", (pk,)).fetchone()
    assert run == ("pokeweed_toxic", "DEADLY", True, False)
    n_experts = conn.execute("SELECT count(*) FROM expert_predictions WHERE run_id=%s",
                             (pk,)).fetchone()[0]
    assert n_experts == 1


def test_toxic_as_edible_persisted(conn):
    from observability.writer import EventWriter
    w = EventWriter(conn)
    # a DEADLY truth resolved to a SAFE species → the row must carry the flag
    pk = w.write(_outcome(species="blueberry_wild", safety_tier="SAFE"))
    flag = conn.execute("SELECT toxic_as_edible FROM inference_runs WHERE id=%s",
                        (pk,)).fetchone()[0]
    assert flag is True


def test_safety_regression_view_counts(conn):
    from observability.writer import EventWriter
    rid = uuid.uuid4()
    w = EventWriter(conn, run_id=rid)
    w.write(_outcome(model_version="viewtest"))                                  # safe
    w.write(_outcome(model_version="viewtest", species="blueberry_wild",         # toxic
                     safety_tier="SAFE"))
    row = conn.execute("SELECT total_runs, toxic_as_edible_count "
                       "FROM v_safety_regression WHERE model_version='viewtest'").fetchone()
    assert row == (2, 1)
