# forager_ml observability harness

A batch harness that replays a validation set through the two-stage forager
pipeline and records a structured event per image, so drift and safety
regressions are **monitored and queryable**, not asserted once.

It exists because "0.0 toxic-as-edible false-accept rate" is a claim worth
verifying continuously, and because a rising out-of-distribution rate is an
early warning of data drift you want to see before accuracy visibly drops.

## What it records

- **`inference_runs`** — one row per image: router decision, resolved species,
  safety tier, whether it abstained, whether deadly-vetoes-safe fired, and
  `toxic_as_edible` (predicted edible on a truly-DEADLY item).
- **`expert_predictions`** — one row per expert invoked: top class, confidence,
  and the **energy score** (the drift signal the live pipeline currently prints
  and discards).

No forage location, no image blob, no user id — the schema has nowhere to store
them. Runs on Alex's own validation set only.

## Design: one event shape, two inference sources

The event schema and analytics don't care where predictions come from:

- **ONNX-on-CPU** (`onnx_pipeline.py`) — runs anywhere, including CI. Uses the
  exported `*_logits.onnx` models, so energy scores are computed on the same
  raw logits the on-device path calibrated against.
- **On-device Hailo** (future) — same `InferenceOutcome`, real HEF inference.

The safety metadata (which species is DEADLY) is imported from the single source
of truth, `inference/pipeline/safety.py`, never copied.

## Run it

```bash
docker compose -f observability/docker-compose.yml up -d
python -m observability.batch_runner \
    --val-dir berry_dataset_split/val \
    --model-version "dev-$(git rev-parse --short HEAD)" \
    --max-per-class 8
```

Then query the analytical views (`views.sql`):

- `v_safety_regression` — toxic-as-edible count/rate per model_version (should be 0)
- `v_energy_drift` — per-expert p50/p95 energy over time; rising p95 = drift
- `v_abstention` — refusal rate by domain and model
- `v_deadly_veto` — how often the safety layer changed the answer
- `v_calibration` — confidence decile vs. actual accuracy

## Known finding (first run, ONNX-CPU, berry val set)

The harness immediately surfaced a real gap: the **berry domain has a single
expert**, so deadly-vetoes-safe (which needs two experts) can't protect it, and
the berry expert's energy OOD gate has `auroc 0.2477` and never fires. On the
ONNX-float path, that left a handful of `canada_moonseed`/`pokeweed` images
resolving to edible. Whether the calibrated int8 HEF closes this on-device is
the exact question this harness now makes answerable — run the Hailo source over
the same set and compare `v_safety_regression`.

## Tests

`python -m pytest observability/tests/` — pure-logic tests for
deadly-vetoes-safe and the toxic-as-edible detector (no deps), plus DB
round-trip and view tests (auto-skipped if no Postgres). CI runs all of them
against a Postgres service — see `.github/workflows/observability.yml`.
