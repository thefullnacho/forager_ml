# forager_ml observability harness

A batch harness that replays a validation set through the two-stage forager
pipeline and records a structured event per image, so drift and safety
regressions are **monitored and queryable**, not asserted once.

It exists because "0.0 toxic-as-edible false-accept rate" is a claim worth
verifying continuously, and because a rising out-of-distribution rate is an
early warning of data drift you want to see before accuracy visibly drops.

## What it records

- **`inference_runs`** — one row per inference: router decision, resolved
  species, safety tier, whether it abstained, whether deadly-vetoes-safe fired,
  and `toxic_as_edible` (predicted edible on a truly-DEADLY item). Currently
  single-shot (K=1); see the two-photo note below.
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

## What single-shot measures (and why the product uses two photos)

This harness currently runs **single-image (K=1)** inference. On the berry val
set that surfaces a real single-shot risk: ~1% of DEADLY specimens
(`canada_moonseed`, `pokeweed`) resolve to an edible tier. That is not a bug —
it is precisely the residual the **two-photo protocol exists to eliminate.**

The shipped product (forager-field-station) asks for a **second photo when the
first is not decisive**, then fuses the shots by geometric mean and requires
consensus (both photos independently pick the same class) before clearing the
0.60 gate — otherwise it abstains. `scripts/multiview_smoketest.py` in that repo
measures the payoff on the same val data:

| K (photos) | berry top-1 | DEADLY-shown-edible |
|---|---|---|
| 1 | 95.3% | 1.12% |
| 2 | 98.9% | **0.00%** |
| 3 | 99.9% | 0.00% |

So the "0.0 toxic-as-edible" claim is a **two-photo** claim, and it holds. The
single-shot number this harness reports is the useful complement: it quantifies
exactly how much safety the second photo is buying.

**Next:** add a multiview (K-photo, product-fusion + consensus) mode so
`v_safety_regression` measures the real two-photo operating point continuously,
with K=1 kept as the "what the second photo saves you from" baseline.

## Tests

`python -m pytest observability/tests/` — pure-logic tests for
deadly-vetoes-safe and the toxic-as-edible detector (no deps), plus DB
round-trip and view tests (auto-skipped if no Postgres). CI runs all of them
against a Postgres service — see `.github/workflows/observability.yml`.
