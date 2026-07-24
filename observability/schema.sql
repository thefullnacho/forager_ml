-- forager_ml observability schema.
--
-- One batch run replays a validation set through the two-stage pipeline and
-- writes one inference_runs row per image plus one expert_predictions row per
-- expert invoked. Deliberately holds NO location, NO image blob, NO user id:
-- there is nowhere to store a forage location, by design.

CREATE TABLE IF NOT EXISTS inference_runs (
    id               BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ts               TIMESTAMPTZ  NOT NULL DEFAULT now(),
    run_id           UUID         NOT NULL,              -- groups one batch/eval run
    model_version    TEXT         NOT NULL,              -- attributes metric shifts to a model
    source           TEXT         NOT NULL,              -- 'hailo' | 'onnx_cpu'
    image_ref        TEXT         NOT NULL,              -- val-set path/hash, NOT a location
    ground_truth     TEXT,                               -- known label (batch harness knows it)
    router_domain    TEXT         NOT NULL,
    router_confidence REAL        NOT NULL,
    router_abstained BOOLEAN      NOT NULL,
    species          TEXT         NOT NULL,              -- resolved class or 'unknown'
    safety_tier      TEXT         NOT NULL,              -- SAFE|CAUTION|DEADLY|UNKNOWN
    confidence       REAL         NOT NULL,
    low_confidence   BOOLEAN      NOT NULL,
    winning_expert   TEXT         NOT NULL,
    abstained        BOOLEAN      NOT NULL,              -- no surviving expert / router abstained
    deadly_veto      BOOLEAN      NOT NULL,              -- DEADLY overrode a more-confident non-deadly
    correct          BOOLEAN,                            -- species == ground_truth (batch only)
    toxic_as_edible  BOOLEAN      NOT NULL DEFAULT FALSE, -- predicted edible on a truly-DEADLY item
    total_ms         REAL
);

CREATE TABLE IF NOT EXISTS expert_predictions (
    id             BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id         BIGINT NOT NULL REFERENCES inference_runs(id) ON DELETE CASCADE,
    expert_name    TEXT   NOT NULL,
    top_class      TEXT   NOT NULL,
    top_confidence REAL   NOT NULL,
    energy_score   REAL   NOT NULL,   -- drift signal: -T * logsumexp(logits / T)
    ood_rejected   BOOLEAN NOT NULL   -- energy above the expert's calibrated threshold
);

CREATE INDEX IF NOT EXISTS ix_runs_model     ON inference_runs (model_version);
CREATE INDEX IF NOT EXISTS ix_runs_ts        ON inference_runs (ts);
CREATE INDEX IF NOT EXISTS ix_runs_run_id    ON inference_runs (run_id);
CREATE INDEX IF NOT EXISTS ix_expert_run     ON expert_predictions (run_id);
CREATE INDEX IF NOT EXISTS ix_expert_name    ON expert_predictions (expert_name);
