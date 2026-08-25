-- Analytical views — the queries that make this observability, not logging.
-- Each is the SQL behind one monitoring question.

-- 1. Safety regression: the flagship 0.0 toxic-as-edible claim, per model,
--    continuously verified instead of asserted once. CI asserts this is empty.
CREATE OR REPLACE VIEW v_safety_regression AS
SELECT model_version,
       count(*)                                   AS total_runs,
       count(*) FILTER (WHERE toxic_as_edible)    AS toxic_as_edible_count,
       round((count(*) FILTER (WHERE toxic_as_edible))::numeric
             / nullif(count(*), 0), 6)            AS toxic_as_edible_rate
FROM inference_runs
GROUP BY model_version
ORDER BY model_version;

-- 2. Energy drift: per-expert p50/p95 energy over time. Rising p95 = more
--    out-of-distribution inputs = data drift, caught before accuracy visibly drops.
CREATE OR REPLACE VIEW v_energy_drift AS
SELECT e.expert_name,
       date_trunc('day', r.ts)                                              AS day,
       count(*)                                                             AS n,
       round(percentile_cont(0.50) WITHIN GROUP (ORDER BY e.energy_score)::numeric, 4) AS p50_energy,
       round(percentile_cont(0.95) WITHIN GROUP (ORDER BY e.energy_score)::numeric, 4) AS p95_energy,
       round(avg(e.ood_rejected::int)::numeric, 4)                          AS ood_reject_rate
FROM expert_predictions e
JOIN inference_runs r ON r.id = e.run_id
GROUP BY e.expert_name, date_trunc('day', r.ts)
ORDER BY e.expert_name, day;

-- 3. Abstention rate by domain and model — is the system getting more/less cautious?
CREATE OR REPLACE VIEW v_abstention AS
SELECT model_version,
       router_domain,
       count(*)                                       AS total,
       count(*) FILTER (WHERE abstained)              AS abstained,
       round(avg(abstained::int)::numeric, 4)         AS abstention_rate
FROM inference_runs
GROUP BY model_version, router_domain
ORDER BY model_version, router_domain;

-- 4. Deadly-veto frequency: how often the safety layer actually changes the answer.
CREATE OR REPLACE VIEW v_deadly_veto AS
SELECT model_version,
       count(*)                                    AS total,
       count(*) FILTER (WHERE deadly_veto)         AS deadly_vetoes,
       round(avg(deadly_veto::int)::numeric, 4)    AS veto_rate
FROM inference_runs
GROUP BY model_version
ORDER BY model_version;

-- 5. Confidence calibration: within each confidence decile, how often was it right?
--    A well-calibrated model tracks the diagonal (0.8 bucket ~ 80% correct).
CREATE OR REPLACE VIEW v_calibration AS
SELECT model_version,
       width_bucket(confidence, 0, 1, 10) AS confidence_decile,
       count(*)                                        AS n,
       round(avg(correct::int)::numeric, 4)           AS accuracy
FROM inference_runs
WHERE correct IS NOT NULL AND NOT abstained
GROUP BY model_version, width_bucket(confidence, 0, 1, 10)
ORDER BY model_version, confidence_decile;
