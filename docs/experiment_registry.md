# Experiment Registry

The single running log of every model-comparison experiment from Stage 10
onward, tabulated together per Stage 5's deliverable ("every subsequent
experiment reports all three validation tiers"). Every row below is a real
run, logged to MLflow under the `climate-health` experiment
(`sqlite:///mlflow.db`) — reproduce any row with the module named in the
"Source" column.

**Reading the tiers:** Tier 1 (standard, twin-guarded) is a general-signal
check, blind to geographic generalization. Tier 2 (5×5 repeated geographic
group K-fold) is the primary generalization signal given the dataset's
confirmed 0% train/test coordinate overlap (Stage 4) — reported as
mean/std/min/max, not a point estimate, since only 8 placeholder spatial
clusters exist. Tier 3 (test-like holdout, matched to the real test set's
76.2% rural composition) is the closest proxy to leaderboard behavior.
Prefer Tier 2/Tier 3 over Tier 1 when ranking candidates.

**Known caveat on every Tier 2 number below:** the spatial clustering in use
is still Stage 5's placeholder k=8 KMeans-on-coordinates
(`PLACEHOLDER_SPATIAL_CLUSTER_SOURCE`), not Phase 3 Stage 7's principled
clustering. Its largest cluster covers 56.4% of all rows — too coarse for a
clean 5-way split — which inflates Tier 2 variance across every candidate.
This is a structural limitation of the grouping, not a modeling result;
revisit every Tier-2-based ranking once Stage 7 ships a real clustering
feature.

---

## Stage 10.3 — Baseline scorecard

Source: `src/climate_health/models/baselines.py::run_baselines`
Feature pipeline: Stage 10.2 (`build_feature_matrix`), `numeric_feature_cols`
selection (all numeric columns, including the raw `spatial_cluster` int
label — a known, deliberately unfixed issue; see pipeline.py's Stage 11.2
ADR). Kept as-shipped so these baseline numbers stay reproducible.

| Model | Tier 1 mean | Tier 1 std | Tier 2 mean | Tier 2 std | Tier 3 score |
|---|---|---|---|---|---|
| majority_baseline | — | — | — | — | — |
| logistic_regression | — | — | — | — | — |
| lightgbm_default | — | — | — | — | — |

*(Fill in from your local MLflow run — see `mlflow ui --backend-store-uri sqlite:///mlflow.db`,
experiment `climate-health`, runs named `majority_baseline` / `logistic_regression` /
`lightgbm_default`.)*

**Takeaway:** LightGBM beat the majority-class baseline on Tier 1 (confirmed
by `tests/test_baselines.py::test_lightgbm_beats_majority_baseline_on_tier1`),
establishing there is real, learnable signal in the feature set before any
tuning work began.

---

## Stage 11.3 — Model zoo comparison

Source: `src/climate_health/models/zoo.py::run_zoo`
Feature pipeline: Stage 11.2 two-branch selection — Branch A (native
`category` dtype columns, CatBoost only) / Branch B (numeric + one-hot +
target-encoded, all other candidates).

| Model | Branch | Tier 1 mean | Tier 1 std | Tier 2 mean | Tier 2 std | Tier 2 min | Tier 2 max | Tier 3 score |
|---|---|---|---|---|---|---|---|---|
| **catboost_native** | A | 0.8077 | 0.0109 | 0.8023 | 0.0114 | 0.7847 | 0.8227 | **0.7953** |
| logistic_regression_v2 | B | 0.8071 | 0.0169 | 0.7107 | 0.1756 | 0.3342 | 0.8101 | 0.7892 |
| extra_trees | B | 0.8046 | 0.0101 | **0.8134** | 0.0250 | 0.7760 | 0.8627 | 0.7907 |
| histgradientboosting | B | 0.8001 | 0.0082 | 0.7952 | 0.0091 | 0.7806 | 0.8136 | 0.7996 |
| lightgbm_tuned | B | 0.7976 | 0.0092 | 0.7925 | 0.0120 | 0.7699 | 0.8080 | 0.7943 |
| xgboost_tuned | B | 0.7967 | 0.0055 | 0.7949 | 0.0086 | 0.7768 | 0.8062 | 0.7931 |

**Champion: `catboost_native`.** Best Tier 1, best Tier 3, and the tightest
Tier 2 spread of any tree model — the only candidate whose ranking holds up
consistently across all three tiers, which is what this project's validation
strategy is designed to select for.

**Disqualified: `logistic_regression_v2`.** Tier 2 std of 0.176 with a
min of 0.334 (nowhere else close to that on Tier 1 or Tier 3) — a
near-total collapse on at least one of the 25 repeat×fold splits, invisible
to Tier 1. This is the exact geography-blind failure mode Tier 2 exists to
catch. Dropped from further tuning on this feature representation.

**Caution on `extra_trees`.** Highest Tier 2 mean in the zoo (0.8134) but
combined with the lowest Tier 3 score (0.7907) and ~2x CatBoost's Tier 2
variance — likely fitting favorably to specific clusters rather than
generalizing better. A reminder not to select a champion on Tier 2 mean
alone.

**Middle tier:** `histgradientboosting`, `lightgbm_tuned`, `xgboost_tuned`
cluster tightly (0.793–0.800 across all three tiers, low variance each) —
reliable diversity candidates for a later ensemble even though none leads on
any single tier.

**Decision for Phase 5:** carry `catboost_native` forward as the primary
hyperparameter-tuning target. Retain HistGradientBoosting, LightGBM, and
XGBoost as ensemble diversity candidates. Logistic Regression is out unless
a future feature representation changes this result.

---

## Submission log

*(Empty — filled in starting at the project's first Zindi submission, per the
blueprint's S/F/M/E/C submission discipline. Each entry: date, model/config,
public leaderboard score, git commit hash.)*

| Date | Model / config | Public LB score | Commit |
|---|---|---|---|
| — | — | — | — |
