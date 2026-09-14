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
| majority_baseline | 0.6730 | 0.0003 | 0.6737 | 0.0078 | 0.6702 |
| logistic_regression | 0.8068 | 0.0181 | 0.6881 | 0.1683 | 0.7891 |
| lightgbm_default | 0.7975 | 0.0079 | 0.7892 | 0.0130 | 0.8040 |

**Takeaway:** LightGBM beat the majority-class baseline on Tier 1 (confirmed
by `tests/test_baselines.py::test_lightgbm_beats_majority_baseline_on_tier1`),
establishing real, learnable signal before any tuning work began. Note that
`logistic_regression`'s Tier 2 std (0.1683, min 0.334) already showed the
same geography-collapse pattern that later disqualified
`logistic_regression_v2` in Stage 11.3 — this was visible from Stage 10.3
onward, not a new Stage 11 finding.

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

## Stage 13.1 — Tuning scope and locked holdout decision

Full rationale: `docs/decisions/ADR-001-stage13-tuning-scope-and-holdout.md`.

**Candidate(s):** `catboost_native` only, tuned as `catboost_tuned`.
LightGBM tuning is deferred until after Stage 14.1's prediction-correlation
analysis shows it would add a meaningful, decorrelated gain — not tuned
speculatively alongside CatBoost.

**Trial budget:** 40 Optuna trials (after a 5-trial smoke test), chosen to
fit a 2-day wall-clock limit for Stage 13 while respecting the blueprint's
warning against overfitting the validation structure at ~39 geographic
groups.

**Locked internal holdout:** `tier2_splits(random_state=42, n_splits=5,
n_repeats=5)`, fold `(repeat=0, fold=0)` — the first deterministically
generated fold under the existing default seed, chosen before inspecting
any per-fold score to avoid holdout selection itself becoming a form of
data snooping. Excluded from every per-trial objective call; spent exactly
once, near the end of tuning, per Stage 13.4.

---

## Stage 13.3 — CatBoost tuning result

Source: `src/climate_health/models/tune.py::run_study`. Config: `configs/model_best.yaml`.
Study `catboost_tuned_tier2` (`sqlite:///optuna_studies.db`), 45 total trials
(5-trial smoke test + 40-trial full run, resumed into the same study, per
ADR-001's 2-day budget). Best trial: **#26**.

| Model | Tier 1 mean | Tier 2 mean (24 non-locked folds) | Tier 2 std | Tier 3 score | Competition score basis |
|---|---|---|---|---|---|
| catboost_native (Stage 11.3, default) | 0.8077 | 0.8023 | 0.0114 | 0.7953 | — |
| **catboost_tuned (Stage 13.3, trial 26)** | **0.8136** | **0.8156** | 0.0198 | 0.7963 | Tier 2 mean, 24/25 folds |

Params: `iterations=306, depth=6, learning_rate=0.01754, l2_leaf_reg=3.662,
bagging_temperature=0.1334`.

**Read the Tier 2 column carefully:** it is the mean over the 24 *non-locked*
folds only (Optuna's actual objective) — not a re-run over all 25, and not
directly comparable digit-for-digit to Stage 11.3's 25-fold `catboost_native`
number above, though both are close enough (0.8023 vs 0.8156) to read as a
real, modest improvement rather than noise. Tier 2 std rose slightly
(0.0114 → 0.0198) alongside the mean — worth watching, not yet a red flag on
its own at n=24.

**Locked holdout status: SPENT (2026-09-13).** See Stage 13.4 below.

---

## Stage 13.4 — Locked-holdout check (spent, one-time)

Full record: `docs/decisions/ADR-001-stage13-tuning-scope-and-holdout.md`,
"Outcome" section.

`catboost_tuned` (trial #26) evaluated once against `(repeat=0, fold=0)`
under `random_state=42` — the fold excluded from every Stage 13.2/13.3
trial:

| Metric | Value |
|---|---|
| Locked-holdout competition score | **0.8344** |
| Tier 2 mean (24 non-locked folds) | 0.8156 |
| Gap (tuning mean − holdout) | −0.0188 |

Gap is negative — the held-out fold scored *above* the tuning-time mean,
the opposite signature of holdout-adjacent overfitting, and within the
tuning distribution's own observed range (min 0.7840, max 0.8465).

**Decision: promote `catboost_tuned` (trial #26) as the Phase 5 champion
candidate.** Proceeds to Stage 14 ensembling as the primary model.
`(random_state=42, repeat=0, fold=0)` is now spent for this candidate and
will not be re-evaluated against it.

---

## Stage 14.1 — Prediction-correlation / diversity analysis (2026-09-13)

Source: `notebooks/04_modeling_experiments.ipynb`, Stage 14.1 cells. OOF
predictions for `catboost_tuned` (trial #26) plus the three remaining Stage
11.3 diversity candidates, on one shared 5-fold partition (`random_state=99`
— deliberately independent of ADR-001's locked holdout, never touches it).

| Model | OOF competition score | Corr. w/ catboost_tuned | Both-wrong w/ catboost_tuned |
|---|---|---|---|
| catboost_tuned | 0.8096 | 1.0000 | 1.0000 |
| extra_trees | 0.8030 | 0.9233 | 0.6473 |
| histgradientboosting | 0.7906 | 0.8835 | 0.6118 |
| logistic_regression_v2 | 0.7752 | 0.7690 | 0.6156 |

**Standalone score and ensembling value disagree.** `extra_trees` ranks 2nd
by score but is the *worst* ensembling partner for `catboost_tuned` — highest
correlation (0.9233) and highest both-wrong rate (0.6473); it's a tree
ensemble making largely the same calls. `logistic_regression_v2` ranks last
by score but is the clearest genuine diversity find — lowest correlation
with every tree model (0.65–0.77 vs. 0.88–0.93 tree-vs-tree) and a low
both-wrong rate against the champion (0.6156). `histgradientboosting` is a
solid secondary partner (0.6118 both-wrong, the best of the three) and forms
the single most decorrelated pair in the matrix when combined with
`logistic_regression_v2` (0.4787 both-wrong).

**LightGBM tuning decision (resolves the open question from ADR-001 §13.1):**
**deferred, not tuned.** Every tree-model pair here clusters tightly at
r=0.88–0.93; LightGBM is another boosted-tree model in the same family, and
`histgradientboosting` already occupies that role at zero extra tuning cost.
No evidence here justifies the ~2x compute spend. Revisit only if Stage
14.2's actual ensembling underperforms and specifically diagnoses
insufficient tree-model diversity as the cause.

**Carry-forward to Stage 14.2:** prioritize `catboost_tuned` +
`logistic_regression_v2` and/or `catboost_tuned` + `histgradientboosting`
blends. Include `extra_trees` in the full comparison for completeness, but
expect its OOF-optimized weight to land near zero — a valid, worth-recording
result on its own, not a failure of the analysis.

---

## Stage 14.2 — Ensembling strategies (2026-09-13)

Source: `src/climate_health/models/ensemble.py`, run via
`scripts/run_stage14_2_ensembling.py` over Stage 14.1's OOF predictions (same
5-fold partition, `random_state=99`).

| Strategy / model | OOF competition score |
|---|---|
| **weighted_average** | **0.8106** |
| oof_stack | 0.8098 |
| catboost_tuned (single model) | 0.8096 |
| extra_trees (single model) | 0.8030 |
| histgradientboosting (single model) | 0.7906 |
| logistic_regression_v2 (single model) | 0.7752 |
| rank_average | 0.7729 |

Optimized weights: `catboost_tuned=0.6701, logistic_regression_v2=0.1658,
extra_trees=0.1398, histgradientboosting=0.0243`.

**Best strategy (`weighted_average`) beats the single best model
(`catboost_tuned`) by only +0.0010** — inside the noise of a single 5-fold
OOF partition (Stage 13.3's Tier 2 std across folds was ~0.02, twenty times
this gap). Not yet trustworthy as a real improvement.

**`rank_average` underperformed every single candidate** (0.7729, below even
`logistic_regression_v2`'s 0.7752 alone) — equal-weighting three correlated
tree models let their shared wrong calls outvote `catboost_tuned`'s correct
ones. A useful negative result: unweighted rank averaging needs real
diversity across *all* inputs, not just one strong pair, to help.

**Surprise vs. Stage 14.1's forecast:** `extra_trees` received a non-trivial
optimized weight (0.1398), not the near-zero weight predicted from its high
correlation/both-wrong overlap with `catboost_tuned`. Noted, not yet
trusted — one partition isn't enough to overturn that forecast.

**Decision: do not promote a blend as champion yet.** The gain is too small
to distinguish from between-fold noise on a single partition. Before
promoting `weighted_average` (or any blend) over standalone `catboost_tuned`,
re-run the weighted-average blend through repeated Tier 2 CV (mirroring
Stage 13.3's 5×5 non-locked folds) to see if the gain holds up on average
across folds — a stretch goal for Stage 14.3, not yet completed. Per
ADR-001's consequence, promoting a blend would also require its own
freshly-locked holdout evaluation (the existing locked holdout is spent for
the standalone `catboost_tuned` config only).

---

## Stage 14.3 — Repeated-CV validation of the blend (stretch, 2026-09-14)

Source: `scripts/run_stage14_3_repeated_cv_validation.py`. Re-scored the
Stage 14.2 weighted-average blend (fixed weights: `catboost_tuned=0.6701,
logistic_regression_v2=0.1658, extra_trees=0.1398,
histgradientboosting=0.0243`) against the same 24 non-locked Tier 2 folds
`catboost_tuned` was tuned on in Stage 13.3, pairing each fold's blend score
against a freshly-computed `catboost_tuned` score on that same fold (Stage
13.3 only ever recorded the 24-fold aggregate, not per-fold scores).

| Model | 24-fold mean | 24-fold std |
|---|---|---|
| catboost_tuned (re-scored here) | 0.8156 | 0.0194 |
| weighted_average blend | 0.8145 | 0.0163 |

**Paired diff (blend − catboost), per fold: mean = −0.0011, std = 0.0068.**
Blend won 12 of 24 folds — a coin flip, not a consistent edge.

**This confirms Stage 14.2's +0.0010 gain was noise, not signal.** On a
single 5-fold OOF partition the blend edged out `catboost_tuned`; on the
full 24-fold repeated CV, the sign flips and the mean difference lands
essentially at zero. The blend tracks `catboost_tuned` closely fold-to-fold
(small diff std, expected given its 67% weight), but with no reliable
direction to that tracking.

**Decision: promote `catboost_tuned` (standalone, Optuna trial #26) as the
Phase 5 champion.** No ensembling strategy from Stage 14 is used in the
final model. This is a clean, evidence-based negative result — closes Stage
14 and clears the way for Stage 15 (calibration + submission scripting)
without carrying an unvalidated blend forward. Per ADR-001, the locked
`(repeat=0, fold=0)` holdout was not touched in this stretch step and
remains spent only for the standalone `catboost_tuned` config, exactly as
already recorded.

---

## Submission log

*(Empty — filled in starting at the project's first Zindi submission, per the
blueprint's S/F/M/E/C submission discipline. Each entry: date, model/config,
public leaderboard score, git commit hash.)*

| Date | Model / config | Public LB score | Commit |
|---|---|---|---|
| — | — | — | — |
