# ADR-001: Stage 13 tuning scope and locked internal holdout

## Status
Accepted

## Date
2026-09-13

## Context
Stage 12 closed with `catboost_native` as the clear Phase 4 champion — best
Tier 1, best Tier 3, and the tightest Tier 2 spread of any tree model in the
zoo (`docs/experiment_registry.md`, Stage 11.3). The blueprint (Phase 5,
Stage 13) calls for tuning "the strongest 1–2 candidates" with Optuna against
the literal competition score inside the Tier-2 geographic CV loop, using "a
modest trial budget" and setting aside one fold combination as a locked
internal holdout, touched only once per candidate near the end of tuning.
Neither the candidate count/budget nor which fold to lock was decided before
this session — both were open questions carried in the Stage 12→13 handoff.

Two constraints given for this decision: a wall-clock budget of two days for
Stage 13 before moving to ensembling and submission, and a willingness to
spend roughly double the compute on a second candidate (LightGBM, the
natural pick per Stage 11.3's numbers) for ensemble diversity — but only
*after* a first submission exists and the diversity analysis (Stage 14.1)
shows it would add a meaningful leaderboard gain, not speculatively now.

`tier2_splits()` (`src/climate_health/evaluation/cv.py`) yields Tier 2's
5×5 repeated geographic group K-folds deterministically: `n_splits=5`,
`n_repeats=5`, `random_state=42` by default, so every `(repeat, fold)` pair
is reproducible from those three numbers alone.

## Decision
**Candidate(s):** Tune `catboost_native` only in Stage 13. Do not tune
LightGBM in parallel. LightGBM (and the other Stage 11.3 diversity
candidates — HistGradientBoosting, ExtraTrees) stay as-is and re-enter the
picture at Stage 14.1's prediction-correlation analysis; a LightGBM tuning
pass is only added afterward, and only if that analysis shows real,
decorrelated value over tuned CatBoost, per the two-day budget and the
"only if it adds a meaningful gain" condition given for this decision.

**Trial budget:** A modest, fixed budget of 40 Optuna trials for
`catboost_native`, run after a 5-trial smoke test (Stage 13.3). 40 is small
enough to respect the blueprint's explicit warning against overfitting the
validation structure at only ~39 geographic groups, and large enough to
cover CatBoost's main hyperparameters (depth, learning rate, l2_leaf_reg,
iterations, bagging temperature) within the two-day wall-clock limit.

**Locked internal holdout:** `tier2_splits(..., n_splits=5, n_repeats=5,
random_state=42)`, fold `(repeat=0, fold=0)` — the first fold yielded by the
default-seeded generator. This combination is excluded from every per-trial
Optuna objective call and is evaluated exactly once, near the end of
tuning, per Stage 13.4. It was picked before looking at any per-fold score
for `catboost_native` or any other candidate, specifically to avoid
selecting a "convenient" fold after the fact — the blueprint's stated
purpose for locking a holdout only holds if the choice itself isn't
data-snooped.

## Alternatives Considered

### Tune CatBoost + LightGBM together now
- Pros: produces two tuned candidates immediately, ready for Stage 14
  ensembling without a second tuning pass later.
- Cons: roughly doubles Stage 13's compute/time inside a 2-day budget, and
  spends that budget before Stage 14.1 has shown whether LightGBM is even
  decorrelated enough from CatBoost to be worth tuning.
- Rejected: given now, in favor of deciding after the diversity analysis,
  per the explicit "only if it adds a meaningful gain" condition.

### Pick the locked holdout fold by inspecting Stage 11.3 per-fold scores and choosing a "typical difficulty" fold
- Pros: might feel like a more representative holdout.
- Cons: this requires looking at fold-level performance before locking the
  holdout, which is itself a form of holdout contamination — the fold
  selection would already be informed by CV signal.
- Rejected: the first deterministically-generated fold under the existing
  default seed is equally valid and carries no snooping risk.

### A larger trial budget (100+)
- Pros: more thorough hyperparameter search.
- Cons: directly conflicts with the blueprint's stated risk ("repeatedly
  optimizing against ~39 geographic groups risks overfitting the validation
  structure itself") and the 2-day wall-clock constraint.
- Rejected in favor of 40 trials plus the smoke test.

## Consequences
- `src/climate_health/models/tune.py` (Stage 13.2) hard-codes
  `LOCKED_HOLDOUT = {"random_state": 42, "repeat": 0, "fold": 0}` and
  excludes it from the per-trial objective loop.
- Only one tuned model (`catboost_tuned`) enters Stage 13.4 and Stage 14;
  the model-comparison table in `docs/experiment_registry.md` gains a
  `CatBoost (tuned)` row, not two.
- A LightGBM tuning pass remains possible post-Stage-14.1 and would be
  recorded as a new dated entry referencing this ADR, not a silent
  amendment to it.
