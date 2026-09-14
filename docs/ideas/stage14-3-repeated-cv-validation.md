# Stage 14.3 — Repeated-CV validation of the weighted-average blend

## Problem Statement
How might we determine whether the weighted-average blend's +0.0010 edge over
standalone `catboost_tuned` (Stage 14.2) is a real, generalizable gain or just
noise from evaluating on a single 5-fold OOF partition?

## Recommended Direction
Re-score the exact blend found in Stage 14.2 — fixed weights
(`catboost_tuned=0.6701, logistic_regression_v2=0.1658, extra_trees=0.1398,
histgradientboosting=0.0243`), not re-optimized per fold — across the same 24
non-locked Tier 2 folds `catboost_tuned` was tuned and scored on in Stage
13.3 (`tune.py::non_locked_tier2_splits`). This keeps the comparison
apples-to-apples against the existing 0.8156 baseline and answers the
operationally real question ("does this specific blend hold up deployed
as-is?") rather than the more expensive question of whether blending in
general generalizes.

`catboost_tuned` is re-scored fold-by-fold alongside the blend, since Stage
13.3's tuning objective (`tune.py::make_objective`) only ever recorded the
24-fold mean/std/min/max — the per-fold scores list was discarded after
`summarize_tier2_scores`, so it isn't available to pair against without
re-running.

Decision rule: compute `(blend_score - catboost_score)` per fold, and look at
the mean and sign-consistency of that paired difference relative to the
~0.02 fold-to-fold std already on record from Stage 13.3 — not just whether
one aggregate mean exceeds the other.

## Key Assumptions to Validate
- [x] Fixed weights transfer across folds without collapsing — checked via a
      2-fold smoke test in a sandbox before shipping the full script; ran
      cleanly, no NaNs or degenerate blends.
- [x] `catboost_tuned`'s per-fold scores from Stage 13.3 are NOT recoverable
      from the study/logs (confirmed by reading `tune.py`) — re-scoring is
      required, which the script does.
- [ ] 24-fold compute cost is acceptable — roughly 1/45th of Stage 13's full
      tuning run (24 CatBoost fits vs. 45 trials × 24 folds), plus 3 fast
      sklearn fits per fold. Expected to run in well under Stage 13.3's
      wall-clock time; confirm on the real run.

## MVP Scope
**In:** `scripts/run_stage14_3_repeated_cv_validation.py` — reuses
`tune.py`'s locked-fold-excluding splitter and Branch A frame, and `zoo.py`'s
Branch B candidate definitions, regenerates all four candidates' predictions
per fold, applies the fixed blend weights, and reports per-fold scores plus
the paired mean/std/win-count.
**Out:** nested per-fold weight re-optimization, touching the locked
holdout, re-testing `oof_stack`/`rank_average` (Stage 14.2 already settled
those).

## Not Doing (and Why)
- Nested per-fold weight re-optimization — ~25x more compute, answers a
  different question (does blending-in-general generalize) than the one
  that matters for deployment (does this specific blend generalize).
- Touching the locked `(repeat=0, fold=0)` holdout — reserved for one real
  promotion decision, not a validation dry run.
- Re-testing `rank_average`/`oof_stack` — already ruled out in Stage 14.2.

## Open Questions
- None outstanding — both open questions from the initial one-pager were
  resolved by reading `tune.py` directly before writing the script.
