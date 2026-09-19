# Implementation Plan: Stage 8.5 — Post-Champion Feature Engineering (Amended Scope)

## Overview
Two of the four originally-scoped feature directions turned out to already be shipped in the champion pipeline (see `docs/ideas/stage8-5-post-champion-feature-engineering.md`'s 2026-09-18 correction). What's actually left is small: wire one already-built, already-tested, but never-called function into the pipeline; build one genuinely new shrinkage-smoothed feature justified by a completed data audit; run one cheap interaction check; then a single retrain-and-resubmit cycle. Every task follows this project's existing discipline — no feature is kept on Tier-1 improvement alone, every new feature gets a provenance-table row, every result gets an experiment_registry.md entry.

## Architecture Decisions
- Cluster-level NDVI shrinkage follows the exact pattern already established by `ClimateAnomalyFeaturizer` (group → global fallback, min-group-size gate) and `fit_target_encoding_map` (Bayesian smoothing toward global mean) — not a new pattern invented for this feature. Consistency here matters more than a "better" bespoke design, since it keeps the codebase's leakage-safety story uniform and reviewable.
- All new feature code goes into `src/climate_health/features/climate.py` (Track D's existing home) and gets wired into `build_feature_matrix()` in `pipeline.py`, matching every prior stage's file layout — no new modules for two small functions.

## Task List

### Phase 1: Wire the existing NDVI trend feature

- [ ] **Task 1: Reproduce that `add_ndvi_trend_feature` is currently unused (Prove-It pattern)**
  **Description:** Before wiring anything in, write a test that fails today, proving the gap: `ndvi_trend_30_90` is NOT a column in `build_feature_matrix()`'s output even though the underlying function exists and is tested in isolation. This is the TDD "reproduce the bug" step applied to a wiring gap instead of a behavioral bug.
  **Why:** Confirms the finding from the doc correction is real before spending time fixing it — cheap insurance against fixing something that was already fine.
  **Acceptance criteria:**
  - [ ] A new test in `tests/test_pipeline.py` calls `build_feature_matrix(df, fit=True, target_col=...)` and asserts `"ndvi_trend_30_90" not in matrix.columns` — and it currently passes (proving the gap exists).
  **Verification:** `pytest tests/test_pipeline.py -k ndvi_trend -v` — new test passes (the "bug" reproduces).
  **Dependencies:** None.
  **Files touched:** `tests/test_pipeline.py`.
  **Scope:** XS.
  **Skill to invoke:** `/agent-skills:test-driven-development` — this is exactly the Prove-It pattern (write the failing/gap-proving test before touching the fix).

- [ ] **Task 2: Wire `add_ndvi_trend_feature` into `build_feature_matrix`**
  **Description:** Add one call to `add_ndvi_trend_feature(out)` inside `build_feature_matrix()` in `pipeline.py`, positioned alongside the other Track A-C calls (after `add_rainfall_rate_features`/`add_heat_exceedance_features`, before the interaction-features loop, since Stage 9 interactions might reasonably want to reference it later). Flip Task 1's test to assert the column IS present now.
  **Why:** This is the actual fix for the wiring gap the doc correction identified — the function and its tests already exist and are correct; this task only makes the pipeline call it.
  **Acceptance criteria:**
  - [ ] `build_feature_matrix(df, fit=True, ...)` output contains `ndvi_trend_30_90`.
  - [ ] `build_feature_matrix(df, fit=False, fitted_state=...)` on test data also contains it (no fit-only asymmetry — `add_ndvi_trend_feature` takes no fitted state, so this should be automatic, but verify explicitly since every other Track B/C feature in this pipeline DOES have a fit/transform split).
  - [ ] Task 1's test now passes with the assertion flipped to "is in columns".
  **Verification:** `pytest tests/test_pipeline.py -v` (full file, not just the new test — guards against breaking `branch_b_columns` schema-freeze logic, which reindexes on a fixed column list). `ruff check` / `black --check` clean before commit.
  **Dependencies:** Task 1.
  **Files touched:** `src/climate_health/features/pipeline.py`, `tests/test_pipeline.py`.
  **Scope:** S.
  **Skill to invoke:** `/agent-skills:test-driven-development` — GREEN step of the cycle Task 1 started RED.

- [ ] **Task 3: CV-check `ndvi_trend_30_90` under Tier 1/2/3**
  **Description:** Run `evaluate_all_tiers` (or the project's existing model-comparison harness) on `catboost_tuned`'s exact tuned config, once with and once without `ndvi_trend_30_90` in the feature set, holding everything else fixed. Record mean/std/min/max for all three tiers, both with and without.
  **Why:** This project's standing rule — no feature ships on Tier-1 improvement alone. A feature that helps Tier 1 but not Tier 2 is exactly the false-signal pattern Phase 2's validation architecture exists to catch.
  **Acceptance criteria:**
  - [ ] A comparison table (with-feature vs. without) for all three CV tiers exists, either as a notebook cell output or a small script's printed table.
  - [ ] A written keep/reject decision based on Tier-2 mean AND std (a feature that wins on mean but blows up variance is not an unconditional win — same standard Stage 13.4's locked-holdout check used).
  **Verification:** The CV script/notebook cell runs end-to-end without error and produces the table; the decision is stated in plain text, not left implicit in numbers.
  **Dependencies:** Task 2.
  **Files touched:** `notebooks/04_modeling_experiments.ipynb` (or a new small `scripts/run_stage8_5_ndvi_trend_cv.py`, matching the pattern of every other Stage 14/15 script this project already has).
  **Scope:** S.
  **Skill to invoke:** none of the build-phase skills — this is evaluation, not construction. If the result is ambiguous (e.g. Tier 2 mean improves but std also rises noticeably), invoke `/agent-skills:doubt-driven-development` for a fresh adversarial read on whether the improvement is real before deciding to keep it — cheap insurance given how small this feature's expected effect size is.

### Checkpoint: Phase 1
- [ ] `pytest` full suite green.
- [ ] `ndvi_trend_30_90` keep/reject decision made and stated in `docs/experiment_registry.md` as an F-series row (e.g. `F-005`).
- [ ] `climate.py`'s `FEATURE_PROVENANCE` entry for `ndvi_trend_30_90` updated if its status changed (it's already documented as "Needs verification" / Medium risk — update to reflect it's now actually wired in and CV-verified either way).

### Phase 2: Shrinkage-smoothed cluster-level NDVI feature

- [ ] **Task 4: Write failing tests for the new `add_ndvi_cluster_features` transformer**
  **Description:** Before writing the transformer, write tests describing its required behavior: (a) a small synthetic cluster with few rows shrinks its NDVI mean toward the global mean rather than using the raw small-sample mean; (b) a large cluster's shrunk mean stays close to its own raw mean (shrinkage should be weak when there's enough data); (c) `fit`/`transform` never lets a row's own value leak into its own cluster statistic when applied to the data it was fit on (same leave-one-out discipline as `ClimateAnomalyFeaturizer.fit_transform`); (d) an unseen cluster at transform time falls back to the global mean, never raises.
  **Why:** The Stage 8.5 audit found 2/8 clusters have n=3 and n=2 rows — a raw per-cluster NDVI mean for those is almost pure noise, and would generalize badly to test-set rows assigned to those clusters. Writing this as failing tests first pins down the exact shrinkage behavior before any implementation exists, so "how much shrinkage is enough" is a design decision made deliberately, not discovered by accident while debugging.
  **Acceptance criteria:**
  - [ ] `tests/test_climate.py` (or a new `tests/test_climate_ndvi_cluster.py`) has at least 4 tests covering (a)-(d) above, all failing (the class doesn't exist yet).
  **Verification:** `pytest tests/test_climate*.py -k ndvi_cluster -v` — all new tests fail with `ImportError`/`AttributeError`, not a wrong-assertion failure (confirms they're testing something not yet built, not a broken test).
  **Dependencies:** None (can run in parallel with Phase 1, but sequenced after it here since Phase 1 is smaller and de-risks the CV harness first).
  **Files touched:** new or existing `tests/test_climate.py`.
  **Scope:** S.
  **Skill to invoke:** `/agent-skills:test-driven-development` — RED step.

- [ ] **Task 5: Implement `add_ndvi_cluster_features` (shrinkage-smoothed cluster NDVI)**
  **Description:** A new `BaseEstimator`/`TransformerMixin` class in `climate.py`, modeled directly on `ClimateAnomalyFeaturizer`'s fit/transform/fit_transform(leave-one-out) contract: `fit()` learns per-cluster NDVI mean and count plus the global mean/count from `spatial_cluster` (Stage 7's grouping, already available by this point in the pipeline); shrinkage formula `shrunk = (n_cluster * cluster_mean + k * global_mean) / (n_cluster + k)` with a fixed prior-strength constant `k` (start with `k` equal to the existing `min_group_month_size` default of 5, or justify a different value in the docstring) — the same Bayesian-shrinkage shape `fit_target_encoding_map` already uses elsewhere in this codebase, not a new formula invented for this one feature. Adds `ndvi_30d_cluster_smoothed` (and optionally `ndvi_90d_cluster_smoothed`) columns. `transform()` uses fit-time cluster statistics only (leakage-safe for genuinely new data); `fit_transform()` uses leave-one-out on the fit data, exactly mirroring `ClimateAnomalyFeaturizer`'s documented reason for that split.
  **Why:** This is the one genuinely new feature in Stage 8.5 — the audit already proved NDVI itself is clean data, and cluster-level aggregation is the direction that survived the idea-refine convergence, but only if it doesn't quietly produce noise for the two thin clusters.
  **Acceptance criteria:**
  - [ ] All 4+ tests from Task 4 pass.
  - [ ] The transformer's docstring states the shrinkage formula and the chosen prior-strength constant with a one-sentence justification, matching the documentation density of every other transformer in this file.
  **Verification:** `pytest tests/test_climate*.py -k ndvi_cluster -v` all green; `ruff check` / `black --check` clean.
  **Dependencies:** Task 4.
  **Files touched:** `src/climate_health/features/climate.py`.
  **Scope:** M.
  **Skill to invoke:** `/agent-skills:test-driven-development` — GREEN then REFACTOR steps. Follow immediately with `/agent-skills:code-review-and-quality` before wiring it into the shared pipeline (Task 6) — this is new leakage-sensitive logic (a group-based statistic touching spatial_cluster), exactly the kind of code this project has consistently sent through review before merging (see Stage 15.3's predict.py precedent).

- [ ] **Task 6: Wire `add_ndvi_cluster_features` into `build_feature_matrix` and update `FeatureFitState`**
  **Description:** Add the new transformer to `build_feature_matrix()`, positioned after `SpatialClusterFeaturizer` (it needs `spatial_cluster` to exist) and stored in `FeatureFitState` exactly like `climate_anomaly` is — a new dataclass field, deepcopied in `__post_init__`, fit on `fit=True` and reused verbatim on `fit=False`.
  **Why:** Every other stateful transformer in this pipeline (`spatial_cluster`, `climate_anomaly`, `target_encoding`) follows the fit-once/reuse-verbatim contract enforced by `FeatureFitState`; skipping that for this feature would silently reintroduce the exact leakage risk the rest of the pipeline was built to prevent.
  **Acceptance criteria:**
  - [ ] `FeatureFitState` has a new field for the NDVI cluster transformer, deepcopied like the others.
  - [ ] `build_feature_matrix(fit=False, fitted_state=...)` on test data produces the smoothed NDVI column using train-fit statistics only (add a test asserting this — same pattern as the existing `test_pipeline_branches.py` fitted-schema tests).
  **Verification:** `pytest tests/test_pipeline.py tests/test_pipeline_branches.py -v` full green.
  **Dependencies:** Task 5.
  **Files touched:** `src/climate_health/features/pipeline.py`, `tests/test_pipeline.py` or `tests/test_pipeline_branches.py`.
  **Scope:** S.
  **Skill to invoke:** `/agent-skills:test-driven-development` (extend existing pipeline tests) followed by `/agent-skills:code-review-and-quality` for the `FeatureFitState`/leakage-safety angle specifically before this touches the shared pipeline every model depends on.

- [ ] **Task 7: CV-check the new cluster-NDVI feature under Tier 1/2/3**
  **Description:** Same procedure as Task 3 — with/without comparison on `catboost_tuned`'s exact config, all three tiers, mean/std/min/max.
  **Why:** Same standing rule as Task 3; doubly important here since this is genuinely new (not just newly-wired) logic with more room for a subtle bug to produce an illusory Tier-1 gain.
  **Acceptance criteria:**
  - [ ] Comparison table produced.
  - [ ] Explicit keep/reject decision recorded, with reasoning tied to Tier-2 mean/std specifically (not Tier 1).
  **Verification:** Script/notebook runs end-to-end; decision stated in plain text.
  **Dependencies:** Task 6.
  **Files touched:** `notebooks/04_modeling_experiments.ipynb` or `scripts/run_stage8_5_ndvi_cluster_cv.py`.
  **Scope:** S.
  **Skill to invoke:** `/agent-skills:doubt-driven-development` — apply this one regardless of how the numbers look, not only if ambiguous (unlike Task 3): this is new statistical logic touching group-based leakage, exactly the "stakes are high enough that a confident output is cheaper to verify now" case that skill is for.

### Checkpoint: Phase 2
- [ ] Full `pytest` suite green.
- [ ] Keep/reject decision recorded as a new F-series row in `docs/experiment_registry.md` (e.g. `F-006`).
- [ ] New `FEATURE_PROVENANCE` entry added to `climate.py` for `ndvi_30d_cluster_smoothed` (reference period, availability, risk tier — likely "Medium," same as the existing NDVI entries, given the same MODIS composite-lag caveat plus the new shrinkage dependency on `spatial_cluster`).
- [ ] Review with yourself (or a fresh `/agent-skills:code-review-and-quality` pass) before proceeding to Phase 3 — this is the natural pause point since Phase 2 is the highest-risk work in this plan.

### Phase 3: Low-priority elevation × zone check

- [ ] **Task 8: Cheap elevation × zone interaction CV check**
  **Description:** Add a single interaction column (`elevation * (zone == "Peri_urban")` or equivalent) as a quick, throwaway script — this does NOT need the full TDD transformer treatment given its "low priority, cheap check only" status in the amended scope doc. Run the same Tier 1/2/3 comparison.
  **Why:** The amended scope doc deprioritizes this because elevation is likely near-redundant with `spatial_cluster` (fit jointly with elevation in Stage 7) — this task exists to confirm or refute that suspicion cheaply, not to build production feature code speculatively.
  **Acceptance criteria:**
  - [ ] A comparison table exists.
  - [ ] A one-line decision: "not worth building as a real feature" (expected) or "surprisingly helped, promote to a proper transformer with tests" (unexpected — if this happens, it gets its own Task 4-6-style treatment before shipping, not a shortcut into the champion path).
  **Verification:** Script runs, table printed, decision stated.
  **Dependencies:** None (can run any time after Task 3, independent of Phase 2).
  **Files touched:** a throwaway script, not committed to `src/` unless the result is positive.
  **Scope:** XS.
  **Skill to invoke:** none — this is intentionally below the threshold where a skill is warranted; that's the point of calling it "cheap" in the scope doc. If it does surprise you and looks worth keeping, invoke `/agent-skills:test-driven-development` at that point, retroactively, before it goes anywhere near `build_feature_matrix`.

### Phase 4: Promotion, retrain, resubmit

- [ ] **Task 9: Decide the final feature set and update `configs/model_best.yaml` if needed**
  **Description:** Based on Phase 1/2/3's keep/reject decisions, decide whether any of the three candidate features actually enters the champion's retrain. If Optuna's tuned hyperparameters were sensitive to the exact feature set (unlikely for a couple of added numeric columns with a tree model, but worth a one-line check), note whether a re-tune is warranted or whether `model_best.yaml`'s existing hyperparameters are still the right ones to reuse verbatim.
  **Why:** This is the actual decision gate this whole stage exists to reach — matches the project's Stage 17 "champion model gate" discipline (improves the competition score; holds across Tier 2 folds; no leakage; reproducible).
  **Acceptance criteria:**
  - [ ] A written decision: which features (if any) are now part of the default `build_feature_matrix()` output used by `predict.py`.
  - [ ] `docs/experiment_registry.md` updated with a `C`-series row only if this becomes an actual submitted candidate (per Stage 16's own numbering rule — C-series is reserved for submission-worthy candidates, not every experiment).
  **Verification:** Re-read `docs/PROJECT_BLUEPRINT.md` Stage 17's gate checklist and confirm each item explicitly (improves score / holds across folds / no leakage / reproducible / valid submission schema).
  **Dependencies:** Tasks 3, 7, 8.
  **Files touched:** `docs/experiment_registry.md`, possibly `configs/model_best.yaml`.
  **Scope:** S.
  **Skill to invoke:** `/agent-skills:documentation-and-adrs` — this decision (what the final feature set is, and why) is exactly the kind of "expensive to silently re-derive later" call that belongs in the registry with reasoning, not just a number.

- [ ] **Task 10: Retrain, regenerate submission, resubmit**
  **Description:** Re-run `predict.py`'s `generate_submission()` (Stage 15.3's scripted entry point — never a manual notebook step, per this project's own standing rule) with whichever feature set Task 9 decided on. Validate against `SampleSubmission.csv`. Submit to Zindi. Compare the new public LB score to the current 0.8310 baseline.
  **Why:** Closes the loop — Stage 8.5 exists specifically to move this number, and the whole point of the incremental, CV-gated approach in Phases 1-3 is that this final submission is the first real test of whether local CV gains (if any) transfer to the public LB, same as the original go/no-go checkpoint discipline from Stage 16.
  **Acceptance criteria:**
  - [ ] `submission.csv` regenerated via `python -m climate_health.models.predict`, validated, named with a clean (non-dirty) commit hash.
  - [ ] Submitted to Zindi.
  - [ ] New public LB score recorded in `submissions/leaderboard_log.csv` and compared explicitly to 0.8310.
  **Verification:** The submission file's schema-validation step inside `predict.py` passes without raising; the Zindi upload confirms a score.
  **Dependencies:** Task 9.
  **Files touched:** `submission.csv`, `submissions/leaderboard_log.csv`.
  **Scope:** XS (assuming `predict.py` needs no changes — if the feature set changed, confirm `build_feature_matrix` is the only thing that needed touching, not `predict.py` itself, since it already calls `build_feature_matrix` generically).
  **Skill to invoke:** `/agent-skills:git-workflow-and-versioning` — commit the final feature-set decision and the new submission archive as one clean, atomic commit before tagging; if this becomes the new best candidate, move the `competition-submission-final` git tag to this commit per Stage 15's rule (one exact commit tagged, not left pointing at a superseded one).

## Risks and Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Cluster-NDVI shrinkage constant `k` is picked arbitrarily and doesn't actually fix the small-cluster noise problem | Medium — feature looks fine on Tier 1, quietly unstable on Tier 2 for the two thin clusters | Task 4's tests explicitly check shrinkage behavior at small vs. large cluster sizes before implementation; Task 7 mandates a doubt-driven-development pass specifically because of this risk |
| Wiring in `ndvi_trend_30_90` retroactively changes `branch_b_columns`' one-hot schema in a way that breaks `predict.py`'s test-time reindexing | Low probability, high impact (silently wrong test predictions) | Task 2's verification step explicitly re-runs the full `test_pipeline.py`/`test_pipeline_branches.py` suite, not just the new test |
| None of the three candidate features clear Tier-2 CV | Low impact — this is a legitimate, already-anticipated outcome | The amended scope doc's Open Questions already name "Stage 8.5 closes as checked, didn't help" as a valid close-out, consistent with this project's Stage 15.1 precedent ("checked, didn't matter" is a real portfolio artifact) |

## Open Questions
- Should Task 3 and Task 7's CV-check scripts be one-off (deleted after the decision) or kept as permanent `scripts/run_stage8_5_*.py` files like every prior stage's CV scripts? (Recommend: keep them, matching the project's existing pattern of one script per validated experiment.)
- Does a mid-plan submission make sense after Phase 1 alone (before Phase 2's higher-risk feature is even built), to get earlier LB signal? This was already flagged as an open question in the amended scope doc and isn't re-decided here.
