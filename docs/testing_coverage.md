# Stage 18 Test Coverage Audit

Author: Henry Otsyula. Written 2026-10-03 as Task 18.1 of the Phase 7 production-engineering plan (`docs/ideas/stage18-20-phase7-plan.md`). Maps every test in `tests/` against the coverage list named in `docs/PROJECT_BLUEPRINT.md`'s Stage 18. A named checklist, not a percentage -- the question is whether each blueprint-named check exists and is real, not how many lines are exercised.

## Blueprint-named checks

| # | Blueprint item | Test file | Status |
|---|---|---|---|
| 1 | Schemas (Stage 2) | `tests/test_schemas.py` | Covered -- 15 tests, including malformed IDs, out-of-range coordinates, leaked target column, missing files |
| 2a | Temporal features (Stage 6) | `tests/test_temporal.py` | Covered |
| 2b | Demographic features (Stage 6) | `tests/test_demographic.py` | Covered |
| 2c | Spatial features (Stage 7) | `tests/test_spatial.py` | Covered |
| 2d | Climate features (Stage 8) | `tests/test_climate.py` | Covered -- largest file in the suite (60+ tests: ratio features, heat thresholds, NDVI, anomaly tiers) |
| 2e | Interaction features (Stage 9) | `tests/test_interactions.py` | Covered |
| 2f | Target encoding (Stage 9) | `tests/test_encoding.py` | Covered |
| 3 | CV splitter, all three tiers (Stage 5) | `tests/test_cv.py` | Covered -- twin-record grouping, Tier 1/2/3 splits, repeated-fold stability, `test_framework_detects_geographic_overfitting` |
| 4 | Leakage / OOF-encoding checks (Stage 9) | `tests/test_encoding.py::TestOofLeakage` | Covered -- confirmed by reading the test bodies, not just names: `test_oof_value_matches_hand_computed_other_folds_only` and `test_changing_a_fold_own_labels_does_not_change_its_own_oof_value` directly verify OOF values never depend on their own fold's labels; `test_overlapping_splits_raise`, `test_train_val_overlap_within_fold_raises`, `test_incomplete_coverage_raises` guard the split contract itself |
| 5 | End-to-end schema smoke test (Stage 17) | `tests/test_submission_schema.py` | Partial -- validates schema against the already-generated, archived `submissions/submission_745a64f.csv`, not a fresh run of `predict.py` against raw `Train.csv`/`Test.csv`. This is Task 18.2's scope, closed there (`tests/test_pipeline_smoke.py`), not a gap fixed here. |

## Additional coverage (not blueprint-named, kept for completeness)

`test_baselines.py`, `test_calibrate.py`, `test_ensemble.py`, `test_environment_smoke.py`, `test_interface.py`, `test_pipeline.py`, `test_pipeline_branches.py`, `test_predict.py`, `test_tracking.py`, `test_tune.py`, `test_utils_validation.py`, `test_zoo.py` -- cover Stages 10, 11, 13, 14, 15, and 17 respectively. Not blueprint Stage 18 requirements; listed here only so the full 20-file suite is accounted for.

## Verdict

Every blueprint-named Stage 18 check has a real, substantive test file. No net-new gap requires a TDD cycle under this task -- the one partial item (#5) is the explicitly-scoped subject of Task 18.2.
