# Stage 8.5 — Task List (see tasks/plan.md for full detail on each task)

## Phase 1: Wire the existing NDVI trend feature
- [ ] Task 1: Reproduce that `add_ndvi_trend_feature` is currently unused (failing/gap test) — skill: test-driven-development
- [ ] Task 2: Wire `add_ndvi_trend_feature` into `build_feature_matrix` — skill: test-driven-development
- [ ] Task 3: CV-check `ndvi_trend_30_90` under Tier 1/2/3 — skill: doubt-driven-development (if ambiguous)

### Checkpoint: Phase 1
- [ ] Full pytest suite green
- [ ] Keep/reject decision recorded in docs/experiment_registry.md (F-005)
- [ ] climate.py FEATURE_PROVENANCE entry updated for ndvi_trend_30_90

## Phase 2: Shrinkage-smoothed cluster-level NDVI feature
- [ ] Task 4: Write failing tests for `add_ndvi_cluster_features` — skill: test-driven-development
- [ ] Task 5: Implement `add_ndvi_cluster_features` — skill: test-driven-development, then code-review-and-quality
- [ ] Task 6: Wire into `build_feature_matrix` + `FeatureFitState` — skill: test-driven-development, then code-review-and-quality
- [ ] Task 7: CV-check the new feature under Tier 1/2/3 — skill: doubt-driven-development (always, not just if ambiguous)

### Checkpoint: Phase 2
- [ ] Full pytest suite green
- [ ] Keep/reject decision recorded in docs/experiment_registry.md (F-006)
- [ ] New FEATURE_PROVENANCE entry added for ndvi_30d_cluster_smoothed
- [ ] Review pause before Phase 3

## Phase 3: Low-priority elevation x zone check
- [ ] Task 8: Cheap elevation x zone interaction CV check — skill: none (unless it surprises you, then test-driven-development)

## Phase 4: Promotion, retrain, resubmit
- [ ] Task 9: Decide final feature set, update configs/model_best.yaml if needed — skill: documentation-and-adrs
- [ ] Task 10: Retrain, regenerate submission, resubmit to Zindi — skill: git-workflow-and-versioning
