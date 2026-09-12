"""
Stage 5 validation-architecture tests.

Three kinds of coverage:
  - Correctness of the grouping primitives (twin-record grouping, union-find merging)
    against both synthetic edge cases and the real data's known Stage 3 forensics numbers.
  - Guarantees every tier actually holds (no twin-record split across train/validation
    in Tier 1 or Tier 2; Tier 3's holdout locations are genuinely disjoint from train).
  - A framework-correctness sanity check (Stage 5 guideline step 7): a synthetic,
    purely location-encoded target should score near-perfectly under Tier 1 (which
    doesn't guard geography) and collapse toward chance under Tier 2 (which does) — if
    that gap doesn't appear, the framework has a bug, regardless of what any unit test
    in isolation says.

Run with: pytest tests/test_cv.py -v
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from climate_health.data.loaders import load_train
from climate_health.evaluation.cv import (
    PLACEHOLDER_SPATIAL_CLUSTER_SOURCE,
    Tier2Result,
    Tier3Holdout,
    compute_preliminary_spatial_clusters,
    compute_twin_groups,
    make_tier3_holdout,
    merge_groups_to_respect_constraint,
    summarize_tier2_scores,
    tier1_splits,
    tier2_splits,
)

RANDOM_STATE = 42


@pytest.fixture(scope="module")
def train_df() -> pd.DataFrame:
    return load_train()


# ---------------------------------------------------------------------------
# compute_twin_groups
# ---------------------------------------------------------------------------


def test_compute_twin_groups_basic_correctness():
    df = pd.DataFrame(
        {
            "location": ["A", "A", "B", "B", "C"],
            "deathdate": pd.to_datetime(
                ["2020-01-01", "2020-01-01", "2020-01-01", "2020-01-02", "2020-01-01"]
            ),
        }
    )
    groups = compute_twin_groups(df, keys=("location", "deathdate"))
    # rows 0,1 share (A, 2020-01-01) -> same group
    assert groups[0] == groups[1]
    # rows 2,3 differ in date -> different groups
    assert groups[2] != groups[3]
    # row 4 is a singleton, distinct from everything else
    assert groups[4] not in (groups[0], groups[2], groups[3])
    assert len(np.unique(groups)) == 4  # {A,01-01}, {B,01-01}, {B,01-02}, {C,01-01}


def test_compute_twin_groups_no_delimiter_collision():
    # Regression test: the original implementation built a composite key via
    # "||".join(str(v) for v in row). That scheme has a real collision: two genuinely
    # different (location, deathdate) pairs can concatenate to the identical string —
    # e.g. ("A||B", "C") and ("A", "B||C") both join to "A||B||C" — which would
    # incorrectly merge two unrelated records into one "twin" group. The fixed
    # implementation groups directly on the key columns' actual values (pandas
    # groupby), which cannot exhibit this collision.
    df = pd.DataFrame(
        {
            "location": ["A||B", "A", "C", "C"],
            "deathdate": ["C", "B||C", "D", "D"],
        }
    )
    groups = compute_twin_groups(df, keys=("location", "deathdate"))
    # rows 0 and 1 are genuinely different (location, deathdate) pairs that collide
    # under the old string-join scheme — must NOT be grouped together.
    assert groups[0] != groups[1]
    # sanity: rows 2 and 3 are genuinely identical and must still be grouped together.
    assert groups[2] == groups[3]
    assert len(np.unique(groups)) == 3


def test_compute_twin_groups_non_default_index_raises():
    df = pd.DataFrame(
        {
            "location": ["A", "A", "B"],
            "deathdate": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"]),
        },
        index=[5, 6, 7],
    )
    with pytest.raises(ValueError, match="RangeIndex"):
        compute_twin_groups(df, keys=("location", "deathdate"))


def test_compute_twin_groups_missing_column_raises():
    df = pd.DataFrame({"location": ["A"], "deathdate": pd.to_datetime(["2020-01-01"])})
    with pytest.raises(KeyError):
        compute_twin_groups(df, keys=("location", "nonexistent_col"))


def test_compute_twin_groups_null_key_raises():
    df = pd.DataFrame(
        {
            "location": ["A", None],
            "deathdate": pd.to_datetime(["2020-01-01", "2020-01-01"]),
        }
    )
    with pytest.raises(ValueError):
        compute_twin_groups(df, keys=("location", "deathdate"))


def test_compute_twin_groups_matches_stage3_forensics(train_df):
    # Stage 3's notebooks/00_data_forensics.ipynb found exactly 49 within-train groups
    # sharing (location, deathdate) with more than one record — this is the empirical
    # number the whole Tier 1/2 twin-guard design is built to protect against, so this
    # test pins it down: if this number ever drifts (e.g. because of a data refresh),
    # it should be a visible, deliberate event, not silently absorbed.
    groups = compute_twin_groups(train_df, keys=("location", "deathdate"))
    group_sizes = pd.Series(groups).value_counts()
    n_groups_with_twins = (group_sizes > 1).sum()
    n_records_in_twin_groups = group_sizes[group_sizes > 1].sum()
    assert n_groups_with_twins == 49, (
        f"Expected 49 within-train (location, deathdate) twin groups per Stage 3 "
        f"forensics, found {n_groups_with_twins} — investigate before trusting Tier 1/2."
    )
    assert n_records_in_twin_groups == 98


# ---------------------------------------------------------------------------
# merge_groups_to_respect_constraint (union-find)
# ---------------------------------------------------------------------------


def test_merge_groups_no_conflict_is_relabeling_only():
    # Every constraint group already sits fully within one primary group -> output
    # should just be a relabeling of the primary groups, same number of unique groups.
    primary = np.array([0, 0, 1, 1, 2, 2])
    constraint = np.array([10, 10, 20, 20, 30, 30])
    merged = merge_groups_to_respect_constraint(primary, constraint)
    assert len(np.unique(merged)) == 3
    assert merged[0] == merged[1]
    assert merged[2] == merged[3]
    assert merged[4] == merged[5]


def test_merge_groups_conflict_forces_merge():
    # constraint group 100 spans primary groups 0 and 1 -> those two primary groups
    # must be merged into one final group.
    primary = np.array([0, 1, 2])
    constraint = np.array([100, 100, 200])
    merged = merge_groups_to_respect_constraint(primary, constraint)
    assert merged[0] == merged[1]  # forced together by the shared constraint group
    assert merged[2] != merged[0]  # untouched, stays separate
    assert len(np.unique(merged)) == 2


def test_merge_groups_transitive_chain_merges_correctly():
    # A chain of conflicts (0-1 via one constraint group, 1-2 via another) should merge
    # all three primary groups into one, via transitive closure.
    primary = np.array([0, 1, 1, 2])
    constraint = np.array([100, 100, 200, 200])
    merged = merge_groups_to_respect_constraint(primary, constraint)
    assert len(np.unique(merged)) == 1


def test_merge_groups_length_mismatch_raises():
    with pytest.raises(ValueError):
        merge_groups_to_respect_constraint(np.array([0, 1]), np.array([0, 1, 2]))


# ---------------------------------------------------------------------------
# compute_preliminary_spatial_clusters
# ---------------------------------------------------------------------------


def test_compute_preliminary_spatial_clusters_shape(train_df):
    clusters = compute_preliminary_spatial_clusters(
        train_df, n_clusters=8, random_state=RANDOM_STATE
    )
    assert len(clusters) == len(train_df)
    assert len(np.unique(clusters)) <= 8
    assert len(np.unique(clusters)) >= 2  # sanity: shouldn't collapse to one cluster


def test_compute_preliminary_spatial_clusters_missing_coord_raises(train_df):
    with pytest.raises(KeyError):
        compute_preliminary_spatial_clusters(train_df, coord_cols=("latitude", "not_a_column"))


def test_compute_preliminary_spatial_clusters_deterministic(train_df):
    c1 = compute_preliminary_spatial_clusters(train_df, random_state=7)
    c2 = compute_preliminary_spatial_clusters(train_df, random_state=7)
    np.testing.assert_array_equal(c1, c2)


# ---------------------------------------------------------------------------
# Tier 1
# ---------------------------------------------------------------------------


def test_tier1_splits_no_twin_leakage(train_df):
    twin_groups = compute_twin_groups(train_df, keys=("location", "deathdate"))
    n_folds_checked = 0
    for train_idx, val_idx in tier1_splits(
        train_df, target_col="is_climate_sensitive", n_splits=5, random_state=RANDOM_STATE
    ):
        train_groups = set(twin_groups[train_idx])
        val_groups = set(twin_groups[val_idx])
        overlap = train_groups & val_groups
        assert (
            not overlap
        ), f"Tier 1 split leaked {len(overlap)} twin group(s) across train/validation"
        n_folds_checked += 1
    assert n_folds_checked == 5


def test_tier1_splits_cover_all_rows_exactly_once_per_fold_set(train_df):
    all_val_idx = []
    for train_idx, val_idx in tier1_splits(
        train_df, target_col="is_climate_sensitive", n_splits=5, random_state=RANDOM_STATE
    ):
        assert len(set(train_idx) & set(val_idx)) == 0  # no row is both train and val
        all_val_idx.extend(val_idx.tolist())
    # every row appears in exactly one validation fold across the 5 folds
    assert sorted(all_val_idx) == list(range(len(train_df)))


def test_tier1_splits_roughly_stratified(train_df):
    overall_rate = train_df["is_climate_sensitive"].mean()
    for _train_idx, val_idx in tier1_splits(
        train_df, target_col="is_climate_sensitive", n_splits=5, random_state=RANDOM_STATE
    ):
        fold_rate = train_df["is_climate_sensitive"].values[val_idx].mean()
        # StratifiedGroupKFold approximates stratification subject to the group
        # constraint -- allow a reasonably generous tolerance rather than assuming exact
        # stratification, which the group constraint can prevent from being perfect.
        assert (
            abs(fold_rate - overall_rate) < 0.08
        ), f"fold target rate {fold_rate:.3f} too far from overall {overall_rate:.3f}"


def test_tier1_splits_missing_target_raises(train_df):
    with pytest.raises(KeyError):
        list(tier1_splits(train_df, target_col="not_a_real_column"))


def test_tier1_splits_null_target_raises():
    df = pd.DataFrame(
        {
            "location": ["A", "B", "C", "D", "E", "F"],
            "deathdate": pd.to_datetime(["2020-01-01"] * 6),
            "is_climate_sensitive": [0, 1, 0, 1, None, 1],
        }
    )
    with pytest.raises(ValueError, match="null"):
        list(tier1_splits(df, target_col="is_climate_sensitive", n_splits=2))


def test_tier1_splits_non_default_index_raises(train_df):
    df = train_df.copy()
    df.index = df.index + 100  # still a RangeIndex, but not starting at 0
    with pytest.raises(ValueError, match="RangeIndex"):
        list(tier1_splits(df, target_col="is_climate_sensitive", n_splits=5))


def test_tier1_splits_too_many_splits_raises():
    # 3 unique twin-groups can't support 5 splits without violating the constraint.
    df = pd.DataFrame(
        {
            "location": ["A", "B", "C"] * 2,
            "deathdate": pd.to_datetime(["2020-01-01"] * 6),
            "is_climate_sensitive": [0, 1, 0, 1, 0, 1],
        }
    )
    with pytest.raises(ValueError):
        list(tier1_splits(df, target_col="is_climate_sensitive", n_splits=5))


# ---------------------------------------------------------------------------
# Tier 2
# ---------------------------------------------------------------------------


def test_tier2_splits_no_twin_leakage(train_df):
    twin_groups = compute_twin_groups(train_df, keys=("location", "deathdate"))
    n_splits_checked = 0
    for split in tier2_splits(train_df, n_splits=5, n_repeats=3, random_state=RANDOM_STATE):
        assert isinstance(split, Tier2Result)
        train_groups = set(twin_groups[split.train_idx])
        val_groups = set(twin_groups[split.val_idx])
        overlap = train_groups & val_groups
        assert not overlap, (
            f"Tier 2 repeat {split.repeat} fold {split.fold} leaked "
            f"{len(overlap)} twin group(s) across train/validation"
        )
        n_splits_checked += 1
    assert n_splits_checked == 5 * 3


def test_tier2_splits_no_row_overlap_within_a_fold(train_df):
    for split in tier2_splits(train_df, n_splits=5, n_repeats=1, random_state=RANDOM_STATE):
        assert len(set(split.train_idx.tolist()) & set(split.val_idx.tolist())) == 0


def test_tier2_repeats_produce_different_fold_assignments(train_df):
    # If every repeat produced the identical fold assignment, "repeated" would be
    # theater, not a real source of additional information about score stability.
    val_idx_by_repeat = {}
    for split in tier2_splits(train_df, n_splits=5, n_repeats=3, random_state=RANDOM_STATE):
        if split.fold == 0:
            val_idx_by_repeat.setdefault(split.repeat, set()).update(split.val_idx.tolist())
    assert len(val_idx_by_repeat) == 3
    repeats = list(val_idx_by_repeat.values())
    # at least one pair of repeats should differ in which rows land in fold 0's validation set
    assert not (repeats[0] == repeats[1] == repeats[2])


def test_tier2_uses_placeholder_and_warns_when_no_cluster_col_given(train_df):
    with pytest.warns(UserWarning, match=PLACEHOLDER_SPATIAL_CLUSTER_SOURCE):
        next(tier2_splits(train_df, n_splits=5, n_repeats=1, random_state=RANDOM_STATE))


def test_tier2_accepts_precomputed_cluster_column_without_warning(train_df, recwarn):
    df = train_df.copy()
    df["my_cluster"] = compute_preliminary_spatial_clusters(df, random_state=RANDOM_STATE)
    list(
        tier2_splits(
            df, spatial_cluster_col="my_cluster", n_splits=5, n_repeats=1, random_state=RANDOM_STATE
        )
    )
    placeholder_warnings = [
        w for w in recwarn.list if PLACEHOLDER_SPATIAL_CLUSTER_SOURCE in str(w.message)
    ]
    assert not placeholder_warnings


def test_tier2_missing_cluster_col_raises(train_df):
    with pytest.raises(KeyError):
        list(
            tier2_splits(train_df, spatial_cluster_col="not_a_real_column", n_splits=5, n_repeats=1)
        )


def test_tier2_fold_assignment_is_row_count_balanced_with_fine_grained_clusters(train_df):
    # Regression test for the fold-size-imbalance bug found during review: the original
    # implementation assigned groups to folds round-robin by *group count*, ignoring
    # each group's *row count* — empirically confirmed (against the real placeholder
    # k=8 clustering) to produce validation folds ranging from ~5% to ~36% of the data
    # for a nominal 5-fold split. With a reasonably fine-grained clustering (many
    # roughly-similar-sized groups, unlike the coarse 8-cluster placeholder), the fixed
    # row-count-balanced assignment should keep every fold close to the 1/n_splits
    # target share.
    df = train_df.copy()
    n_splits = 5
    # 20 synthetic clusters built as a deterministic function of the *twin group id*
    # (not row position) so that every twin-record group is already fully contained in
    # one cluster — this isolates the fold-assignment algorithm's row-count balancing
    # from `merge_groups_to_respect_constraint`'s twin-merging behavior (a cluster
    # assignment that ignores twin structure, e.g. plain row-order buckets, can trigger
    # a cascade of merges that collapses most of the data into one giant group,
    # confounding this test with unrelated behavior).
    twin_groups_for_synthetic = compute_twin_groups(df, keys=("location", "deathdate"))
    df["fine_cluster"] = twin_groups_for_synthetic % 20

    target_frac = 1.0 / n_splits
    for split in tier2_splits(
        df,
        spatial_cluster_col="fine_cluster",
        n_splits=n_splits,
        n_repeats=3,
        random_state=RANDOM_STATE,
    ):
        val_frac = len(split.val_idx) / len(df)
        assert val_frac == pytest.approx(target_frac, abs=0.06), (
            f"repeat={split.repeat} fold={split.fold} val_frac={val_frac:.3f} is not "
            f"close to the {target_frac:.3f} target — fold assignment is not "
            "row-count balanced."
        )


def test_tier2_fold_sizes_near_optimal_given_placeholder_clustering(train_df):
    # The placeholder k=8 KMeans clustering only yields ~8 groups total (see module
    # docstring), which structurally bounds how balanced a 5-way split can ever be: no
    # algorithm can avoid an oversized fold once a single group's row count exceeds a
    # fold's fair share. This test checks the *achievable* bound (derived directly from
    # the actual group sizes) is nearly met, rather than asserting an arbitrary tight
    # balance that the data cannot support.
    n_splits = 5
    twin_groups = compute_twin_groups(train_df, keys=("location", "deathdate"))
    spatial_clusters = compute_preliminary_spatial_clusters(train_df, random_state=RANDOM_STATE)
    final_groups = merge_groups_to_respect_constraint(spatial_clusters, twin_groups)
    group_sizes = pd.Series(final_groups).value_counts()
    largest_group_frac = group_sizes.max() / len(train_df)

    max_val_frac = 0.0
    for split in tier2_splits(train_df, n_splits=n_splits, n_repeats=1, random_state=RANDOM_STATE):
        max_val_frac = max(max_val_frac, len(split.val_idx) / len(train_df))

    # The largest fold can never be smaller than the largest single group's share; a
    # good bin-packing algorithm should land close to that structural floor, not
    # meaningfully above it (generous 12-point slack for the remaining groups' packing).
    assert max_val_frac >= largest_group_frac - 1e-9
    assert max_val_frac <= largest_group_frac + 0.12


def test_summarize_tier2_scores_basic():
    summary = summarize_tier2_scores([0.5, 0.6, 0.7, 0.8])
    assert summary["mean"] == pytest.approx(0.65)
    assert summary["min"] == 0.5
    assert summary["max"] == 0.8
    assert summary["n"] == 4
    assert summary["std"] > 0


def test_summarize_tier2_scores_empty_raises():
    with pytest.raises(ValueError):
        summarize_tier2_scores([])


def test_summarize_tier2_scores_single_value_zero_std():
    summary = summarize_tier2_scores([0.7])
    assert summary["std"] == 0.0


# ---------------------------------------------------------------------------
# Tier 3
# ---------------------------------------------------------------------------


def test_make_tier3_holdout_locations_disjoint_from_train(train_df):
    holdout = make_tier3_holdout(train_df, n_trials=500, random_state=RANDOM_STATE)
    assert isinstance(holdout, Tier3Holdout)
    train_locations = set(train_df.iloc[holdout.train_idx]["location"])
    holdout_locations_actual = set(train_df.iloc[holdout.holdout_idx]["location"])
    assert train_locations & holdout_locations_actual == set()
    assert holdout_locations_actual == holdout.holdout_locations


def test_make_tier3_holdout_no_row_overlap(train_df):
    holdout = make_tier3_holdout(train_df, n_trials=500, random_state=RANDOM_STATE)
    assert len(set(holdout.train_idx.tolist()) & set(holdout.holdout_idx.tolist())) == 0
    assert len(holdout.train_idx) + len(holdout.holdout_idx) == len(train_df)


def test_make_tier3_holdout_reasonably_matches_targets(train_df):
    # Stage 4 found the real test set is ~24.7% of train+test and 76.21% Rural. A
    # randomized search over location subsets won't hit these exactly, but should get
    # reasonably close -- loose tolerances here since this is inherently a best-effort
    # match, not an exact one.
    holdout = make_tier3_holdout(train_df, n_trials=3000, random_state=RANDOM_STATE)
    assert abs(holdout.achieved_holdout_fraction - holdout.target_holdout_fraction) < 0.05
    assert abs(holdout.achieved_rural_fraction - holdout.target_rural_fraction) < 0.10


def test_make_tier3_holdout_deterministic(train_df):
    h1 = make_tier3_holdout(train_df, n_trials=500, random_state=123)
    h2 = make_tier3_holdout(train_df, n_trials=500, random_state=123)
    assert h1.holdout_locations == h2.holdout_locations


def test_make_tier3_holdout_inconsistent_zone_raises():
    df = pd.DataFrame(
        {
            "location": ["A", "A", "B"],
            "zone": ["Rural", "Peri_urban", "Rural"],  # location A has two different zones
        }
    )
    with pytest.raises(ValueError):
        make_tier3_holdout(df, n_trials=10)


def test_make_tier3_holdout_missing_column_raises(train_df):
    with pytest.raises(KeyError):
        make_tier3_holdout(train_df, location_col="not_a_real_column", n_trials=10)


def test_make_tier3_holdout_null_location_raises():
    df = pd.DataFrame(
        {
            "location": ["A", None, "B", "B"],
            "zone": ["Rural", "Rural", "Peri_urban", "Peri_urban"],
        }
    )
    with pytest.raises(ValueError, match="null"):
        make_tier3_holdout(df, n_trials=10)


def test_make_tier3_holdout_null_zone_raises():
    df = pd.DataFrame(
        {
            "location": ["A", "A", "B", "B"],
            "zone": ["Rural", "Rural", None, "Peri_urban"],
        }
    )
    with pytest.raises(ValueError, match="null"):
        make_tier3_holdout(df, n_trials=10)


def test_make_tier3_holdout_non_default_index_raises(train_df):
    df = train_df.copy()
    df.index = df.index + 1
    with pytest.raises(ValueError, match="RangeIndex"):
        make_tier3_holdout(df, n_trials=10)


def test_evaluate_all_tiers_passes_through_tier3_location_and_zone_cols(train_df):
    # Regression test: evaluate_all_tiers used to hardcode make_tier3_holdout's
    # location_col/zone_col to their defaults ("location"/"zone") instead of exposing
    # them as parameters. Rename the columns and confirm the orchestrator still finds
    # them when told where to look — this would raise KeyError before the fix (since
    # "location"/"zone" would no longer exist) if the passthrough were missing.
    from sklearn.dummy import DummyClassifier

    from climate_health.evaluation.cv import evaluate_all_tiers

    df = train_df.rename(columns={"location": "loc2", "zone": "zone2"})
    result = evaluate_all_tiers(
        make_estimator=lambda: DummyClassifier(strategy="stratified", random_state=0),
        df=df,
        feature_cols=["latitude", "longitude", "age"],
        target_col="is_climate_sensitive",
        score_fn=lambda y_true, y_pred: 0.5,  # score value is irrelevant to this test
        twin_keys=("loc2", "deathdate"),
        tier1_n_splits=3,
        tier2_n_splits=3,
        tier2_n_repeats=1,
        tier3_location_col="loc2",
        tier3_zone_col="zone2",
        tier3_n_trials=50,
        random_state=RANDOM_STATE,
    )
    assert result.tier3_holdout is not None
    assert len(result.tier3_holdout.holdout_idx) > 0


# ---------------------------------------------------------------------------
# Framework-correctness sanity check (Stage 5 guideline step 7)
#
# A model that can only succeed by memorizing which spatial cluster a point belongs to
# should score near-perfectly under Tier 1 (no geographic guard) and collapse toward
# chance under Tier 2 (geographic groups fully held out). If this gap doesn't appear,
# the CV framework has a bug -- this is infrastructure verification, not a test of the
# real target.
# ---------------------------------------------------------------------------


def test_framework_detects_geographic_overfitting(train_df):
    df = train_df.copy()
    clusters = compute_preliminary_spatial_clusters(df, n_clusters=8, random_state=RANDOM_STATE)
    df["_cluster"] = clusters

    # Assign each cluster a binary label via a size-balanced greedy partition (not an
    # independent random draw per cluster -- with only 8 clusters, an independent random
    # draw can easily land 6-7 clusters on the same label by chance, e.g. a ~91%/9% split
    # was observed with this exact random_state, which would make "chance" accuracy ~91%
    # rather than ~50% and make the test's threshold meaningless). Balancing by row count
    # keeps the global class rate near 50/50 regardless of luck, so Tier 2's accuracy is a
    # genuine test of geographic generalization, not an artifact of label imbalance.
    cluster_sizes = pd.Series(clusters).value_counts().sort_values(ascending=False)
    group_totals = {0: 0, 1: 0}
    cluster_labels = {}
    for cluster_id, size in cluster_sizes.items():
        smaller_group = 0 if group_totals[0] <= group_totals[1] else 1
        cluster_labels[cluster_id] = smaller_group
        group_totals[smaller_group] += size
    df["_synthetic_target"] = df["_cluster"].map(cluster_labels)
    global_rate = df["_synthetic_target"].mean()
    assert 0.35 < global_rate < 0.65, (
        f"Synthetic target's global rate {global_rate:.3f} isn't balanced -- the "
        "size-balanced partition didn't work as intended, fix before trusting this test."
    )

    def make_estimator():
        # n_neighbors=1 on raw coordinates: can perfectly "memorize" a cluster's label
        # if any same-cluster point is present in its training fold.
        return KNeighborsClassifier(n_neighbors=1)

    X = df[["latitude", "longitude"]].values
    y = df["_synthetic_target"].values

    # Accuracy, not AUC: with a target that's fully deterministic per cluster, a
    # geographically-held-out validation fold can easily be single-class (every member
    # of the held-out cluster(s) shares one label), which makes ROC AUC undefined. Since
    # the label is deterministic rather than probabilistic here, accuracy is both simpler
    # and doesn't have this failure mode.
    tier1_scores = []
    for train_idx, val_idx in tier1_splits(
        df, target_col="_synthetic_target", n_splits=5, random_state=RANDOM_STATE
    ):
        est = make_estimator()
        est.fit(X[train_idx], y[train_idx])
        preds = est.predict(X[val_idx])
        tier1_scores.append(accuracy_score(y[val_idx], preds))

    tier2_scores = []
    for split in tier2_splits(
        df, spatial_cluster_col="_cluster", n_splits=5, n_repeats=3, random_state=RANDOM_STATE
    ):
        est = make_estimator()
        est.fit(X[split.train_idx], y[split.train_idx])
        preds = est.predict(X[split.val_idx])
        tier2_scores.append(accuracy_score(y[split.val_idx], preds))

    tier1_mean = np.mean(tier1_scores)
    tier2_mean = np.mean(tier2_scores)

    assert tier1_mean > 0.90, (
        f"Tier 1 mean accuracy {tier1_mean:.3f} unexpectedly low for a memorizable, "
        "non-geographically-guarded synthetic target -- framework may be over-guarding Tier 1."
    )
    # Tier 2 should be close to chance (0.5) on a target that's impossible to generalize
    # to unseen clusters -- allow a fairly wide band since only 8 clusters / a handful of
    # held-out clusters per fold makes this noisy, but it must not look like Tier 1.
    assert abs(tier2_mean - 0.5) < 0.20, (
        f"Tier 2 mean accuracy {tier2_mean:.3f} too far from chance (0.5) for a target "
        "that's impossible to generalize to unseen clusters -- Tier 2 may be leaking geography."
    )
    assert tier1_mean - tier2_mean > 0.30, (
        "Expected a large Tier 1 vs Tier 2 gap on a purely location-encoded synthetic "
        "target -- this gap is the whole point of having two separate tiers."
    )
