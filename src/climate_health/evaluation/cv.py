"""
Multi-tier cross-validation strategy (Stage 5).

Why three tiers, not one: Stage 3's forensics and Stage 4's EDA confirmed two distinct,
real leakage mechanisms in this dataset, not hypothetical ones —

  1. Location-generalization leakage: 0% exact coordinate overlap between Train and Test
     (43 unique train coordinates, 12 unique test coordinates, zero shared), and Stage 4's
     adversarial validation found `latitude`/`longitude`/`elevation`/`slope` alone separate
     train from test almost perfectly (AUC ~1.0). A plain random K-fold never encounters
     this train/test relationship during development.
  2. Non-independence ("twin record") leakage: 49-55 groups of *training* records share a
     `(location, deathdate)` and therefore identical climate features, because they're
     different people who died the same day in the same place. A plain K-fold, even
     stratified, can split these twins across train/validation and inflate the apparent
     score regardless of genuine generalization.

These are guarded separately, not conflated — see docs/PROJECT_BLUEPRINT.md Stage 5.

Tier 1 (Standard) answers "does this change help at all," guarded against twin leakage.
Tier 2 (Geographic) answers "does this generalize to unseen geography," guarded against
    both leakage mechanisms, and reported as a distribution (mean/std/min/max) across
    repeated splits, not a single number, since only ~39 location groups exist.
Tier 3 (Test-like holdout) answers "how well would this do on data that resembles the
    real test set specifically" — used sparingly (see Phase 6's champion gate) so it
    isn't itself optimized into uselessness.

A note on Tier 2's spatial grouping: the principled spatial-clustering feature is Phase 3
Stage 7 work, not yet built. Until then, Tier 2 uses a placeholder clustering
(`compute_preliminary_spatial_clusters`, the same approach prototyped diagnostically in
notebooks/01_eda.ipynb Section 6) so this stage isn't blocked on Phase 3. Every function
that consumes a spatial-cluster grouping logs which source produced it, so a later run
using Phase 3's real clustering is never silently compared against an old placeholder-based
result as if they were the same thing.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedGroupKFold

logger = logging.getLogger(__name__)

PLACEHOLDER_SPATIAL_CLUSTER_SOURCE = "preliminary_kmeans_k8"

DEFAULT_TWIN_KEYS = ("location", "deathdate")

# Stage 4's confirmed train/test distributional differences (notebooks/01_eda.ipynb,
# Section 6 / Executive Summary) — used as Tier 3's matching targets by default.
STAGE4_TARGET_RURAL_FRACTION = 0.7621  # test zone == "Rural" proportion
STAGE4_TARGET_HOLDOUT_FRACTION = 0.25  # test / (train + test) is ~1030 / 4176 ~= 0.247


def _assert_default_range_index(df: pd.DataFrame, fn_name: str) -> None:
    """Every splitter in this module returns *positional* integer indices (safe for
    `.iloc`, not `.loc`) — see the "Positional indices" note in each public function's
    docstring. That guarantee only holds if `df`'s index is the default 0..n-1
    RangeIndex; a DataFrame that has been filtered, merged, or re-indexed upstream
    (e.g. after a `.query()`) could otherwise cause a caller using `.loc` to silently
    select the wrong rows. Failing loudly here is cheap insurance against that class of
    bug — call `df.reset_index(drop=True)` before passing data to this module if this
    raises.
    """
    if not isinstance(df.index, pd.RangeIndex) or df.index.start != 0 or df.index.step != 1:
        raise ValueError(
            f"{fn_name}: df must have a default RangeIndex (0..n-1, step 1) because "
            "this module's return values are positional integer indices intended for "
            "`.iloc`, not `.loc`. Call `df.reset_index(drop=True)` first."
        )


# =============================================================================
# Twin-record grouping
# =============================================================================


def compute_twin_groups(df: pd.DataFrame, keys: Sequence[str] = DEFAULT_TWIN_KEYS) -> np.ndarray:
    """Assigns an integer group id to every row sharing the same value across `keys`.

    Default keys are `(location, deathdate)`, matching Stage 3's forensics finding of
    49 within-train groups sharing this combination (98 records, ~3% of train) — records
    that received identical climate-feature rows because they're different people who
    died the same day in the same place, and are therefore not independent draws.

    Rows with a unique key combination each get their own singleton group — this
    function does not filter, it labels every row, singleton or not, so it can be
    passed directly as a `groups=` argument to any group-aware splitter.

    Raises
    ------
    KeyError
        If any of `keys` is missing from `df`.
    ValueError
        If any of `keys` contains a null value — a null key would silently group
        unrelated records together (all rows with a null date, for instance), which is
        worse than failing loudly.

    Notes
    -----
    Returns *positional* integer group ids aligned to `df`'s row order (i.e. `df` must
    have a default RangeIndex — see `_assert_default_range_index`).
    """
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise KeyError(f"compute_twin_groups: missing required column(s) {missing}")
    _assert_default_range_index(df, "compute_twin_groups")
    key_df = df[list(keys)]
    if key_df.isna().any().any():
        bad_cols = key_df.columns[key_df.isna().any()].tolist()
        raise ValueError(
            f"compute_twin_groups: null values found in key column(s) {bad_cols} — "
            "a null key would silently merge unrelated rows into one group."
        )
    # Group directly on the key columns' actual values (not a string-concatenated
    # composite key) so two genuinely different key combinations can never collide into
    # the same group purely because their string forms concatenate to the same text
    # (e.g. a stray literal delimiter substring inside a value). `ngroup()` assigns a
    # dense integer id per distinct combination, order-independent of row order.
    group_ids = df.groupby(list(keys), sort=True).ngroup().to_numpy()
    return group_ids


# =============================================================================
# Group merging (union-find) — guarantees twin records are never split even when a
# different, independently-computed grouping (e.g. spatial cluster) is also in play.
# =============================================================================


class _UnionFind:
    """Minimal union-find (disjoint-set) with path compression and union by rank.

    Used to merge two independently-computed group labelings so that any two records
    sharing *either* grouping's label end up in the same final group. This is what lets
    Tier 2 group by spatial cluster while still guaranteeing no twin-record pair is ever
    split, even in the edge case where two twins happen to fall into different
    preliminary spatial clusters (e.g. a coordinate sitting near a cluster boundary).
    """

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def merge_groups_to_respect_constraint(
    primary_group_ids: np.ndarray, constraint_group_ids: np.ndarray
) -> np.ndarray:
    """Merges `primary_group_ids` so that any two rows sharing a `constraint_group_ids`
    value always end up in the same final group, even if their primary group differs.

    Concretely: `primary_group_ids` is the spatial cluster assignment (Tier 2's actual
    grouping criterion); `constraint_group_ids` is the twin-record grouping (the
    hard constraint that must never be violated). If every twin pair already shares a
    spatial cluster, the output equals a relabeled `primary_group_ids`. If not — a twin
    pair straddles two clusters — those two clusters are merged into one group for the
    purposes of fold assignment, and the merge is logged so it's visible rather than
    silent.

    Returns
    -------
    np.ndarray
        Integer final group ids, same length as the inputs, guaranteed to respect the
        constraint.
    """
    if len(primary_group_ids) != len(constraint_group_ids):
        raise ValueError("primary_group_ids and constraint_group_ids must be the same length")

    n = len(primary_group_ids)
    # Relabel primary groups to a dense 0..k-1 integer range for the union-find.
    primary_relabeled, primary_uniques = pd.factorize(primary_group_ids, sort=True)
    uf = _UnionFind(len(primary_uniques))

    # For every constraint group with more than one row, union all the primary groups
    # its members belong to.
    constraint_series = pd.Series(constraint_group_ids)
    n_merges = 0
    for _, idx in constraint_series.groupby(constraint_series).groups.items():
        primary_labels_in_group = set(primary_relabeled[list(idx)])
        if len(primary_labels_in_group) > 1:
            labels = list(primary_labels_in_group)
            for other in labels[1:]:
                uf.union(labels[0], other)
                n_merges += 1

    if n_merges:
        logger.warning(
            "merge_groups_to_respect_constraint: %d twin-record group(s) straddled more "
            "than one primary (spatial cluster) group and were merged to preserve the "
            "twin-record constraint. This shrinks the effective number of independent "
            "groups available to Tier 2 slightly — expected to be rare, worth checking "
            "if n_merges is large relative to the number of primary groups.",
            n_merges,
        )

    final_root = np.array([uf.find(primary_relabeled[i]) for i in range(n)])
    final_ids, _ = pd.factorize(final_root, sort=True)
    return final_ids


# =============================================================================
# Preliminary spatial clustering (placeholder for Phase 3 Stage 7)
# =============================================================================


def compute_preliminary_spatial_clusters(
    df: pd.DataFrame,
    n_clusters: int = 8,
    coord_cols: Sequence[str] = ("latitude", "longitude"),
    random_state: int = 42,
) -> np.ndarray:
    """K-means clustering on raw coordinates — the same diagnostic approach prototyped
    in notebooks/01_eda.ipynb Section 6. This is explicitly a placeholder for Phase 3
    Stage 7's principled spatial-clustering feature (which will likely incorporate
    climate normals alongside coordinates, per the blueprint) — it exists so Tier 2 of
    this stage isn't blocked on Phase 3 being finished first.

    Callers should treat the output as provisional: prefer passing a real spatial
    cluster assignment once Phase 3 delivers one, and check `PLACEHOLDER_SPATIAL_CLUSTER_SOURCE`
    is not still in use by the time model comparisons in Phase 5 are made.
    """
    missing = [c for c in coord_cols if c not in df.columns]
    if missing:
        raise KeyError(f"compute_preliminary_spatial_clusters: missing column(s) {missing}")
    coords = df[list(coord_cols)].values
    if np.isnan(coords).any():
        raise ValueError("compute_preliminary_spatial_clusters: null coordinate values found")
    n_clusters_eff = min(n_clusters, len(df))
    if n_clusters_eff < n_clusters:
        warnings.warn(
            f"compute_preliminary_spatial_clusters: requested {n_clusters} clusters but "
            f"only {len(df)} rows available — using {n_clusters_eff} clusters instead.",
            stacklevel=2,
        )
    kmeans = KMeans(n_clusters=n_clusters_eff, random_state=random_state, n_init=10)
    return kmeans.fit_predict(coords)


# =============================================================================
# Tier 1 — Standard (stratified, twin-guarded)
# =============================================================================


def tier1_splits(
    df: pd.DataFrame,
    target_col: str,
    twin_keys: Sequence[str] = DEFAULT_TWIN_KEYS,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Tier 1 (Standard): stratified K-fold on `target_col`, additionally grouped by
    the twin-record key so no `(location, deathdate)` group is ever split across
    train/validation — even though this tier isn't testing geographic generalization,
    it shouldn't be silently inflated by twin-record leakage while checking ordinary
    predictive capability. Uses `sklearn.model_selection.StratifiedGroupKFold`, which
    stratifies by target while respecting the group constraint.

    Yields (train_idx, val_idx) pairs of *positional* integer indices into `df` — use
    `.iloc`, not `.loc`, when applying them (requires `df` to have a default
    RangeIndex; see `_assert_default_range_index`).
    """
    if target_col not in df.columns:
        raise KeyError(f"tier1_splits: target column '{target_col}' not found")
    _assert_default_range_index(df, "tier1_splits")
    if df[target_col].isna().any():
        raise ValueError(
            f"tier1_splits: target column '{target_col}' contains null value(s) — "
            "StratifiedGroupKFold cannot stratify on a null target; resolve upstream."
        )
    twin_groups = compute_twin_groups(df, keys=twin_keys)
    n_unique_groups = len(np.unique(twin_groups))
    if n_unique_groups < n_splits:
        raise ValueError(
            f"tier1_splits: only {n_unique_groups} twin-groups available, cannot make "
            f"{n_splits} splits without violating the group constraint."
        )
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    y = df[target_col].values
    X_placeholder = np.zeros(len(df))  # StratifiedGroupKFold only needs y and groups
    yield from splitter.split(X_placeholder, y, groups=twin_groups)


# =============================================================================
# Tier 2 — Geographic (repeated group K-fold)
# =============================================================================


@dataclass
class Tier2Result:
    """One fold's result from Tier 2, plus bookkeeping for how the grouping was built."""

    repeat: int
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray


def tier2_splits(
    df: pd.DataFrame,
    spatial_cluster_col: str | None = None,
    spatial_cluster_source: str = PLACEHOLDER_SPATIAL_CLUSTER_SOURCE,
    twin_keys: Sequence[str] = DEFAULT_TWIN_KEYS,
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: int = 42,
) -> Iterator[Tier2Result]:
    """Tier 2 (Geographic): repeated group K-fold, grouped by spatial cluster, guarded
    against twin-record splitting via `merge_groups_to_respect_constraint`.

    Because there are only ~39 real location groups (and correspondingly few spatial
    clusters), a single fold assignment is unstable — one unusual location/cluster can
    swing a fold's score noticeably. This is why Tier 2 is *repeated*: `n_repeats`
    independent random group-to-fold assignments are yielded, so the caller can (and
    should, per Stage 5's design) report mean/std/min/max across all
    `n_repeats * n_splits` folds rather than a single number. See
    `summarize_tier2_scores`.

    Parameters
    ----------
    spatial_cluster_col : str, optional
        Column in `df` holding a precomputed spatial cluster assignment (e.g. Phase 3
        Stage 7's principled clustering). If None, `compute_preliminary_spatial_clusters`
        is used and a warning is logged that this is a placeholder — see module docstring.
    spatial_cluster_source : str
        A label describing where the clustering came from, logged with every split so a
        placeholder-based run is never silently mixed with a Phase-3-based run.

    Notes
    -----
    Yields *positional* integer indices into `df` — use `.iloc`, not `.loc` (requires
    `df` to have a default RangeIndex; see `_assert_default_range_index`).

    Fold assignment is **row-count balanced**, not just group-count balanced: because
    spatial-cluster group sizes are highly unequal in this data (empirically, group
    sizes range from ~54 to ~1,150 rows), a naive round-robin over shuffled group ids
    can produce wildly uneven validation fold sizes (one early version of this function
    produced folds ranging from 5% to 36% of the data). Instead, groups are visited in
    a randomized (per-repeat) order and each is greedily assigned to whichever fold
    currently holds the fewest rows — a standard greedy bin-packing heuristic. This
    doesn't guarantee perfectly equal folds (a single very large group can still
    dominate one fold), but it is far more balanced than ignoring size, and the
    randomized visit order is what gives repeats genuinely different assignments.
    """
    if spatial_cluster_col is not None:
        if spatial_cluster_col not in df.columns:
            raise KeyError(f"tier2_splits: spatial_cluster_col '{spatial_cluster_col}' not found")
        spatial_clusters = df[spatial_cluster_col].values
    else:
        warnings.warn(
            "tier2_splits: no spatial_cluster_col provided — falling back to "
            f"'{PLACEHOLDER_SPATIAL_CLUSTER_SOURCE}'. This is a placeholder for Phase 3 "
            "Stage 7's principled spatial clustering; do not compare these Tier 2 "
            "results against a run using a different spatial_cluster_source.",
            stacklevel=2,
        )
        spatial_clusters = compute_preliminary_spatial_clusters(df, random_state=random_state)
        spatial_cluster_source = PLACEHOLDER_SPATIAL_CLUSTER_SOURCE

    _assert_default_range_index(df, "tier2_splits")
    twin_groups = compute_twin_groups(df, keys=twin_keys)
    final_groups = merge_groups_to_respect_constraint(spatial_clusters, twin_groups)
    n_unique_groups = len(np.unique(final_groups))

    logger.info(
        "tier2_splits: spatial_cluster_source=%s, %d unique groups after twin-merge, "
        "n_splits=%d, n_repeats=%d",
        spatial_cluster_source,
        n_unique_groups,
        n_splits,
        n_repeats,
    )
    if n_unique_groups < n_splits:
        raise ValueError(
            f"tier2_splits: only {n_unique_groups} groups available after merging spatial "
            f"clusters with twin-record constraints, cannot make {n_splits} splits."
        )

    rng = np.random.RandomState(random_state)
    unique_groups = np.unique(final_groups)
    # Row count carried by each final group, needed for the row-count-balanced
    # assignment below (group *count* per fold is not what matters here — group *size*
    # varies a great deal, so a naive round-robin over group ids alone would produce
    # very uneven validation fold sizes; see the docstring "Notes" above).
    group_row_counts = pd.Series(final_groups).value_counts().to_dict()

    # No bin-packing algorithm can balance folds better than the single largest group
    # allows: if one group alone already exceeds a fold's fair share, whichever fold it
    # lands in is structurally oversized regardless of how the rest are packed. Surface
    # this loudly rather than let a caller wrongly conclude the *algorithm* produced an
    # imbalanced split.
    target_fold_size = len(df) / n_splits
    largest_group_id, largest_group_n = max(group_row_counts.items(), key=lambda kv: kv[1])
    if largest_group_n > 1.5 * target_fold_size:
        logger.warning(
            "tier2_splits: the largest group (id=%s) has %d rows — %.1f%% of the data — "
            "which alone exceeds 1.5x the target fold size (%.0f rows for n_splits=%d). "
            "No fold assignment can avoid at least one oversized fold while this group "
            "stays intact; only %d unique groups exist in total, which is coarse for a "
            "%d-way split. This is a structural limitation of the current spatial "
            "grouping (particularly acute for the placeholder k=8 clustering — see "
            "module docstring), not a bug in the fold-assignment algorithm.",
            largest_group_id,
            largest_group_n,
            100.0 * largest_group_n / len(df),
            target_fold_size,
            n_splits,
            n_unique_groups,
            n_splits,
        )

    for repeat in range(n_repeats):
        shuffled_groups = unique_groups.copy()
        rng.shuffle(shuffled_groups)
        # Greedy row-count-balanced bin packing: visit groups in a randomized (repeat-
        # specific) order and assign each one to whichever fold currently holds the
        # fewest accumulated rows. This is a lightweight, transparent stand-in for
        # sklearn's GroupKFold that both supports true repeated randomization (plain
        # GroupKFold has no shuffle/random_state guarantee across scikit-learn versions
        # old enough to matter here) and keeps fold sizes close to n/n_splits despite
        # highly unequal group sizes.
        fold_row_totals = np.zeros(n_splits, dtype=np.int64)
        fold_of_group: dict = {}
        for g in shuffled_groups:
            target_fold = int(np.argmin(fold_row_totals))
            fold_of_group[g] = target_fold
            fold_row_totals[target_fold] += group_row_counts[g]
        fold_assignment = np.array([fold_of_group[g] for g in final_groups])

        for fold in range(n_splits):
            val_mask = fold_assignment == fold
            train_idx = np.where(~val_mask)[0]
            val_idx = np.where(val_mask)[0]
            if len(val_idx) == 0 or len(train_idx) == 0:
                logger.warning(
                    "tier2_splits: repeat %d fold %d produced an empty train or "
                    "validation split — skipping this fold.",
                    repeat,
                    fold,
                )
                continue
            yield Tier2Result(repeat=repeat, fold=fold, train_idx=train_idx, val_idx=val_idx)


def summarize_tier2_scores(scores: Sequence[float]) -> dict[str, float]:
    """Mean/std/min/max across all Tier 2 repeat*fold scores — Stage 5's explicit rule
    is to report spread, not just a mean, given how few independent geographic groups
    exist (see module docstring)."""
    scores_arr = np.asarray(list(scores), dtype=float)
    if len(scores_arr) == 0:
        raise ValueError("summarize_tier2_scores: no scores provided")
    return {
        "mean": float(np.mean(scores_arr)),
        "std": float(np.std(scores_arr, ddof=1)) if len(scores_arr) > 1 else 0.0,
        "min": float(np.min(scores_arr)),
        "max": float(np.max(scores_arr)),
        "n": int(len(scores_arr)),
    }


# =============================================================================
# Tier 3 — Test-like holdout
# =============================================================================


@dataclass
class Tier3Holdout:
    """The result of `make_tier3_holdout`: a single held-out split plus a transparent
    report of how well it actually matched its targets, so a poor match is visible
    rather than silently assumed."""

    train_idx: np.ndarray
    holdout_idx: np.ndarray
    holdout_locations: set
    achieved_holdout_fraction: float
    target_holdout_fraction: float
    achieved_rural_fraction: float
    target_rural_fraction: float
    n_trials_run: int


def make_tier3_holdout(
    df: pd.DataFrame,
    location_col: str = "location",
    zone_col: str = "zone",
    target_holdout_fraction: float = STAGE4_TARGET_HOLDOUT_FRACTION,
    target_rural_fraction: float = STAGE4_TARGET_RURAL_FRACTION,
    n_trials: int = 3000,
    random_state: int = 42,
) -> Tier3Holdout:
    """Tier 3 (Test-like holdout): a single held-out split built to resemble the real
    train/test structure — entire locations held out (mimicking the confirmed 0%
    train/test coordinate overlap), with the resulting holdout's zone composition
    matched, within what a randomized search over location subsets can achieve, to
    Stage 4's confirmed real test-set composition (76.21% Rural — a 13.5-percentage-point
    shift from train's 62.68%).

    This does not exhaustively optimize over all 2^39 location subsets — a randomized
    search (`n_trials` random candidate subsets, keeping the best-matching one) is used
    instead, which is transparent, fast, and sufficient given the goal is "resembles the
    real test set," not "is a provably optimal match."

    Raises
    ------
    ValueError
        If any location has an inconsistent zone value across its records (would
        indicate `zone` is not actually a location-level attribute, contradicting an
        assumption this function relies on) — surfaced loudly rather than silently
        averaged over.

    Notes
    -----
    `Tier3Holdout.train_idx` / `.holdout_idx` are *positional* integer indices into
    `df` — use `.iloc`, not `.loc` (requires `df` to have a default RangeIndex; see
    `_assert_default_range_index`).
    """
    for col in (location_col, zone_col):
        if col not in df.columns:
            raise KeyError(f"make_tier3_holdout: missing column '{col}'")
    _assert_default_range_index(df, "make_tier3_holdout")
    null_cols = [c for c in (location_col, zone_col) if df[c].isna().any()]
    if null_cols:
        raise ValueError(
            f"make_tier3_holdout: null values found in column(s) {null_cols} — a null "
            "location or zone would silently corrupt the location-level grouping this "
            "function relies on."
        )

    location_zone = df.groupby(location_col)[zone_col].nunique()
    inconsistent = location_zone[location_zone > 1]
    if len(inconsistent):
        raise ValueError(
            f"make_tier3_holdout: {len(inconsistent)} location(s) have more than one "
            f"distinct zone value, e.g. {inconsistent.index[0]!r} — zone is assumed to be "
            "a location-level attribute; this assumption doesn't hold for this data."
        )

    location_stats = df.groupby(location_col).agg(
        n_rows=(zone_col, "size"), zone=(zone_col, "first")
    )
    locations = location_stats.index.to_numpy()
    n_rows_per_location = location_stats["n_rows"].to_numpy()
    is_rural_per_location = (location_stats["zone"] == "Rural").to_numpy()
    total_rows = len(df)

    rng = np.random.RandomState(random_state)
    best_score = np.inf
    best_holdout_locations: set = set()
    best_holdout_frac = 0.0
    best_rural_frac = 0.0

    for _ in range(n_trials):
        order = rng.permutation(len(locations))
        cumulative_rows = 0
        chosen_mask = np.zeros(len(locations), dtype=bool)
        for idx in order:
            if cumulative_rows / total_rows >= target_holdout_fraction:
                break
            chosen_mask[idx] = True
            cumulative_rows += n_rows_per_location[idx]

        if cumulative_rows == 0:
            continue

        achieved_holdout_frac = cumulative_rows / total_rows
        achieved_rural_frac = (
            n_rows_per_location[chosen_mask & is_rural_per_location].sum() / cumulative_rows
        )
        score = abs(achieved_holdout_frac - target_holdout_fraction) + abs(
            achieved_rural_frac - target_rural_fraction
        )
        if score < best_score:
            best_score = score
            best_holdout_locations = set(locations[chosen_mask])
            best_holdout_frac = achieved_holdout_frac
            best_rural_frac = achieved_rural_frac

    if not best_holdout_locations:
        raise RuntimeError(
            "make_tier3_holdout: no valid holdout subset found in "
            f"{n_trials} trials — try increasing n_trials or relaxing targets."
        )

    holdout_mask = df[location_col].isin(best_holdout_locations).to_numpy()
    holdout_idx = np.where(holdout_mask)[0]
    train_idx = np.where(~holdout_mask)[0]

    logger.info(
        "make_tier3_holdout: achieved holdout_fraction=%.4f (target %.4f), "
        "rural_fraction=%.4f (target %.4f), %d locations held out, %d trials",
        best_holdout_frac,
        target_holdout_fraction,
        best_rural_frac,
        target_rural_fraction,
        len(best_holdout_locations),
        n_trials,
    )

    return Tier3Holdout(
        train_idx=train_idx,
        holdout_idx=holdout_idx,
        holdout_locations=best_holdout_locations,
        achieved_holdout_fraction=best_holdout_frac,
        target_holdout_fraction=target_holdout_fraction,
        achieved_rural_fraction=best_rural_frac,
        target_rural_fraction=target_rural_fraction,
        n_trials_run=n_trials,
    )


# =============================================================================
# High-level orchestrator — runs all three tiers and returns one tabulated result
# =============================================================================


@dataclass
class AllTiersResult:
    tier1_scores: list[float] = field(default_factory=list)
    tier2_scores: list[float] = field(default_factory=list)
    tier2_summary: dict[str, float] = field(default_factory=dict)
    tier3_score: float | None = None
    tier3_holdout: Tier3Holdout | None = None

    def as_summary_row(self) -> dict:
        """Flattened dict suitable for appending to a model-comparison table — the
        format every Stage 10+ experiment reports, per Stage 5's deliverable."""
        t1 = np.asarray(self.tier1_scores, dtype=float)
        return {
            "tier1_mean": float(np.mean(t1)) if len(t1) else np.nan,
            "tier1_std": float(np.std(t1, ddof=1)) if len(t1) > 1 else 0.0,
            "tier2_mean": self.tier2_summary.get("mean", np.nan),
            "tier2_std": self.tier2_summary.get("std", np.nan),
            "tier2_min": self.tier2_summary.get("min", np.nan),
            "tier2_max": self.tier2_summary.get("max", np.nan),
            "tier3_score": self.tier3_score if self.tier3_score is not None else np.nan,
        }


def evaluate_all_tiers(
    make_estimator: Callable[[], object],
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    score_fn: Callable[[np.ndarray, np.ndarray], float],
    spatial_cluster_col: str | None = None,
    twin_keys: Sequence[str] = DEFAULT_TWIN_KEYS,
    tier1_n_splits: int = 5,
    tier2_n_splits: int = 5,
    tier2_n_repeats: int = 5,
    tier3_location_col: str = "location",
    tier3_zone_col: str = "zone",
    tier3_target_holdout_fraction: float = STAGE4_TARGET_HOLDOUT_FRACTION,
    tier3_target_rural_fraction: float = STAGE4_TARGET_RURAL_FRACTION,
    tier3_n_trials: int = 3000,
    random_state: int = 42,
    predict_proba: bool = True,
) -> AllTiersResult:
    """Runs all three validation tiers for one candidate model and returns a single
    consolidated result — the standard reporting unit for every experiment from
    Stage 10 onward, per Stage 5's deliverable ("every subsequent experiment reports
    all three tiers, tabulated together").

    `make_estimator` is a zero-argument callable returning a fresh, unfitted
    scikit-learn-compatible estimator (called once per fold, so estimators are never
    accidentally reused/refit across folds sharing state).

    `tier3_location_col` / `tier3_zone_col` are passed through to `make_tier3_holdout`
    (previously hardcoded to its defaults here) so a caller can override them without
    bypassing this orchestrator.
    """
    X = df[list(feature_cols)].values
    y = df[target_col].values

    def _fit_predict_score(train_idx: np.ndarray, val_idx: np.ndarray) -> float:
        est = make_estimator()
        est.fit(X[train_idx], y[train_idx])
        if predict_proba:
            preds = est.predict_proba(X[val_idx])[:, 1]
        else:
            preds = est.predict(X[val_idx])
        return score_fn(y[val_idx], preds)

    result = AllTiersResult()

    for train_idx, val_idx in tier1_splits(
        df,
        target_col=target_col,
        twin_keys=twin_keys,
        n_splits=tier1_n_splits,
        random_state=random_state,
    ):
        result.tier1_scores.append(_fit_predict_score(train_idx, val_idx))

    for split in tier2_splits(
        df,
        spatial_cluster_col=spatial_cluster_col,
        twin_keys=twin_keys,
        n_splits=tier2_n_splits,
        n_repeats=tier2_n_repeats,
        random_state=random_state,
    ):
        result.tier2_scores.append(_fit_predict_score(split.train_idx, split.val_idx))
    result.tier2_summary = summarize_tier2_scores(result.tier2_scores)

    holdout = make_tier3_holdout(
        df,
        location_col=tier3_location_col,
        zone_col=tier3_zone_col,
        target_holdout_fraction=tier3_target_holdout_fraction,
        target_rural_fraction=tier3_target_rural_fraction,
        n_trials=tier3_n_trials,
        random_state=random_state,
    )
    result.tier3_holdout = holdout
    result.tier3_score = _fit_predict_score(holdout.train_idx, holdout.holdout_idx)

    return result
