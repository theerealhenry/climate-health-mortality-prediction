"""
Stage 7 — Spatial features.

Why this exists, and why it looks the way it does: Stage 3's forensics and Stage 4's
adversarial validation together confirmed the central fact this module is built around
— there is **zero exact coordinate overlap** between Train (43 unique lat/lon pairs) and
Test (12 unique lat/lon pairs), and `latitude`/`longitude`/`elevation`/`slope` alone
separate Train from Test almost perfectly (single-feature AUCs of 0.9992 / 1.0000 / 0.9507
/ 0.9502; see notebooks/01_eda.ipynb §5). Every test coordinate is geographically novel
relative to training. A feature representation that can only describe *training*
locations (e.g. parsed district/region tokens from the raw `location` string — kept here
only as a documented fallback, see `location_fallback_tokens`) cannot say anything useful
about a test coordinate whose district was never seen. A representation based on
*distance in feature space* can: it assigns any coordinate, seen or not, to whichever
learned cluster it most resembles.

That is what `SpatialClusterFeaturizer` does. It clusters not on coordinates alone, but
jointly on coordinates and location-level climate normals (`tavg_30d`, `rain_sum_90d`,
`elevation` by default) — so an unseen test coordinate is placed near training locations
with *similar climate*, not just similar latitude/longitude numbers, which is the more
defensible generalization signal per docs/PROJECT_BLUEPRINT.md Phase 3 Stage 7 and the
adversarial-validation reinterpretation in §0.2 (a high train/test-distinguishability AUC
identifies *where* the distributions differ, not features to blindly drop — climate
normals and elevation may be exactly what lets a model generalize to new places, as
opposed to raw lat/lon which risks becoming a lookup table over the 43 training points).

Design choice — cluster assignment is a stable **per-coordinate** attribute, not a
per-record one: every row sharing the same (latitude, longitude) is assigned the same
`spatial_cluster`, computed from that coordinate's own mean climate profile within
whatever dataset is being transformed (fit or transform time), not from each individual
record's day-specific climate reading. This mirrors how `zone` already behaves in this
project (`make_tier3_holdout` explicitly requires zone to be a location-level constant —
see cv.py) and is what makes `spatial_cluster` a sensible thing to group by in
`tier2_splits`, interact with `zone` in Stage 9, or otherwise treat as an identity rather
than a value that drifts across a location's own record history.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)

DEFAULT_COORD_COLS: tuple[str, str] = ("latitude", "longitude")
# Per docs/PROJECT_BLUEPRINT.md Stage 7: coordinates jointly with location-level climate
# normals. `elevation` is included even though it's not literally a rolling-window
# "normal" like the other two, because Stage 4 found it among the strongest single
# train/test-distinguishing features (0.9507 AUC) and Stage 3 noted it sits on a coarse
# 23-value grid worth folding into the spatial representation rather than treated in
# isolation.
DEFAULT_CLIMATE_NORMAL_COLS: tuple[str, ...] = ("tavg_30d", "rain_sum_90d", "elevation")
# 5-8 per the blueprint's stated range. With only 43 unique training coordinates, k=8
# already means an average of ~5.4 points per cluster -- going higher risks clusters of
# size 1-2 that aren't meaningfully generalizable.
DEFAULT_K_RANGE: tuple[int, ...] = (5, 6, 7, 8)
DEFAULT_OUTPUT_COL = "spatial_cluster"


# =============================================================================
# Location-profile construction (shared by cluster selection and the featurizer)
# =============================================================================


def build_location_profiles(
    df: pd.DataFrame,
    coord_cols: Sequence[str] = DEFAULT_COORD_COLS,
    climate_normal_cols: Sequence[str] = DEFAULT_CLIMATE_NORMAL_COLS,
) -> pd.DataFrame:
    """Collapses `df` to one row per unique coordinate, averaging `climate_normal_cols`
    across every record observed at that coordinate.

    This is a deliberate design choice, not an implementation shortcut: fitting KMeans
    directly on row-level data would let a coordinate with many mortality records (some
    training locations have far more records than others) pull the cluster centroids
    toward it purely by weight of repetition, rather than each *place* counting once.
    Collapsing to one row per coordinate first treats every distinct location as a single
    point, which is what "cluster the locations" should mean.

    Raises
    ------
    KeyError
        If any of `coord_cols` / `climate_normal_cols` is missing from `df`.
    ValueError
        If any of those columns contains a null value.
    """
    required = list(coord_cols) + list(climate_normal_cols)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"build_location_profiles: missing required column(s) {missing}")
    if df[required].isna().any().any():
        bad_cols = df[required].columns[df[required].isna().any()].tolist()
        raise ValueError(
            f"build_location_profiles: null values found in column(s) {bad_cols} — "
            "cannot compute a location's climate normal from incomplete data."
        )
    profiles = (
        df.groupby(list(coord_cols), as_index=False)[list(climate_normal_cols)]
        .mean()
        .reset_index(drop=True)
    )
    return profiles


# =============================================================================
# Cluster-count selection
# =============================================================================


@dataclass
class ClusterSelectionResult:
    """One candidate `k`'s evaluation, for inspection/plotting in the research notebook."""

    k: int
    silhouette: float
    inertia: float
    n_locations: int


def select_n_clusters(
    df: pd.DataFrame,
    coord_cols: Sequence[str] = DEFAULT_COORD_COLS,
    climate_normal_cols: Sequence[str] = DEFAULT_CLIMATE_NORMAL_COLS,
    k_range: Sequence[int] = DEFAULT_K_RANGE,
    random_state: int = 42,
) -> tuple[int, list[ClusterSelectionResult]]:
    """Evaluates KMeans over `k_range` on standardized location profiles built from `df`,
    returning `(best_k, all_results)`.

    `best_k` maximizes silhouette score (ties broken toward the smaller `k` — a simpler
    clustering is preferred when two choices separate the data equally well). This is a
    *candidate* selection tool, not the final word: docs/PROJECT_BLUEPRINT.md Stage 7
    explicitly expects this choice to also be checked against Tier 2 CV behavior (does a
    finer/coarser clustering actually help geographic generalization), which requires
    fitting a model and is done in the research notebook, not here — this function only
    has access to unsupervised structure in the coordinates/climate normals themselves.

    Raises
    ------
    ValueError
        If any `k` in `k_range` is `>=` the number of unique locations in `df` (cannot
        usefully cluster more groups than there are points), or if `k_range` is empty.
    """
    if len(k_range) == 0:
        raise ValueError("select_n_clusters: k_range must not be empty")
    profiles = build_location_profiles(
        df, coord_cols=coord_cols, climate_normal_cols=climate_normal_cols
    )
    n_locations = len(profiles)
    feature_cols = list(coord_cols) + list(climate_normal_cols)

    too_large = [k for k in k_range if k >= n_locations]
    if too_large:
        raise ValueError(
            f"select_n_clusters: k value(s) {too_large} >= n_unique_locations={n_locations} "
            "in the fit data; cannot cluster meaningfully at that k."
        )

    scaler = StandardScaler()
    X = scaler.fit_transform(profiles[feature_cols].to_numpy())

    results: list[ClusterSelectionResult] = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        n_unique_labels = len(np.unique(labels))
        sil = (
            silhouette_score(X, labels)
            if n_unique_labels > 1 and n_unique_labels < len(X)
            else float("nan")
        )
        results.append(
            ClusterSelectionResult(
                k=k, silhouette=float(sil), inertia=float(km.inertia_), n_locations=n_locations
            )
        )

    valid = [r for r in results if not np.isnan(r.silhouette)]
    if not valid:
        raise ValueError(
            "select_n_clusters: no candidate k produced a valid silhouette score — "
            "check k_range against the data."
        )
    best = max(valid, key=lambda r: (r.silhouette, -r.k))
    logger.info(
        "select_n_clusters: best_k=%d (silhouette=%.4f) among %s, n_locations=%d",
        best.k,
        best.silhouette,
        list(k_range),
        n_locations,
    )
    return best.k, results


# =============================================================================
# SpatialClusterFeaturizer
# =============================================================================


class SpatialClusterFeaturizer(BaseEstimator, TransformerMixin):
    """Assigns each record a `spatial_cluster` label, learned jointly from geographic
    coordinates and location-level climate normals, and stable per coordinate (see
    module docstring).

    Parameters
    ----------
    n_clusters : int
        Number of KMeans clusters. Choose via `select_n_clusters` plus the Tier 2 CV
        check described there, not by default alone.
    coord_cols, climate_normal_cols : Sequence[str]
        Columns used to build each location's profile (see `build_location_profiles`).
    output_col : str
        Name of the cluster-label column added by `transform`.
    random_state : int
        Passed to KMeans for reproducibility.

    Usage
    -----
    `fit(X)` learns cluster centroids from `X`'s unique-coordinate climate profiles (e.g.
    the 43 training coordinates). `transform(X)` — on the *same or different* data —
    builds `X`'s own unique-coordinate profiles independently (so a test set's climate
    profile is its own, never borrowed from train) and assigns each to its nearest
    training-fitted centroid, including coordinates never seen during `fit`. Every row
    sharing a coordinate gets the same label.

    Raises
    ------
    KeyError
        If a required column is missing from `X` at `fit` or `transform`.
    ValueError
        If `X` contains null values in the required columns, or if `n_clusters` is `>=`
        the number of unique coordinates available to `fit`.
    sklearn.exceptions.NotFittedError
        If `transform` is called before `fit`.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        coord_cols: Sequence[str] = DEFAULT_COORD_COLS,
        climate_normal_cols: Sequence[str] = DEFAULT_CLIMATE_NORMAL_COLS,
        output_col: str = DEFAULT_OUTPUT_COL,
        random_state: int = 42,
    ):
        # sklearn convention: __init__ must store every parameter exactly as received,
        # with no validation or type coercion — get_params()/clone() round-trip these
        # values verbatim, and casting here (e.g. list -> tuple) would make
        # clone(estimator).get_params() disagree with the original constructor call.
        # Sequences are normalized to tuples only where actually used (_feature_cols).
        self.n_clusters = n_clusters
        self.coord_cols = coord_cols
        self.climate_normal_cols = climate_normal_cols
        self.output_col = output_col
        self.random_state = random_state

    def _feature_cols(self) -> list[str]:
        return list(self.coord_cols) + list(self.climate_normal_cols)

    def fit(self, X: pd.DataFrame, y=None) -> SpatialClusterFeaturizer:
        profiles = build_location_profiles(
            X, coord_cols=self.coord_cols, climate_normal_cols=self.climate_normal_cols
        )
        n_locations = len(profiles)
        if self.n_clusters >= n_locations:
            raise ValueError(
                f"SpatialClusterFeaturizer.fit: n_clusters={self.n_clusters} >= "
                f"n_unique_locations={n_locations} in fit data; cannot cluster meaningfully. "
                "Use select_n_clusters to pick a k appropriate for this data's location count."
            )

        feature_cols = self._feature_cols()
        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(profiles[feature_cols].to_numpy())
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.kmeans_.fit(Xs)

        # Kept for inspection/notebook use (cluster interpretation, `summarize_clusters`)
        # and for a defensive check in `transform`, not required for the transform itself.
        fit_labels = self.kmeans_.predict(Xs)
        self.location_profiles_ = profiles.assign(**{self.output_col: fit_labels})
        self.n_locations_fit_ = n_locations
        self.feature_names_in_ = feature_cols
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["kmeans_", "scaler_"])
        if self.output_col in X.columns:
            raise ValueError(
                f"SpatialClusterFeaturizer.transform: X already has a column named "
                f"'{self.output_col}' — refusing to merge, since pandas would silently "
                f"rename both to '{self.output_col}_x'/'{self.output_col}_y' instead of "
                "raising, which would otherwise corrupt the output without any warning. "
                "Either drop/rename that column first, or use a different output_col."
            )
        profiles = build_location_profiles(
            X, coord_cols=self.coord_cols, climate_normal_cols=self.climate_normal_cols
        )
        feature_cols = self._feature_cols()
        Xs = self.scaler_.transform(profiles[feature_cols].to_numpy())
        profiles = profiles.assign(**{self.output_col: self.kmeans_.predict(Xs)})

        n_before = len(X)
        out = X.merge(
            profiles[list(self.coord_cols) + [self.output_col]],
            on=list(self.coord_cols),
            how="left",
        )
        if len(out) != n_before:
            raise AssertionError(
                "SpatialClusterFeaturizer.transform: row count changed after merging cluster "
                f"labels back on ({n_before} -> {len(out)}) — this should be impossible for a "
                "left merge on a deduplicated profile table; investigate duplicate coordinates "
                "with inconsistent types (e.g. float precision drift)."
            )
        # pandas documents that how="left" preserves the left frame's key order, which is
        # what lets `out.index = X.index` below be a safe direct reassignment rather than a
        # positional guess — but that's still an assumption about merge internals, not
        # something the row-count check above actually verifies. Confirm real row-for-row
        # alignment directly (coordinates must match X's own, position for position) before
        # trusting the reassignment; if this ever fires, it means the order-preservation
        # assumption broke (e.g. a future pandas behavior change), not that this specific
        # coordinate/cluster mapping is wrong.
        coord_cols = list(self.coord_cols)
        if not np.array_equal(out[coord_cols].to_numpy(), X[coord_cols].to_numpy()):
            raise AssertionError(
                "SpatialClusterFeaturizer.transform: merged row order does not match X's "
                "original row order — pandas' left-merge key-order guarantee did not hold "
                "as expected. Do not trust `out.index = X.index` here; investigate before "
                "using this transformer's output."
            )
        out.index = X.index
        return out

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        if fit_params:
            raise TypeError(
                f"SpatialClusterFeaturizer.fit_transform: received unexpected fit_params "
                f"{list(fit_params)} — fit() takes no extra keyword arguments, and silently "
                "ignoring them would hide a caller's mistake (e.g. a misspelled sample_weight "
                "or a param meant for a different pipeline step)."
            )
        return self.fit(X, y).transform(X)

    def cluster_centers_original_scale(self) -> pd.DataFrame:
        """Cluster centroids inverse-transformed back to original units (degrees, °C, mm,
        meters) — for human-readable cluster interpretation in the research notebook."""
        check_is_fitted(self, ["kmeans_", "scaler_"])
        centers = self.scaler_.inverse_transform(self.kmeans_.cluster_centers_)
        return pd.DataFrame(centers, columns=self._feature_cols())


def summarize_clusters(
    featurizer: SpatialClusterFeaturizer,
    df: pd.DataFrame,
    zone_col: str = "zone",
) -> pd.DataFrame:
    """Per-cluster summary for the research notebook: record count, unique-location
    count, dominant zone, and mean of every clustering feature — the "what separates the
    clusters" table docs/PROJECT_BLUEPRINT.md Stage 7 asks the notebook to document.

    `featurizer` must already be fitted. `df` is any dataset to summarize over (typically
    the training data used to fit it, but can be test or combined data too).
    """
    check_is_fitted(featurizer, ["kmeans_", "scaler_"])
    labeled = featurizer.transform(df)
    summary = labeled.groupby(featurizer.output_col).agg(
        n_records=(featurizer.coord_cols[0], "size"),
        **{f"{c}_mean": (c, "mean") for c in featurizer._feature_cols()},
    )
    n_unique_coords = (
        labeled.drop_duplicates(subset=list(featurizer.coord_cols))
        .groupby(featurizer.output_col)
        .size()
        .rename("n_unique_locations")
    )
    summary = summary.join(n_unique_coords)
    if zone_col in labeled.columns:
        dominant_zone = (
            labeled.groupby(featurizer.output_col)[zone_col]
            .agg(lambda s: s.value_counts().idxmax())
            .rename("dominant_zone")
        )
        summary = summary.join(dominant_zone)
    return summary.reset_index()


# =============================================================================
# Raw coordinate transforms
# =============================================================================


def add_coordinate_polynomial_features(
    df: pd.DataFrame, coord_cols: Sequence[str] = DEFAULT_COORD_COLS
) -> pd.DataFrame:
    """Adds lat², lon², and lat×lon alongside the raw coordinates — per
    docs/PROJECT_BLUEPRINT.md Stage 7: "Raw lat/lon and simple transforms ... are
    included directly as features alongside the cluster assignment." These let a linear
    or shallow model pick up nonlinear geographic effects the cluster label alone might
    not capture, at the cost of exactly the risk the module docstring already flags
    (Stage 4's near-perfect adversarial AUC on raw coordinates) — kept here, judged by
    Tier 2 CV like every other feature in this phase, not assumed safe.
    """
    lat_col, lon_col = coord_cols
    missing = [c for c in coord_cols if c not in df.columns]
    if missing:
        raise KeyError(f"add_coordinate_polynomial_features: missing column(s) {missing}")
    if df[list(coord_cols)].isna().any().any():
        bad_cols = [c for c in coord_cols if df[c].isna().any()]
        raise ValueError(
            f"add_coordinate_polynomial_features: null values found in column(s) {bad_cols} — "
            "would silently propagate as NaN into every derived feature; fix upstream instead."
        )
    out = df.copy()
    out[f"{lat_col}_sq"] = out[lat_col] ** 2
    out[f"{lon_col}_sq"] = out[lon_col] ** 2
    out[f"{lat_col}_x_{lon_col}"] = out[lat_col] * out[lon_col]
    return out


# =============================================================================
# Location-string fallback (secondary, brittle — see module docstring)
# =============================================================================


def location_fallback_tokens(
    df: pd.DataFrame, location_col: str = "location", max_tokens: int = 3
) -> pd.DataFrame:
    """Splits the raw `location` string on commas into up to `max_tokens` trailing
    tokens (e.g. "Nakigo I, Iganga, Uganda" -> district="Iganga", country="Uganda"),
    kept only as a documented **fallback/secondary** feature per
    docs/PROJECT_BLUEPRINT.md Stage 7 — NOT the primary geographic representation.

    This is brittle by construction: Stage 3/4 found `location` strings vary from 3 to 8
    comma-separated tokens, so token position doesn't reliably mean the same
    administrative level across rows, and — more fundamentally — a test-set district
    string that never appears in training is just as unseen to this representation as
    the coordinate itself, without `SpatialClusterFeaturizer`'s distance-based fallback.
    Tokens are taken from the *end* of the string (assumed most-general -> e.g. country,
    region) since trailing tokens are more likely to repeat across locations than
    leading ones (the specific village name).

    Missing tokens (a location string with fewer than `max_tokens` comma-separated
    parts) are filled with `None`, not dropped, so this always returns one row per input
    row with a consistent set of columns.
    """
    if location_col not in df.columns:
        raise KeyError(f"location_fallback_tokens: missing column '{location_col}'")
    if df[location_col].isna().any():
        n_null = int(df[location_col].isna().sum())
        raise ValueError(
            f"location_fallback_tokens: {n_null} null value(s) in '{location_col}' — "
            "a null location string cannot be split into tokens."
        )
    out = df.copy()
    split = out[location_col].str.split(",").apply(lambda parts: [p.strip() for p in parts])
    for i in range(max_tokens):
        # i=0 -> last token, i=1 -> second-to-last, etc.
        out[f"{location_col}_token_{i}"] = split.apply(
            lambda parts, i=i: parts[-(i + 1)] if len(parts) > i else None
        )
    return out
