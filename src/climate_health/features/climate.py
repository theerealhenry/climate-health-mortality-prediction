"""
Stage 8 — Climate feature engineering & enrichment.

Treated as its own research track, not a routine feature-engineering afterthought,
because Stage 3's forensics already found the provided climate columns carry more
target signal than the demographic ones alone (rolling-window aggregates like
`rain_sum_90d`/`tavg_90d`/`ndvi_30d`/`ndvi_90d` meaningfully outperform same-day
readings), and getting the exposure-window framing right — acute vs. chronic, ratio
vs. raw level, anomaly vs. absolute value — is where a meaningful share of the model's
real lift is likely to come from.

Four tracks, per docs/PROJECT_BLUEPRINT.md Stage 8:

  A — Provided data, made right: drop the confirmed-dead `hot_days_30d` column;
      normalized/per-day and ratio features from the existing rainfall windows.
  B — Temperature: anomalies relative to a *learned* (train-only) climate baseline;
      a heat-exceedance feature that is honest about what can and can't be
      reconstructed from the provided aggregates (see below).
  C — Rainfall: the same anomaly machinery applied to rainfall; an explicit,
      documented limitation on what a "dry-spell" feature can mean here.
  D — NDVI/vegetation (stretch): a short-term vegetation trend feature.

Two honesty notes worth stating up front, because both correct a naive first
instinct that would have reproduced a mistake this project has already found once:

1. **`hot_days_30d` cannot be meaningfully "recomputed" from `tmax_30d`/`tmin_30d`.**
   Those columns are each a single scalar per 30-day window (the window's max and
   min), not a daily time series — there is no way to count how many *individual
   days* exceeded a threshold from two window-level scalars. Worse, in this specific
   dataset `tmax_30d` never exceeds 34.97°C (verified against the real data), so a
   literal "count of days >35°C" feature recomputed at the traditional WHO-style
   35°C threshold would *also* be constant zero — reproducing `hot_days_30d`'s exact
   problem under a different name. `compute_heat_threshold`/`add_heat_exceedance_features`
   below use a data-driven, train-only threshold (a percentile of `tmax_30d`, not a
   fixed absolute degree value) specifically to avoid this trap, and add a continuous
   margin feature alongside the binary flag so the feature isn't only informative
   near one arbitrary cutoff.
2. **A genuine "dry-spell" (consecutive dry days) feature is not computable from the
   provided aggregates either.** `rain_days_30d` is a *count* of rainy days in the
   window, not their positions — two windows with the same count can have very
   different spell structure (13 scattered rainy days vs. 13 consecutive ones look
   identical here). Track C does not fabricate a consecutive-dry-spell feature; the
   provenance table below records this as a known limitation rather than silently
   shipping a feature that looks like it measures something it doesn't.

Leakage-safety design, consistent with Stage 7's `SpatialClusterFeaturizer`: any
feature requiring a *learned baseline* (an anomaly, a percentile threshold) is built
as fit-on-train-only, then applied to any dataset (train or test) via lookup —
`ClimateAnomalyFeaturizer` and `compute_heat_threshold` follow this pattern. A version
that recomputed a "baseline" independently from whatever dataset was passed in would
be methodologically wrong even without leaking the target: at real prediction time,
there is no batch of other test records to average over for a single new record, only
whatever was learned in advance from training data.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from climate_health.utils.validation import (
    check_no_output_collision as _check_no_output_collision,
)
from climate_health.utils.validation import validate_no_nulls as _validate_no_nulls
from climate_health.utils.validation import (
    validate_required_columns as _validate_columns,
)

logger = logging.getLogger(__name__)

# Stage 3 forensics: confirmed constant (all zero) across all 4,176 rows (train+test).
# `verify_constant=True` in drop_dead_columns re-checks this against whatever data is
# actually passed, rather than trusting this list blindly forever (see schemas.py's own
# comment: a future data refresh could resurrect real variance in this column).
DEAD_COLUMNS: tuple[str, ...] = ("hot_days_30d",)

DEFAULT_HEAT_QUANTILE = 0.90
DEFAULT_MIN_GROUP_MONTH_SIZE = 5


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Elementwise `numerator / denominator`, returning NaN (not inf or 0) wherever the
    denominator is zero — a genuinely undefined ratio, not a fabricated value. NaN is
    handled natively by every model in this project's zoo (LightGBM/XGBoost/CatBoost);
    a linear/logistic path (Stage 11's dual-encoder branch) will need explicit
    imputation, deferred to that stage rather than guessed at here."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    return result.mask(denominator == 0, other=np.nan)


# =============================================================================
# Track A — provided data, made right
# =============================================================================


def drop_dead_columns(
    df: pd.DataFrame, cols: Sequence[str] = DEAD_COLUMNS, verify_constant: bool = True
) -> pd.DataFrame:
    """Drops `cols` (default: `hot_days_30d`) after re-verifying each is actually
    constant in `df` — not a blind drop based on a historical finding that could have
    gone stale.

    Raises
    ------
    KeyError
        If a column in `cols` is missing from `df`.
    ValueError
        If `verify_constant=True` and a column in `cols` is *not* constant in `df` —
        this means the historical finding no longer holds (e.g. a data refresh), and
        dropping it would silently discard now-informative data. Investigate before
        overriding with `verify_constant=False`.
    """
    _validate_columns(df, cols, "drop_dead_columns")
    if verify_constant:
        not_constant = [c for c in cols if df[c].nunique(dropna=False) > 1]
        if not_constant:
            raise ValueError(
                f"drop_dead_columns: column(s) {not_constant} are NOT constant in this "
                "data — the historical 'confirmed dead' finding no longer holds here. "
                "Investigate before dropping (pass verify_constant=False to override "
                "once you've confirmed dropping is still the right call)."
            )
    return df.drop(columns=list(cols))


def add_rainfall_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Track A: normalized/per-day and ratio features from the provided rainfall
    windows — deliberately not the arithmetically-wrong `rain_sum_30d - rain_sum_7d*4`
    v1 had (7×4=28≠30), and not an arbitrary interpretation either.

    Adds:
      - `rain_rate_7d` / `_30d` / `_90d`: mm per day over each window (comparable
        across windows of different length, unlike the raw sums).
      - `rain_ratio_acute_medium` = rain_sum_7d / rain_sum_30d: is the most recent
        week wetter or drier than the last month overall (>1 = recent uptick).
      - `rain_ratio_medium_chronic` = rain_sum_30d / rain_sum_90d: same idea, medium
        vs. chronic window.
      - `rain_intensity_30d` = max_daily_rain_30d / rain_sum_30d: what fraction of
        the whole month's rain fell in a single day (concentrated downpour vs.
        spread-out rain), bounded in [0, 1] whenever defined.
      - `rain_day_fraction_30d` = rain_days_30d / 30: fraction of days in the window
        with measurable rain.

    All ratio features use `_safe_ratio` — NaN, not inf/0, when a denominator is zero.
    Empirically: `rain_sum_30d` (the denominator of `rain_ratio_acute_medium` and
    `rain_intensity_30d`) is exactly zero for 4/3,146 training rows, and `rain_sum_90d`
    (the denominator of `rain_ratio_medium_chronic`) is never zero in the observed
    data — both real, not edge cases to special-case away. `rain_sum_7d` is zero for
    345/3,146 training rows, but it is never used as a denominator here (only as
    `rain_ratio_acute_medium`'s numerator, where a zero value is a perfectly well
    defined 0.0 ratio, not an undefined one) — noted so a reader doesn't assume that
    figure implies a third NaN-prone feature that doesn't actually exist.
    """
    required = [
        "rain_sum_7d",
        "rain_sum_30d",
        "rain_sum_90d",
        "rain_days_30d",
        "max_daily_rain_30d",
    ]
    _validate_columns(df, required, "add_rainfall_rate_features")
    _validate_no_nulls(df, required, "add_rainfall_rate_features")
    new_cols = [
        "rain_rate_7d",
        "rain_rate_30d",
        "rain_rate_90d",
        "rain_ratio_acute_medium",
        "rain_ratio_medium_chronic",
        "rain_intensity_30d",
        "rain_day_fraction_30d",
    ]
    _check_no_output_collision(df, new_cols, "add_rainfall_rate_features")

    out = df.copy()
    out["rain_rate_7d"] = out["rain_sum_7d"] / 7
    out["rain_rate_30d"] = out["rain_sum_30d"] / 30
    out["rain_rate_90d"] = out["rain_sum_90d"] / 90
    out["rain_ratio_acute_medium"] = _safe_ratio(out["rain_sum_7d"], out["rain_sum_30d"])
    out["rain_ratio_medium_chronic"] = _safe_ratio(out["rain_sum_30d"], out["rain_sum_90d"])
    out["rain_intensity_30d"] = _safe_ratio(out["max_daily_rain_30d"], out["rain_sum_30d"])
    out["rain_day_fraction_30d"] = out["rain_days_30d"] / 30
    return out


# =============================================================================
# Track B — temperature
# =============================================================================


def compute_heat_threshold(
    train_df: pd.DataFrame, quantile: float = DEFAULT_HEAT_QUANTILE, col: str = "tmax_30d"
) -> float:
    """Computes a data-driven heat-exceedance threshold as a percentile of `col` in
    `train_df` **only** — never on test or combined data, to avoid leaking test's
    distribution into a feature-engineering decision.

    Deliberately not a fixed absolute value (e.g. the traditional 35°C extreme-heat
    convention): verified against this dataset, `tmax_30d` never exceeds 34.97°C, so a
    fixed 35°C threshold would produce a constant-zero flag — exactly reproducing
    `hot_days_30d`'s original problem. The default 90th-percentile threshold (~30.7°C
    on the real training data) instead marks genuinely elevated windows *within this
    dataset's own range*, with a train flag rate of ~9.9% and a comparable test flag
    rate of ~11.9% — similar enough to suggest the threshold generalizes reasonably,
    not so similar that it's uninformative.

    Call this once on training data; reuse the returned value for both
    `add_heat_exceedance_features(train_df, threshold=...)` and
    `add_heat_exceedance_features(test_df, threshold=...)` — the same pattern as
    Stage 7's `select_n_clusters` -> chosen `k` -> `SpatialClusterFeaturizer(n_clusters=k)`.
    """
    _validate_columns(train_df, [col], "compute_heat_threshold")
    _validate_no_nulls(train_df, [col], "compute_heat_threshold")
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"compute_heat_threshold: quantile must be in (0, 1), got {quantile}")
    return float(train_df[col].quantile(quantile))


def add_heat_exceedance_features(
    df: pd.DataFrame, threshold: float, col: str = "tmax_30d"
) -> pd.DataFrame:
    """Track B: a binary flag and a continuous margin for whether/how far `col`
    exceeds `threshold` (from `compute_heat_threshold`, fit on train only).

    Adds `{col}_exceeds_flag` (0/1) and `{col}_exceedance_margin` (°C above threshold;
    negative when below it, so this is informative on both sides of the cutoff, not
    just at the single binary boundary the flag alone would give).
    """
    _validate_columns(df, [col], "add_heat_exceedance_features")
    _validate_no_nulls(df, [col], "add_heat_exceedance_features")
    if not np.isfinite(threshold):
        raise ValueError(
            f"add_heat_exceedance_features: threshold must be a finite number, got {threshold!r} "
            "— did you forget to call compute_heat_threshold() first?"
        )
    new_cols = [f"{col}_exceeds_flag", f"{col}_exceedance_margin"]
    _check_no_output_collision(df, new_cols, "add_heat_exceedance_features")

    out = df.copy()
    out[f"{col}_exceeds_flag"] = (out[col] > threshold).astype(int)
    out[f"{col}_exceedance_margin"] = out[col] - threshold
    return out


# =============================================================================
# Track B/C shared — learned climate anomalies
# =============================================================================


class ClimateAnomalyFeaturizer(BaseEstimator, TransformerMixin):
    """Adds `{col}_anomaly` = value minus a *learned, train-only* baseline, for each
    of `value_cols` — the shared machinery behind Track B's temperature anomalies and
    Track C's rainfall anomalies.

    Baseline resolution, per row, in priority order (mirrors how a real deployment
    would have to work — nothing here is computed from the row's own dataset):
      1. That row's `(group_col, month)` mean, if `fit()` saw at least
         `min_group_month_size` training rows in that exact bucket — the most
         specific, seasonally-aware baseline ("was this unusual for this place at
         this time of year").
      2. Otherwise, that row's `group_col` mean (ignoring month) — used when the
         seasonal bucket is too sparse or entirely unseen in training (empirically,
         10 of 63 non-empty (cluster, month) buckets in the real training data have
         fewer than 5 records, and 9 of the 72 possible combinations never occur at
         all — sparsity here is real, not a hypothetical edge case).
      3. Otherwise (a `group_col` value never seen in training at all), the global
         training mean of that column.

    `{col}_anomaly_baseline_level` records which tier was used per row (values
    `"group_month"`, `"group"`, `"global"`) — one diagnostic column shared across all
    `value_cols`, since the tier selected depends only on the row's group/month
    bucket, not which value column is being computed.

    **`fit_transform(X)` is deliberately NOT equivalent to `fit(X).transform(X)`,
    unlike the sklearn default.** `transform()` always applies the plain (in-sample)
    bucket means learned in `fit()` — correct and leakage-free for genuinely new data
    (test/production), since none of those rows contributed to the bucket it looks
    up. But calling `transform()` on the *same* data `fit()` was just trained on would
    let each row's own value contribute to its own baseline — for a bucket of `n`
    rows, this shrinks that row's computed anomaly toward zero by a factor of exactly
    `(n-1)/n` relative to the honest value (verified empirically: for this project's
    real k=6 spatial clustering, cluster 3 has only 3 training records, and the
    self-inclusion shrinkage is a full 33%). This does not leak the *target*
    (`is_climate_sensitive`) — it's a bias in a covariate summary statistic, the same
    category as any `fit_transform` that reuses training statistics on training rows
    (e.g. `StandardScaler`) — but it disproportionately weakens the very feature this
    class exists to produce for exactly the small, sparse groups where a real anomaly
    signal matters most. `fit_transform()` therefore computes true leave-one-out
    baselines for the rows it was fit on (each row's own value is excluded from its
    own bucket statistic, falling back to a coarser tier if removing it would leave
    the bucket empty), following the same convention as `category_encoders`'
    `LeaveOneOutEncoder`. `transform()` on a *different* dataframe is unaffected and
    unchanged.

    Month is derived internally from `deathdate_col` purely to build the seasonal
    baseline bucket — it is *not* added as an output column. Stage 6 owns the
    project's public temporal feature columns (cyclical month/day-of-year encodings,
    etc.); this is an internal grouping key only, not a competing feature.

    Parameters
    ----------
    value_cols : Sequence[str]
        Columns to compute anomalies for (e.g. `("tavg_30d", "rain_sum_90d")`).
    group_col : str
        The spatial grouping column — typically `"spatial_cluster"` from Stage 7's
        `SpatialClusterFeaturizer`, fit and applied to `df` *before* this transformer
        (this class does not fit a clusterer itself; it expects the column already
        present, so it stays independently testable and doesn't hide a KMeans fit
        inside a feature-engineering step).
    min_group_month_size : int
        Minimum training-row count in a `(group_col, month)` bucket before it's
        trusted as a baseline; below this, falls back to the group-only mean.

    Raises
    ------
    KeyError
        If a required column is missing at `fit` or `transform`.
    ValueError
        If required columns contain nulls, or if `transform`'s `X` already has an
        output column this transformer would add.
    sklearn.exceptions.NotFittedError
        If `transform` is called before `fit`.
    """

    def __init__(
        self,
        value_cols: Sequence[str],
        group_col: str = "spatial_cluster",
        deathdate_col: str = "deathdate",
        min_group_month_size: int = DEFAULT_MIN_GROUP_MONTH_SIZE,
        output_suffix: str = "_anomaly",
    ):
        if isinstance(value_cols, (str, bytes)):
            raise TypeError(
                "ClimateAnomalyFeaturizer: value_cols must be a sequence of column "
                f"names (e.g. a list), not a bare string — got {value_cols!r}. Passing "
                "a string here silently iterates its individual characters instead of "
                "treating it as one column name, which is almost never what's intended "
                '(wrap it: value_cols=["' + str(value_cols) + '"])'
            )
        self.value_cols = value_cols
        self.group_col = group_col
        self.deathdate_col = deathdate_col
        self.min_group_month_size = min_group_month_size
        self.output_suffix = output_suffix

    def _required_cols(self) -> list[str]:
        return list(self.value_cols) + [self.group_col, self.deathdate_col]

    def fit(self, X: pd.DataFrame, y=None) -> ClimateAnomalyFeaturizer:
        _validate_columns(X, self._required_cols(), "ClimateAnomalyFeaturizer.fit")
        _validate_no_nulls(X, self._required_cols(), "ClimateAnomalyFeaturizer.fit")

        value_cols = list(self.value_cols)
        month = X[self.deathdate_col].dt.month
        group = X[self.group_col]

        self.n_train_ = len(X)
        self.global_mean_: dict[str, float] = {c: float(X[c].mean()) for c in value_cols}
        self._global_sum_: dict[str, float] = {c: float(X[c].sum()) for c in value_cols}

        group_stats = X.groupby(group)[value_cols].mean()
        self.group_mean_: dict[str, dict] = {c: group_stats[c].to_dict() for c in value_cols}
        self._group_count_: dict = group.value_counts().to_dict()
        group_sums = X.groupby(group)[value_cols].sum()
        self._group_sum_: dict[str, dict] = {c: group_sums[c].to_dict() for c in value_cols}

        gm_key = pd.MultiIndex.from_arrays([group, month], names=["_group", "_month"])
        gm_df = X[value_cols].set_index(gm_key)
        gm_stats = gm_df.groupby(level=[0, 1]).mean()
        gm_sums = gm_df.groupby(level=[0, 1]).sum()
        gm_counts = gm_df.groupby(level=[0, 1]).size()
        valid_buckets = set(gm_counts[gm_counts >= self.min_group_month_size].index)
        self.group_month_mean_: dict[str, dict] = {
            c: {k: v for k, v in gm_stats[c].to_dict().items() if k in valid_buckets}
            for c in value_cols
        }
        self._group_month_sum_: dict[str, dict] = {
            c: {k: v for k, v in gm_sums[c].to_dict().items() if k in valid_buckets}
            for c in value_cols
        }
        self._group_month_count_: dict = {
            k: v for k, v in gm_counts.to_dict().items() if k in valid_buckets
        }
        self.n_group_month_buckets_fit_ = len(valid_buckets)
        self.n_group_month_buckets_total_ = len(gm_counts)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["global_mean_", "group_mean_", "group_month_mean_"])
        _validate_columns(X, self._required_cols(), "ClimateAnomalyFeaturizer.transform")
        _validate_no_nulls(X, self._required_cols(), "ClimateAnomalyFeaturizer.transform")

        value_cols = list(self.value_cols)
        new_cols = [f"{c}{self.output_suffix}" for c in value_cols] + [
            f"{c}{self.output_suffix}_baseline_level" for c in value_cols
        ]
        _check_no_output_collision(X, new_cols, "ClimateAnomalyFeaturizer.transform")

        month = X[self.deathdate_col].dt.month
        group = X[self.group_col]
        keys = list(zip(group.to_numpy(), month.to_numpy(), strict=True))

        out = X.copy()
        for c in value_cols:
            gm_lookup = self.group_month_mean_[c]
            g_lookup = self.group_mean_[c]
            global_val = self.global_mean_[c]

            baseline = np.empty(len(X), dtype=float)
            level = np.empty(len(X), dtype=object)
            for i, (g, key) in enumerate(zip(group.to_numpy(), keys, strict=True)):
                if key in gm_lookup:
                    baseline[i] = gm_lookup[key]
                    level[i] = "group_month"
                elif g in g_lookup:
                    baseline[i] = g_lookup[g]
                    level[i] = "group"
                else:
                    baseline[i] = global_val
                    level[i] = "global"

            out[f"{c}{self.output_suffix}"] = out[c].to_numpy() - baseline
            out[f"{c}{self.output_suffix}_baseline_level"] = level
        return out

    def _transform_loo(self, X: pd.DataFrame) -> pd.DataFrame:
        """Leave-one-out variant of `transform`, used only by `fit_transform` — see
        the class docstring for why this must differ from `fit(X).transform(X)`.
        Assumes `X` is (row-for-row) the same data just passed to `fit`."""
        value_cols = list(self.value_cols)
        new_cols = [f"{c}{self.output_suffix}" for c in value_cols] + [
            f"{c}{self.output_suffix}_baseline_level" for c in value_cols
        ]
        _check_no_output_collision(X, new_cols, "ClimateAnomalyFeaturizer.fit_transform")

        month = X[self.deathdate_col].dt.month
        group = X[self.group_col]
        keys = list(zip(group.to_numpy(), month.to_numpy(), strict=True))

        out = X.copy()
        for c in value_cols:
            gm_sum = self._group_month_sum_[c]
            gm_count = self._group_month_count_
            g_sum = self._group_sum_[c]
            g_count = self._group_count_
            global_sum = self._global_sum_[c]
            n_train = self.n_train_

            values = out[c].to_numpy()
            baseline = np.empty(len(X), dtype=float)
            level = np.empty(len(X), dtype=object)
            for i, (g, key, x_i) in enumerate(zip(group.to_numpy(), keys, values, strict=True)):
                # Tier 1: (group, month) — only if excluding this row still leaves
                # at least one other row to average over.
                if key in gm_count and gm_count[key] - 1 >= 1:
                    baseline[i] = (gm_sum[key] - x_i) / (gm_count[key] - 1)
                    level[i] = "group_month"
                # Tier 2: group only.
                elif g in g_count and g_count[g] - 1 >= 1:
                    baseline[i] = (g_sum[g] - x_i) / (g_count[g] - 1)
                    level[i] = "group"
                # Tier 3: global — always has n_train - 1 >= 1 other rows unless the
                # entire fit set had a single row, an unrealistic edge case guarded
                # against by falling back to the plain (non-LOO) global mean instead
                # of dividing by zero.
                elif n_train - 1 >= 1:
                    baseline[i] = (global_sum - x_i) / (n_train - 1)
                    level[i] = "global"
                else:
                    baseline[i] = self.global_mean_[c]
                    level[i] = "global"

            out[f"{c}{self.output_suffix}"] = values - baseline
            out[f"{c}{self.output_suffix}_baseline_level"] = level
        return out

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        if fit_params:
            raise TypeError(
                f"ClimateAnomalyFeaturizer.fit_transform: received unexpected fit_params "
                f"{list(fit_params)} — fit() takes no extra keyword arguments."
            )
        self.fit(X, y)
        return self._transform_loo(X)


# =============================================================================
# Track D — NDVI/vegetation (stretch)
# =============================================================================


def add_ndvi_trend_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Track D (stretch): `ndvi_trend_30_90` = ndvi_30d - ndvi_90d — positive means
    recent (30-day) vegetation greenness exceeds the longer (90-day) baseline
    (greening), negative means recent browning relative to the longer window.

    Only two NDVI windows are provided (30d, 90d composites), so this is a simple
    difference, not a fitted slope over a real time series — scoped to what the data
    actually supports, not a proxy for a richer trend the columns can't provide.
    """
    required = ["ndvi_30d", "ndvi_90d"]
    _validate_columns(df, required, "add_ndvi_trend_feature")
    _validate_no_nulls(df, required, "add_ndvi_trend_feature")
    _check_no_output_collision(df, ["ndvi_trend_30_90"], "add_ndvi_trend_feature")

    out = df.copy()
    out["ndvi_trend_30_90"] = out["ndvi_30d"] - out["ndvi_90d"]
    return out


# =============================================================================
# Provenance / leakage-timing table
# =============================================================================


@dataclass(frozen=True)
class ProvenanceEntry:
    feature: str
    reference_period: str
    available_at_prediction_time: str
    risk: str
    notes: str = ""


FEATURE_PROVENANCE: tuple[ProvenanceEntry, ...] = (
    # --- as provided in climate_features.csv ---
    ProvenanceEntry("rain_sum_7d", "T-7d -> T", "Yes (as provided)", "Low"),
    ProvenanceEntry("rain_sum_30d", "T-30d -> T", "Yes (as provided)", "Low"),
    ProvenanceEntry("rain_sum_90d", "T-90d -> T", "Yes (as provided)", "Low"),
    ProvenanceEntry("rain_days_30d", "T-30d -> T", "Yes (as provided)", "Low"),
    ProvenanceEntry("max_daily_rain_30d", "T-30d -> T", "Yes (as provided)", "Low"),
    ProvenanceEntry("tavg_7d", "T-7d -> T", "Yes (as provided)", "Low"),
    ProvenanceEntry("tavg_30d", "T-30d -> T", "Yes (as provided)", "Low"),
    ProvenanceEntry("tavg_90d", "T-90d -> T", "Yes (as provided)", "Low"),
    ProvenanceEntry("tmax_30d", "T-30d -> T", "Yes (as provided)", "Low"),
    ProvenanceEntry("tmin_30d", "T-30d -> T", "Yes (as provided)", "Low"),
    ProvenanceEntry(
        "temp_range_mean_30d",
        "T-30d -> T",
        "Yes (as provided)",
        "Low",
    ),
    ProvenanceEntry(
        "ndvi_30d",
        "T-30d (satellite composite, may lag)",
        "Needs verification",
        "Medium",
        "MODIS composites can have real-world processing/publication lag; fine for "
        "competition use as delivered, flagged for the Stage 19 deployment boundary.",
    ),
    ProvenanceEntry(
        "ndvi_90d",
        "T-90d (satellite composite, may lag)",
        "Needs verification",
        "Medium",
        "Same lag caveat as ndvi_30d.",
    ),
    ProvenanceEntry("elevation", "Static (terrain)", "Yes", "Low"),
    ProvenanceEntry("slope", "Static (terrain)", "Yes", "Low"),
    ProvenanceEntry(
        "hot_days_30d",
        "T-30d -> T",
        "N/A — dropped",
        "N/A",
        "Confirmed constant (0) across all 4,176 rows (Stage 3 forensics, "
        "re-verified by drop_dead_columns at every pipeline run). Dropped by Track A.",
    ),
    # --- Track A derived ---
    ProvenanceEntry(
        "rain_rate_7d / 30d / 90d",
        "same as source window",
        "Yes (pure arithmetic of provided columns)",
        "Low",
    ),
    ProvenanceEntry(
        "rain_ratio_acute_medium",
        "T-7d & T-30d",
        "Yes",
        "Low",
        "NaN when rain_sum_30d == 0 (4/3,146 training rows) — genuinely undefined, "
        "not imputed to 0.",
    ),
    ProvenanceEntry(
        "rain_ratio_medium_chronic",
        "T-30d & T-90d",
        "Yes",
        "Low",
        "rain_sum_90d is never 0 in the observed data, but the ratio is still "
        "NaN-safe if that changes.",
    ),
    ProvenanceEntry(
        "rain_intensity_30d",
        "T-30d",
        "Yes",
        "Low",
        "NaN when rain_sum_30d == 0.",
    ),
    ProvenanceEntry("rain_day_fraction_30d", "T-30d", "Yes", "Low"),
    # --- Track B derived ---
    ProvenanceEntry(
        "tmax_30d_exceeds_flag / _exceedance_margin",
        "T-30d",
        "Yes, but threshold is a train-only-fit constant",
        "Low",
        "Threshold (compute_heat_threshold) must be fit once on training data and "
        "reused verbatim for test/production — never recomputed per-dataset.",
    ),
    ProvenanceEntry(
        "{col}_anomaly (temperature)",
        "T-30d/90d vs. a learned train-only baseline",
        "Yes, but the baseline must be the one learned from training, never "
        "recomputed from the batch being scored",
        "Medium",
        "See ClimateAnomalyFeaturizer docstring for the 3-tier fallback "
        "(group x month -> group -> global) and why it must be fit-on-train-only. "
        "fit_transform() (used to build this feature on the training set itself) "
        "uses leave-one-out bucket means so a training row's own value never "
        "inflates its own baseline; transform() on any other dataset (test, "
        "production) is unaffected since those rows never contributed to the "
        "baseline in the first place.",
    ),
    # --- Track C derived ---
    ProvenanceEntry(
        "{col}_anomaly (rainfall)",
        "T-30d/90d vs. a learned train-only baseline",
        "Yes, same caveat as the temperature anomaly",
        "Medium",
        "Same leave-one-out fit_transform() behavior as the temperature anomaly.",
    ),
    ProvenanceEntry(
        "consecutive dry-spell length",
        "n/a — not implemented",
        "N/A",
        "N/A",
        "NOT computable from the provided aggregates: rain_days_30d is a count of "
        "rainy days in the window, not their positions, so two windows with the same "
        "count can have very different spell structure. Documented as a known "
        "limitation rather than shipped as a feature that looks like it measures "
        "something it doesn't.",
    ),
    # --- Track D derived (stretch) ---
    ProvenanceEntry(
        "ndvi_trend_30_90",
        "T-30d & T-90d (satellite composite, may lag)",
        "Needs verification",
        "Medium",
        "Inherits both underlying columns' composite-lag caveat.",
    ),
)


def render_provenance_markdown() -> str:
    """Renders `FEATURE_PROVENANCE` as a markdown table for `docs/`."""
    header = "| Feature | Reference period | Available at prediction time? | Risk | Notes |\n"
    header += "|---|---|---|---|---|\n"
    rows = [
        f"| `{e.feature}` | {e.reference_period} | {e.available_at_prediction_time} | "
        f"{e.risk} | {e.notes} |"
        for e in FEATURE_PROVENANCE
    ]
    return header + "\n".join(rows) + "\n"
