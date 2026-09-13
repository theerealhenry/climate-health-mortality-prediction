"""Stage 10.2 — single feature-assembly entry point chaining Stages 6-9.

Why this exists: `evaluation.cv.evaluate_all_tiers` (Stage 5, already shipped)
takes one fixed `df[feature_cols]` matrix and reuses it, unmodified, across
every CV fold in all three tiers — it never refits anything per fold. That
means the fit/reuse contract every Stage 6-9 transform already follows
(SpatialClusterFeaturizer, ClimateAnomalyFeaturizer, compute_reference_year,
compute_heat_threshold, fit_target_encoding_map) is exactly the contract this
module needs too: fit everything once on train, reuse verbatim on test. There
is no fold-scoped variant to build here — `compute_oof_target_encoding`
(Stage 9) stays available as its own tool for anyone doing a more careful,
CV-aware leakage check outside `evaluate_all_tiers`, but wiring it into this
module would build a caller that doesn't exist in this codebase.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from climate_health.features.climate import (
    ClimateAnomalyFeaturizer,
    add_heat_exceedance_features,
    add_rainfall_rate_features,
    compute_heat_threshold,
    drop_dead_columns,
)
from climate_health.features.demographic import (
    add_age_band_features,
    add_age_continuous_transforms,
)
from climate_health.features.encoding import apply_target_encoding, fit_target_encoding_map
from climate_health.features.spatial import (
    SpatialClusterFeaturizer,
    add_coordinate_polynomial_features,
)
from climate_health.features.temporal import (
    add_date_decomposition_features,
    add_year_trend_feature,
    compute_reference_year,
)

# Encoded columns, fit on train only, reused verbatim on test.
_ENCODED_COLS = ("spatial_cluster", "zone")

# Raw categorical columns build_feature_matrix leaves in native (non-numeric) form
# alongside their target-encoded counterparts where one exists. Public (no leading
# underscore): branch_a_native_categorical uses this directly, and other modules
# (e.g. models/zoo.py, for CatBoost's cat_features) go through
# `categorical_feature_positions` below rather than reaching in for the raw tuple.
RAW_CATEGORICAL_COLS = ("zone", "gender", "spatial_cluster", "age_band")


def categorical_feature_positions(feature_cols: Sequence[str]) -> list[int]:
    """Positional indices within `feature_cols` that are raw categorical columns —
    exactly the input CatBoost's `cat_features` constructor argument expects. Exists
    so callers outside this module (e.g. the model zoo) don't need to import
    RAW_CATEGORICAL_COLS and re-derive this themselves."""
    return [i for i, c in enumerate(feature_cols) if c in RAW_CATEGORICAL_COLS]


# Interaction functions (Stage 9) aren't validated as helpful yet (see
# session_handoff_stage9_to_10.md) — opt in explicitly per name, default off.
_INTERACTION_FNS = {}
try:
    from climate_health.features.interactions import (
        add_age_climate_interaction,
        add_anomaly_age_vulnerability_interaction,
        add_cluster_zone_interaction,
    )

    def _anomaly_age_vulnerability(df):
        return add_anomaly_age_vulnerability_interaction(df, anomaly_col="tavg_30d_anomaly")

    _INTERACTION_FNS = {
        "age_climate_interaction": lambda df: add_age_climate_interaction(df),
        "cluster_zone_interaction": lambda df: add_cluster_zone_interaction(df),
        "anomaly_age_vulnerability_interaction": _anomaly_age_vulnerability,
    }
except ImportError:
    pass


@dataclass(frozen=True)
class FeatureFitState:
    """Everything build_feature_matrix(fit=True) learned from train, to be reused
    verbatim (never refit) on test. Every mutable field is deepcopy'd on construction
    so a caller holding a reference to `spatial_cluster`/`climate_anomaly`/
    `target_encoding` can't mutate a state another caller is still using — e.g. by
    calling `.fit()` again, or editing a target-encoding mapping in place.

    `branch_b_columns` freezes the one-hot output schema `branch_b_encoded` produced
    on the fit data, so a later call on test data (which may not contain every
    category value train did) can reindex to the exact same columns instead of
    silently producing a differently-shaped matrix — see branch_b_encoded's
    `fitted_columns` argument.
    """

    reference_year: int
    heat_threshold: float
    spatial_cluster: SpatialClusterFeaturizer
    climate_anomaly: ClimateAnomalyFeaturizer
    target_encoding: dict[str, tuple[dict, float]]  # col -> (mapping, global_mean)
    branch_b_columns: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "spatial_cluster", copy.deepcopy(self.spatial_cluster))
        object.__setattr__(self, "climate_anomaly", copy.deepcopy(self.climate_anomaly))
        object.__setattr__(self, "target_encoding", copy.deepcopy(self.target_encoding))


# --- Stage 11.2 ADR: two-branch column selection ---------------------------------
#
# Decision: build_feature_matrix's output is branched by *selection*, not by a
# second fitting pass. It already carries both representations of every
# categorical column that has one — the raw value (`zone`, `spatial_cluster`, ...)
# and, where Stage 9 built one, the target-encoded value (`zone_te`,
# `spatial_cluster_te`). Branching is just picking which columns a given model
# family gets, so there is no new transform to fit/leak-check here.
#
# Why it exists: v1 fed every model — including CatBoost, chosen specifically for
# native categorical support — the same pre-one-hot matrix, making the stated
# reason for including CatBoost untrue of what it actually received (blueprint
# Stage 11, Review 2 point 3). Branch A below is the actual fix.
#
# Branch A (CatBoost): raw categoricals kept as pandas "category" dtype,
# target-encoded columns dropped — native handling needs the real category, not a
# numeric proxy for it.
#
# XGBoost note: the original plan (see the blueprint) was to also give XGBoost
# Branch A via `enable_categorical=True`. In practice `models/zoo.py`'s
# `xgboost_tuned` runs on Branch B instead: `evaluation.cv.evaluate_all_tiers`
# converts the whole feature matrix to a numpy array via `.values` before fitting
# (`X = df[list(feature_cols)].values`), which strips real pandas `category` dtype
# from every column — there is no way for `enable_categorical=True` to see a
# genuine category through that conversion. CatBoost is unaffected because its
# `cat_features` argument identifies categorical columns by *position*, so it
# still treats those columns natively even after the array conversion; XGBoost's
# `enable_categorical` requires an actual DataFrame with category dtype, which
# `evaluate_all_tiers` never gives it. This is why Branch A currently has exactly
# one consumer (CatBoost) rather than two.
#
# Branch B (LogisticRegression, XGBoost, and anything else without a categorical
# path through this harness): numeric columns plus the target-encoded columns; the
# raw categorical columns are dropped, INCLUDING `spatial_cluster` (an int label)
# even though it is numeric dtype — a cluster id is nominal, not ordinal, and a
# linear model reading it as a plain number would silently learn a meaningless
# order across clusters. This is a correctness fix relative to Stage 10.3's
# `numeric_feature_cols`, which currently lets that raw label through —
# deliberately not touched here, so Stage 10.3's already-committed baseline
# numbers stay reproducible; Stage 11.3 wiring picks up branch_b_encoded for its
# own Logistic Regression and XGBoost candidates instead.
# -----------------------------------------------------------------------------


def branch_a_native_categorical(X: pd.DataFrame, target_col: str = "is_climate_sensitive"):
    """CatBoost branch: numeric columns as-is, categoricals as pandas 'category'
    dtype, no target-encoded columns (redundant with the native category)."""
    cat = [c for c in RAW_CATEGORICAL_COLS if c in X.columns]
    numeric = [
        c
        for c in X.select_dtypes(include="number").columns
        if c != target_col and not c.endswith("_te") and c not in cat
    ]
    out = X[numeric + cat].copy()
    for c in cat:
        out[c] = out[c].astype("category")
    return out


def branch_b_encoded(
    X: pd.DataFrame,
    target_col: str = "is_climate_sensitive",
    *,
    fitted_columns: Sequence[str] | None = None,
):
    """Logistic-regression/XGBoost-style branch: numeric columns (minus the raw
    `spatial_cluster` ordinal label — see ADR above) plus one-hot for whichever raw
    categoricals don't already have a target-encoded form.

    `fitted_columns`: the exact column set (order included) to reindex the output
    to, typically `FeatureFitState.branch_b_columns` captured at fit time.
    `pd.get_dummies` derives its output columns from whatever category values are
    actually present in `X` — if a category present at fit time is entirely absent
    from `X` (a real, observed risk in this dataset: some `spatial_cluster` values
    exist in train but not in test), calling this without `fitted_columns` on that
    later data would silently produce a differently-shaped matrix. Pass
    `fitted_columns` on every call except the original fit-time call that produces
    the schema in the first place.
    """
    one_hot_cols = [c for c in RAW_CATEGORICAL_COLS if c not in _ENCODED_COLS and c in X.columns]
    numeric = [
        c
        for c in X.select_dtypes(include="number").columns
        if c != target_col and c != "spatial_cluster"
    ]
    out = pd.get_dummies(X[numeric + one_hot_cols], columns=one_hot_cols)
    if fitted_columns is not None:
        fitted_columns = list(fitted_columns)
        dropped = set(out.columns) - set(fitted_columns)
        added = set(fitted_columns) - set(out.columns)
        if dropped or added:
            warnings.warn(
                "branch_b_encoded: this call's data doesn't match the fitted schema — "
                f"{len(dropped)} column(s) present here but not at fit time (dropped), "
                f"{len(added)} column(s) present at fit time but absent here (filled "
                "with 0). This is expected when a category seen during fit doesn't "
                "appear in this data; if the counts look large, investigate before "
                "trusting predictions from this matrix.",
                stacklevel=2,
            )
        out = out.reindex(columns=fitted_columns, fill_value=0)
    return out


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    fit: bool,
    target_col: str = "is_climate_sensitive",
    fitted_state: FeatureFitState | None = None,
    extra_features: frozenset[str] = frozenset(),
) -> tuple[pd.DataFrame, FeatureFitState]:
    """Chains Stages 6-9 in dependency order into one model-ready matrix.

    fit=True: fits every transformer on `df` and returns (matrix, new_state).
    fit=False: requires `fitted_state`, reuses it verbatim, returns (matrix, fitted_state).
    `extra_features`: names from _INTERACTION_FNS to opt into (default: none — Stage 9's
    interactions aren't validated yet, so they don't ship in the default matrix).

    The returned state's `branch_b_columns` freezes the one-hot schema
    `branch_b_encoded` produced on this fit — pass it back in on any later
    `branch_b_encoded(..., fitted_columns=state.branch_b_columns)` call (e.g. at
    inference time) so that call's output columns match this one's exactly.
    """
    if fit and fitted_state is not None:
        raise ValueError(
            "build_feature_matrix: fit=True fits fresh state — pass fitted_state=None, "
            "or fit=False to reuse an existing state. Passing both is ambiguous."
        )
    if not fit and fitted_state is None:
        raise ValueError("build_feature_matrix: fit=False requires fitted_state.")
    unknown = extra_features - _INTERACTION_FNS.keys()
    if unknown:
        raise ValueError(f"build_feature_matrix: unknown extra_features {unknown}")

    out = drop_dead_columns(df)

    reference_year = compute_reference_year(out) if fit else fitted_state.reference_year
    out = add_date_decomposition_features(out)
    out = add_year_trend_feature(out, reference_year=reference_year)

    out = add_age_band_features(out)
    out = add_age_continuous_transforms(out)

    spatial = SpatialClusterFeaturizer() if fit else fitted_state.spatial_cluster
    out = spatial.fit_transform(out) if fit else spatial.transform(out)
    out = add_coordinate_polynomial_features(out)

    out = add_rainfall_rate_features(out)
    heat_threshold = compute_heat_threshold(out) if fit else fitted_state.heat_threshold
    out = add_heat_exceedance_features(out, threshold=heat_threshold)

    climate_anomaly = (
        ClimateAnomalyFeaturizer(value_cols=["tavg_30d", "rain_sum_90d"])
        if fit
        else fitted_state.climate_anomaly
    )
    out = climate_anomaly.fit_transform(out) if fit else climate_anomaly.transform(out)

    for name in extra_features:
        out = _INTERACTION_FNS[name](out)

    target_encoding: dict[str, tuple[dict, float]] = {}
    for col in _ENCODED_COLS:
        if fit:
            mapping, global_mean = fit_target_encoding_map(out, target_col, col)
            target_encoding[col] = (mapping, global_mean)
        else:
            mapping, global_mean = fitted_state.target_encoding[col]
        out[f"{col}_te"] = apply_target_encoding(out, col, mapping, global_mean)

    branch_b_columns = (
        tuple(branch_b_encoded(out, target_col=target_col).columns)
        if fit
        else fitted_state.branch_b_columns
    )

    state = (
        FeatureFitState(
            reference_year=reference_year,
            heat_threshold=heat_threshold,
            spatial_cluster=spatial,
            climate_anomaly=climate_anomaly,
            target_encoding=target_encoding,
            branch_b_columns=branch_b_columns,
        )
        if fit
        else fitted_state
    )
    return out, state
