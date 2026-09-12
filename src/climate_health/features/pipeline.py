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
    verbatim (never refit) on test. Transformers are deepcopy'd on construction so
    a caller holding a reference to `spatial_cluster`/`climate_anomaly` can't call
    `.fit()` on it again and silently mutate a state another caller is still using."""

    reference_year: int
    heat_threshold: float
    spatial_cluster: SpatialClusterFeaturizer
    climate_anomaly: ClimateAnomalyFeaturizer
    target_encoding: dict[str, tuple[dict, float]]  # col -> (mapping, global_mean)

    def __post_init__(self):
        object.__setattr__(self, "spatial_cluster", copy.deepcopy(self.spatial_cluster))
        object.__setattr__(self, "climate_anomaly", copy.deepcopy(self.climate_anomaly))


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

    state = (
        FeatureFitState(
            reference_year=reference_year,
            heat_threshold=heat_threshold,
            spatial_cluster=spatial,
            climate_anomaly=climate_anomaly,
            target_encoding=target_encoding,
        )
        if fit
        else fitted_state
    )
    return out, state
