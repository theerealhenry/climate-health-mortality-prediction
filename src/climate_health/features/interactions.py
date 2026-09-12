"""
Stage 9 — Interaction features.

Three interaction terms, each motivated by a specific finding already in hand rather
than speculative crossing of arbitrary columns (per the blueprint's explicit warning
that interactions are where "cheap-looking" AUC gain hides):

  1. age x tavg_30d: does climate exposure matter differently by age? Motivated by
     age's dominance (Stage 4: corr=-0.44, univariate AUC~=0.74) and the under-5/60+
     vulnerability pattern in the age-band rates (Stage 6).
  2. spatial_cluster x zone: does Rural/Peri_urban mean something different in
     different climate clusters? A combined category, not a product, since both
     inputs are categorical.
  3. climate-anomaly x age-vulnerability: does an anomalous climate reading matter
     more for the two vulnerable age groups specifically (under-5, 60+) rather than
     uniformly across all ages?

None of these are wired into a "default" feature set here. Per this project's
established pattern (Stage 6's rainy-season flag, Stage 8's dry-spell non-feature):
a plausible-sounding interaction is validated (kept or dropped) against Tier 2 CV
during modeling, not assumed useful because it compiled. These functions only
compute the candidate columns; the accept/reject decision happens later, once a model
exists to score them against (see docs/PROJECT_BLUEPRINT.md Stage 9).
"""

from __future__ import annotations

from climate_health.utils.validation import (
    check_no_output_collision,
    validate_no_nulls,
    validate_required_columns,
)


def add_age_climate_interaction(df, age_col="age", climate_col="tavg_30d"):
    """Adds `{age_col}_x_{climate_col}` = age_col * climate_col (elementwise product).

    Raises
    ------
    KeyError
        If either column is missing.
    ValueError
        If either column contains nulls, or the output column already exists.
    """
    validate_required_columns(df, [age_col, climate_col], "add_age_climate_interaction")
    validate_no_nulls(df, [age_col, climate_col], "add_age_climate_interaction")
    out_col = f"{age_col}_x_{climate_col}"
    check_no_output_collision(df, [out_col], "add_age_climate_interaction")

    out = df.copy()
    out[out_col] = out[age_col] * out[climate_col]
    return out


def add_cluster_zone_interaction(df, cluster_col="spatial_cluster", zone_col="zone"):
    """Adds `{cluster_col}_x_{zone_col}` = a combined category string
    "<cluster>_<zone>" — a categorical x categorical cross, not a product, since
    neither input is numeric. Downstream, this combined category can be one-hot or
    target-encoded like any other categorical column.

    Raises
    ------
    KeyError
        If either column is missing.
    ValueError
        If either column contains nulls, or the output column already exists.
    """
    validate_required_columns(df, [cluster_col, zone_col], "add_cluster_zone_interaction")
    validate_no_nulls(df, [cluster_col, zone_col], "add_cluster_zone_interaction")
    out_col = f"{cluster_col}_x_{zone_col}"
    check_no_output_collision(df, [out_col], "add_cluster_zone_interaction")

    out = df.copy()
    out[out_col] = out[cluster_col].astype(str) + "_" + out[zone_col].astype(str)
    return out


def add_anomaly_age_vulnerability_interaction(df, anomaly_col, age_band_col="age_band"):
    """Adds two columns, `{anomaly_col}_x_under5` and `{anomaly_col}_x_60plus`: the
    anomaly value multiplied by a 0/1 flag for each named vulnerable age band. Two
    flag interactions rather than one per age_band, because the hypothesis being
    tested is specifically "does this anomaly matter more for the two vulnerable
    groups" (Stage 6's age-band rates), not an unmotivated cross of every band.

    Raises
    ------
    KeyError
        If either column is missing.
    ValueError
        If either column contains nulls, either output column already exists, or
        `age_band_col` contains a value other than the four bands `demographic.py`
        produces (0-4, 5-17, 18-59, 60+).
    """
    validate_required_columns(
        df, [anomaly_col, age_band_col], "add_anomaly_age_vulnerability_interaction"
    )
    validate_no_nulls(df, [anomaly_col, age_band_col], "add_anomaly_age_vulnerability_interaction")
    under5_col = f"{anomaly_col}_x_under5"
    over60_col = f"{anomaly_col}_x_60plus"
    check_no_output_collision(
        df, [under5_col, over60_col], "add_anomaly_age_vulnerability_interaction"
    )

    valid_bands = {"0-4", "5-17", "18-59", "60+"}
    bad = set(df[age_band_col].unique()) - valid_bands
    if bad:
        raise ValueError(
            f"add_anomaly_age_vulnerability_interaction: unexpected age_band value(s) "
            f"{sorted(bad)} — expected one of {sorted(valid_bands)} (from "
            "features.demographic.add_age_band_features)."
        )

    out = df.copy()
    out[under5_col] = out[anomaly_col] * (out[age_band_col] == "0-4").astype(int)
    out[over60_col] = out[anomaly_col] * (out[age_band_col] == "60+").astype(int)
    return out
