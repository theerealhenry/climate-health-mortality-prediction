"""Shared DataFrame-contract checks used across `climate_health.features` modules.

These three checks originated in `features/climate.py` (Stage 8) as private,
module-local helpers. When `features/temporal.py` and `features/demographic.py`
(Stage 6) needed the identical checks, the choice was between a third/fourth
private copy (guaranteed to drift eventually — exactly the kind of duplication this
project's own adversarial reviews exist to catch) or promoting them here once. This
module is that promotion; `climate.py` now imports from here instead of defining its
own copies.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def validate_required_columns(df: pd.DataFrame, cols: Sequence[str], fn_name: str) -> None:
    """Raises `KeyError` listing every missing column at once (not just the first),
    so a caller fixing a call site sees the whole problem in one error, not one
    missing column per re-run."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{fn_name}: missing required column(s) {missing}")


def validate_no_nulls(df: pd.DataFrame, cols: Sequence[str], fn_name: str) -> None:
    """Raises `ValueError` listing every column that contains a null, given `cols`
    are all confirmed present (call `validate_required_columns` first)."""
    sub = df[list(cols)]
    if sub.isna().any().any():
        bad_cols = sub.columns[sub.isna().any()].tolist()
        raise ValueError(f"{fn_name}: null values found in column(s) {bad_cols}")


def check_no_output_collision(df: pd.DataFrame, new_cols: Sequence[str], fn_name: str) -> None:
    """Raises `ValueError` if `df` already has any of the columns a feature function
    is about to add — refusing rather than silently overwriting a same-named column
    that came from somewhere else (or a stale re-run), which is exactly the bug class
    Stage 7's review found and fixed in a plain merge."""
    collisions = [c for c in new_cols if c in df.columns]
    if collisions:
        raise ValueError(
            f"{fn_name}: X already has column(s) {collisions} that this function would "
            "add — refusing, since a stale or unrelated column with the same name would "
            "otherwise be silently overwritten. Drop/rename it first."
        )
