"""Stage 14.2 — ensembling strategies over out-of-fold (OOF) predictions.

Shared contract every strategy in this module follows (Stage 14.1's diversity
analysis is what these operate on):

    oof_preds: dict[str, np.ndarray]
        {model_name: array of OOF predicted probabilities}, one array per
        candidate, all the same length and row-aligned to the same dataset.
        Never in-fold predictions — see module docstring of
        notebooks/04_modeling_experiments.ipynb's Stage 14.1 cells for how
        these are produced.

Each combination strategy below takes `oof_preds` (plus whatever extra
arguments it specifically needs) and returns one `np.ndarray` of blended
predicted probabilities, same length and row order as the inputs — so all
three plug into `climate_health.models.baselines.competition_score` the same
way a single model's predictions would.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression


def weighted_average(oof_preds: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Weighted average of OOF predictions. `weights` need not sum to 1 —
    they're normalized here, so callers can pass raw optimizer output directly."""
    if set(weights) != set(oof_preds):
        raise ValueError(
            f"weighted_average: weights keys {set(weights)} must match "
            f"oof_preds keys {set(oof_preds)}"
        )
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("weighted_average: weights must sum to a positive number")
    blended = np.zeros_like(next(iter(oof_preds.values())), dtype=float)
    for name, preds in oof_preds.items():
        blended += (weights[name] / total) * preds
    return blended


def optimize_weights(
    oof_preds: dict[str, np.ndarray],
    y_true: np.ndarray,
    score_fn: Callable[[np.ndarray, np.ndarray], float],
    n_samples: int = 2000,
    random_state: int = 42,
) -> dict[str, float]:
    """Finds non-negative weights (summing to 1) that maximize `score_fn` on
    `weighted_average(oof_preds, weights)`.

    Uses random search over the weight simplex (Dirichlet-sampled), not a
    gradient-based optimizer: `score_fn` here is the competition score
    (F1@0.5 + AUC), which is piecewise-constant almost everywhere — its
    gradient is zero at nearly every point, so a gradient method like SLSQP
    simply never leaves its starting guess (confirmed by a failing test
    during development: SLSQP returned the initial equal weights unchanged
    even when one candidate predicted the label perfectly and the other was
    pure noise). Random search over the simplex has no such blind spot.

    Always evaluates the uniform-weight point and every single-candidate
    corner (weight 1 on one model, 0 on the rest) in addition to
    `n_samples` random draws, so a candidate that's simply the best choice
    on its own is never missed by sampling variance. Deterministic for a
    fixed `random_state`.
    """
    names = list(oof_preds)
    n = len(names)
    rng = np.random.RandomState(random_state)

    candidate_weights = [
        np.full(n, 1.0 / n),
        *np.eye(n),
        *rng.dirichlet(np.ones(n), size=n_samples),
    ]

    best_score = -np.inf
    best_weights = dict(zip(names, candidate_weights[0], strict=True))
    for w in candidate_weights:
        weights = dict(zip(names, w, strict=True))
        score = score_fn(y_true, weighted_average(oof_preds, weights))
        if score > best_score:
            best_score = score
            best_weights = weights
    return best_weights


def rank_average(oof_preds: dict[str, np.ndarray]) -> np.ndarray:
    """Average of each candidate's fractional rank (0-1, via `rankdata`),
    not raw probabilities — invariant to any monotonic miscalibration
    between candidates, unlike a plain probability average. Threshold at 0.5
    on the output the same way as a probability, per the blueprint's use of
    rank averaging for exactly this fixed-threshold competition metric."""
    n = len(next(iter(oof_preds.values())))
    ranks = np.zeros(n, dtype=float)
    for preds in oof_preds.values():
        ranks += rankdata(preds) / n  # fractional rank in (0, 1]
    return ranks / len(oof_preds)


def oof_stack(
    oof_preds: dict[str, np.ndarray],
    y_true: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    meta_learner_factory: Callable[[], object] = lambda: LogisticRegression(max_iter=1000),
) -> np.ndarray:
    """OOF stacking: a meta-learner trained on the candidates' OOF predictions
    as features, itself evaluated out-of-fold via `splits` — so the
    meta-learner never sees the label of a row it's predicting, the same
    discipline the base models' own OOF predictions were built under.

    `splits` is a sequence of (train_idx, val_idx) *positional* index pairs
    partitioning every row exactly once (e.g. one repeat of `tier2_splits`)."""
    names = list(oof_preds)
    meta_X = np.column_stack([oof_preds[name] for name in names])
    meta_oof = np.full(len(y_true), np.nan)

    for train_idx, val_idx in splits:
        model = meta_learner_factory()
        model.fit(meta_X[train_idx], y_true[train_idx])
        meta_oof[val_idx] = model.predict_proba(meta_X[val_idx])[:, 1]

    if np.isnan(meta_oof).any():
        raise ValueError(
            "oof_stack: splits did not cover every row exactly once — "
            f"{np.isnan(meta_oof).sum()} row(s) never appeared in a val_idx."
        )
    return meta_oof
