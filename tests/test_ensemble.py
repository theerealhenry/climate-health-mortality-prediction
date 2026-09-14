"""Tests for Stage 14.2's ensembling strategies (src/climate_health/models/ensemble.py).

Each strategy is tested against a small synthetic case where the "correct"
answer is knowable by construction, rather than against real project data —
these are pure-logic combination functions, so a fast, deterministic unit
test proves more than an expensive end-to-end run would.
"""

import numpy as np
import pytest
from sklearn.model_selection import KFold

from climate_health.models.ensemble import (
    oof_stack,
    optimize_weights,
    rank_average,
    weighted_average,
)


def competition_score_like(y_true, y_pred_proba) -> float:
    """Local stand-in for baselines.competition_score (avoids importing the
    full models package, which pulls in lightgbm/xgboost/catboost, for tests
    that only exercise pure-numpy combination logic)."""
    calls = (y_pred_proba >= 0.5).astype(int)
    tp = np.sum((calls == 1) & (y_true == 1))
    fp = np.sum((calls == 1) & (y_true == 0))
    fn = np.sum((calls == 0) & (y_true == 1))
    f1 = 0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)
    # AUC via rank-sum (Mann-Whitney U), no sklearn dependency needed here
    pos, neg = y_pred_proba[y_true == 1], y_pred_proba[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        auc = 0.5
    else:
        ranks = pd_rank(np.concatenate([pos, neg]))
        auc = (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return 0.60 * f1 + 0.40 * auc


def pd_rank(arr):
    from scipy.stats import rankdata

    return rankdata(arr)


# ---------------------------------------------------------------------------
# weighted_average
# ---------------------------------------------------------------------------


def test_weighted_average_computes_the_linear_combination():
    oof_preds = {"a": np.array([1.0, 0.0, 0.5]), "b": np.array([0.0, 1.0, 0.5])}
    result = weighted_average(oof_preds, {"a": 0.75, "b": 0.25})
    np.testing.assert_allclose(result, [0.75, 0.25, 0.5])


def test_weighted_average_normalizes_weights_that_dont_sum_to_one():
    oof_preds = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    # 3:1 ratio, unnormalized (sums to 4) — should behave identically to 0.75/0.25
    result = weighted_average(oof_preds, {"a": 3.0, "b": 1.0})
    np.testing.assert_allclose(result, [0.75, 0.25])


def test_weighted_average_mismatched_keys_raises():
    oof_preds = {"a": np.array([1.0]), "b": np.array([0.0])}
    with pytest.raises(ValueError, match="keys"):
        weighted_average(oof_preds, {"a": 1.0, "c": 1.0})


# ---------------------------------------------------------------------------
# optimize_weights
# ---------------------------------------------------------------------------


def test_optimize_weights_favors_the_candidate_that_predicts_perfectly():
    rng = np.random.RandomState(0)
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0] * 5)
    perfect = y_true.astype(float)  # candidate "a" predicts the label exactly
    # candidate "b" is anti-correlated (actively wrong), not just uninformative —
    # a merely-uninformative candidate at 0.5/0.5 weight can still leave "a"'s
    # separation intact (as an earlier version of this test accidentally did),
    # which doesn't actually exercise the optimizer's ability to down-weight a
    # harmful candidate. Anti-correlation forces a real tradeoff: uniform
    # weights should score noticeably worse than weighting "a" alone.
    anti_correlated = 1.0 - y_true.astype(float) + rng.uniform(-0.05, 0.05, size=len(y_true))

    weights = optimize_weights({"a": perfect, "b": anti_correlated}, y_true, competition_score_like)

    assert weights["a"] > 0.9, f"expected near-all weight on the perfect predictor, got {weights}"
    blended = weighted_average({"a": perfect, "b": anti_correlated}, weights)
    assert competition_score_like(y_true, blended) > 0.95

    uniform_blend = weighted_average({"a": perfect, "b": anti_correlated}, {"a": 0.5, "b": 0.5})
    assert competition_score_like(y_true, uniform_blend) < 0.6, (
        "sanity check: uniform weighting of a perfect + anti-correlated candidate "
        "should score badly, otherwise this test isn't exercising anything"
    )


# ---------------------------------------------------------------------------
# rank_average
# ---------------------------------------------------------------------------


def test_rank_average_is_invariant_to_monotonic_miscalibration():
    p = np.array([0.05, 0.4, 0.35, 0.8, 0.6])
    p_miscalibrated = p**3  # monotonic transform: same ordering, very different scale

    result_raw = rank_average({"a": p, "b": p})
    result_miscalibrated = rank_average({"a": p, "b": p_miscalibrated})

    np.testing.assert_allclose(result_raw, result_miscalibrated)


def test_rank_average_output_is_bounded_zero_to_one():
    oof_preds = {"a": np.array([0.1, 0.9, 0.5]), "b": np.array([0.8, 0.2, 0.4])}
    result = rank_average(oof_preds)
    assert np.all(result > 0) and np.all(result <= 1.0)


# ---------------------------------------------------------------------------
# oof_stack
# ---------------------------------------------------------------------------


def test_oof_stack_combination_beats_either_input_alone_at_0_5_threshold():
    rng = np.random.RandomState(42)
    n = 200
    # y is driven by two weak, complementary signals — neither alone clears
    # 0.5 reliably, but a learned linear combination does.
    signal_a = rng.uniform(0, 1, n)
    signal_b = rng.uniform(0, 1, n)
    y_true = ((0.5 * signal_a + 0.5 * signal_b) > 0.5).astype(int)
    # noisy, weakly-informative individual "OOF predictions"
    pred_a = np.clip(signal_a + rng.normal(0, 0.35, n), 0, 1)
    pred_b = np.clip(signal_b + rng.normal(0, 0.35, n), 0, 1)
    oof_preds = {"a": pred_a, "b": pred_b}

    splits = list(KFold(n_splits=5, shuffle=True, random_state=0).split(np.arange(n)))
    stacked = oof_stack(oof_preds, y_true, splits)

    stacked_score = competition_score_like(y_true, stacked)
    a_alone = competition_score_like(y_true, pred_a)
    b_alone = competition_score_like(y_true, pred_b)

    assert stacked_score >= max(a_alone, b_alone), (
        f"stacked ({stacked_score:.4f}) should be at least as good as the better "
        f"single input (a={a_alone:.4f}, b={b_alone:.4f})"
    )


def test_oof_stack_raises_if_splits_do_not_cover_every_row():
    y_true = np.array([0, 1, 0, 1])
    oof_preds = {"a": np.array([0.1, 0.9, 0.2, 0.8])}
    incomplete_splits = [(np.array([0, 1]), np.array([2]))]  # row 3 never in a val_idx

    with pytest.raises(ValueError, match="cover every row"):
        oof_stack(oof_preds, y_true, incomplete_splits)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
