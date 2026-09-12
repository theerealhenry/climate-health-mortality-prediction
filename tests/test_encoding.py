"""Tests for Stage 9 — src/climate_health/features/encoding.py.

The leakage tests (`TestOofLeakage`) are the most important tests in this stage: a
subtle bug in out-of-fold encoding would silently and severely inflate every
downstream model's apparent performance, per the module's own docstring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from climate_health.features import encoding as enc


def _synthetic_df(n_per_cat=20, n_cats=4, seed=0):
    """Small synthetic dataset with a real, learnable category->target relationship
    (each category has its own base rate) plus per-row noise."""
    rng = np.random.RandomState(seed)
    cats = np.repeat(np.arange(n_cats), n_per_cat)
    base_rates = {0: 0.1, 1: 0.9, 2: 0.5, 3: 0.3}
    y = np.array([rng.binomial(1, base_rates[c]) for c in cats])
    return pd.DataFrame({"cat": cats, "y": y}).reset_index(drop=True)


def _five_fold_splits(n, n_splits=5, seed=0):
    """A trivial, non-twin-aware K-fold partition — used only to test this module's
    OOF machinery in isolation; production use should pass cv.tier1_splits/tier2_splits
    instead (see encoding.py's docstring)."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_splits)
    for i in range(n_splits):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        yield train_idx, val_idx


class TestFitApplyTargetEncoding:
    def test_smoothing_pulls_toward_global_mean(self):
        df = _synthetic_df()
        mapping, global_mean = enc.fit_target_encoding_map(df, "y", "cat", smoothing=1000.0)
        # With enormous smoothing, every category's encoding collapses to ~global_mean.
        for v in mapping.values():
            assert v == pytest.approx(global_mean, abs=0.05)

    def test_zero_smoothing_matches_raw_group_mean(self):
        df = _synthetic_df()
        mapping, _ = enc.fit_target_encoding_map(df, "y", "cat", smoothing=0.0)
        raw = df.groupby("cat")["y"].mean().to_dict()
        for k in raw:
            assert mapping[k] == pytest.approx(raw[k])

    def test_apply_unseen_category_falls_back_to_global_mean(self):
        df = _synthetic_df()
        mapping, global_mean = enc.fit_target_encoding_map(df, "y", "cat")
        test_df = pd.DataFrame({"cat": [999]})
        out = enc.apply_target_encoding(test_df, "cat", mapping, global_mean)
        assert out.iloc[0] == pytest.approx(global_mean)

    def test_negative_smoothing_raises(self):
        with pytest.raises(ValueError, match="smoothing"):
            enc.fit_target_encoding_map(_synthetic_df(), "y", "cat", smoothing=-1.0)

    def test_missing_column_raises_keyerror(self):
        with pytest.raises(KeyError):
            enc.fit_target_encoding_map(pd.DataFrame({"y": [1]}), "y", "cat")

    def test_apply_missing_column_raises_keyerror(self):
        mapping, global_mean = enc.fit_target_encoding_map(_synthetic_df(), "y", "cat")
        with pytest.raises(KeyError):
            enc.apply_target_encoding(pd.DataFrame({"other": [1]}), "cat", mapping, global_mean)

    def test_apply_null_raises_valueerror(self):
        mapping, global_mean = enc.fit_target_encoding_map(_synthetic_df(), "y", "cat")
        df = pd.DataFrame({"cat": [0, np.nan]})
        with pytest.raises(ValueError, match="null values"):
            enc.apply_target_encoding(df, "cat", mapping, global_mean)

    def test_apply_output_name_and_seen_category_value(self):
        mapping, global_mean = enc.fit_target_encoding_map(
            _synthetic_df(), "y", "cat", smoothing=0.0
        )
        out = enc.apply_target_encoding(pd.DataFrame({"cat": [0]}), "cat", mapping, global_mean)
        assert out.name == "cat_te"
        assert out.iloc[0] == pytest.approx(mapping[0])


class TestOofLeakage:
    """The critical property: a fold's OOF encoding must never be influenced by that
    fold's own target values."""

    def test_oof_value_matches_hand_computed_other_folds_only(self):
        df = _synthetic_df()
        splits = list(_five_fold_splits(len(df)))
        oof = enc.compute_oof_target_encoding(df, "y", "cat", splits, smoothing=5.0)

        global_mean = df["y"].mean()
        for train_idx, val_idx in splits:
            train = df.iloc[train_idx]
            stats = train.groupby("cat")["y"].agg(["mean", "count"])
            expected = (stats["mean"] * stats["count"] + global_mean * 5.0) / (stats["count"] + 5.0)
            got = oof.iloc[val_idx]
            want = df["cat"].iloc[val_idx].map(expected)
            pd.testing.assert_series_equal(
                got.reset_index(drop=True), want.reset_index(drop=True), check_names=False
            )

    def test_changing_a_fold_own_labels_does_not_change_its_own_oof_value(self):
        """The single most important leakage check: perturb the target *only* within
        one fold's own validation rows, recompute OOF, and assert that fold's OOF
        values are completely unchanged — proof the encoding never looked at the
        labels of the rows it was applied to."""
        df = _synthetic_df()
        splits = list(_five_fold_splits(len(df)))
        oof_before = enc.compute_oof_target_encoding(df, "y", "cat", splits, smoothing=5.0)

        _, first_val_idx = splits[0]
        df_perturbed = df.copy()
        # Flip every label in fold 0's own validation rows.
        df_perturbed.loc[df_perturbed.index[first_val_idx], "y"] = (
            1 - df_perturbed.loc[df_perturbed.index[first_val_idx], "y"]
        )
        oof_after = enc.compute_oof_target_encoding(df_perturbed, "y", "cat", splits, smoothing=5.0)

        pd.testing.assert_series_equal(
            oof_before.iloc[first_val_idx].reset_index(drop=True),
            oof_after.iloc[first_val_idx].reset_index(drop=True),
        )
        # Sanity: some *other* fold's OOF values DID change, since fold 0's flipped
        # labels are part of their training data — otherwise this test would pass
        # trivially by the function ignoring the target column entirely.
        other_idx = np.concatenate([v for _, v in splits[1:]])
        assert not np.allclose(oof_before.iloc[other_idx], oof_after.iloc[other_idx])

    def test_overlapping_splits_raise(self):
        df = _synthetic_df()
        bad_splits = [
            (np.array([0, 1, 2]), np.array([3, 4])),
            (np.array([0, 1, 2]), np.array([4, 5])),  # index 4 covered twice
        ]
        with pytest.raises(ValueError, match="more than one val_idx"):
            enc.compute_oof_target_encoding(df.iloc[:6], "y", "cat", bad_splits)

    def test_train_val_overlap_within_fold_raises(self):
        """The check added by the code review: train_idx wrongly containing its own
        fold's val_idx must be refused, not silently run — this is the exact bug
        shape that would let a row's own label leak into its own encoding."""
        df = pd.DataFrame({"cat": [0, 0, 0, 1, 1, 1], "y": [0, 0, 0, 1, 1, 1]})
        bad_splits = [
            (np.array([0, 1, 2, 3, 4, 5]), np.array([0, 1, 2])),  # train_idx includes val_idx
            (np.array([0, 1, 2, 3, 4, 5]), np.array([3, 4, 5])),
        ]
        with pytest.raises(ValueError, match="train_idx and val_idx overlap"):
            enc.compute_oof_target_encoding(df, "y", "cat", bad_splits, smoothing=0.0)

    def test_incomplete_coverage_raises(self):
        df = _synthetic_df().iloc[:10].reset_index(drop=True)
        incomplete_splits = [(np.array([0, 1, 2, 3, 4]), np.array([5, 6, 7]))]  # rows 8,9 uncovered
        with pytest.raises(ValueError, match="never covered"):
            enc.compute_oof_target_encoding(df, "y", "cat", incomplete_splits)

    def test_real_data_respects_twin_grouped_tier1_splits(self):
        """Integration check with the real project data and the real (twin-record-
        safe) splitter this module is meant to be used with — confirms the plumbing
        works end to end, not just against a hand-rolled K-fold."""
        from climate_health.data.loaders import load_train_full
        from climate_health.evaluation.cv import tier1_splits

        train = load_train_full()
        splits = list(tier1_splits(train, target_col="is_climate_sensitive"))
        oof = enc.compute_oof_target_encoding(
            train, "is_climate_sensitive", "zone", splits, smoothing=10.0
        )
        assert oof.isna().sum() == 0
        assert len(oof) == len(train)


class TestOofInputValidation:
    def test_missing_column_raises_keyerror(self):
        with pytest.raises(KeyError):
            enc.compute_oof_target_encoding(pd.DataFrame({"y": [1]}), "y", "cat", [])

    def test_null_raises_valueerror(self):
        df = pd.DataFrame({"y": [1, np.nan], "cat": [0, 1]})
        with pytest.raises(ValueError, match="null values"):
            enc.compute_oof_target_encoding(df, "y", "cat", [])

    def test_negative_smoothing_raises(self):
        df = _synthetic_df()
        with pytest.raises(ValueError, match="smoothing"):
            enc.compute_oof_target_encoding(df, "y", "cat", [], smoothing=-1.0)
