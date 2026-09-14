"""Stage 14.3 (stretch) — does the weighted-average blend's Stage 14.2 edge over
standalone `catboost_tuned` (+0.0010 on one 5-fold OOF partition) survive repeated
Tier 2 CV, or is it noise?

Per the Stage 14.3 idea-refine one-pager (docs/ideas/stage14-3-repeated-cv-validation.md):
  - Blend weights are FIXED at Stage 14.2's optimized values, not re-optimized per fold —
    this validates the specific blend as it would actually be deployed.
  - Evaluated on the same 24 non-locked Tier 2 folds `catboost_tuned` was tuned/scored on
    in Stage 13.3 (tune.py's `non_locked_tier2_splits`) — apples-to-apples, and the locked
    (repeat=0, fold=0) holdout stays untouched per ADR-001.
  - `catboost_tuned` is re-scored fold-by-fold here too (Stage 13.3 only ever recorded the
    24-fold *mean*, not per-fold scores — see tune.py's objective(), which discards
    `scores` after summarizing), so the two are paired per fold, not just compared as
    aggregate means.

Run from the repo root:  python scripts/run_stage14_3_repeated_cv_validation.py
"""

from __future__ import annotations

import catboost as cb
import numpy as np
import yaml
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from climate_health.data.loaders import load_train_full
from climate_health.features.pipeline import branch_b_encoded, build_feature_matrix
from climate_health.models.baselines import RANDOM_STATE, TARGET_COL, competition_score
from climate_health.models.ensemble import weighted_average
from climate_health.models.tune import build_tuning_frame, non_locked_tier2_splits

# Stage 14.2's optimized weights (fixed here, not re-fit per fold — see module docstring).
BLEND_WEIGHTS = {
    "catboost_tuned": 0.6701,
    "logistic_regression_v2": 0.1658,
    "extra_trees": 0.1398,
    "histgradientboosting": 0.0243,
}

with open("configs/model_best.yaml") as f:
    CATBOOST_TUNED_PARAMS = yaml.safe_load(f)["params"]


def make_branch_b_models():
    return {
        "extra_trees": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(n_estimators=300, random_state=RANDOM_STATE),
        ),
        "histgradientboosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        "logistic_regression_v2": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        ),
    }


def main() -> None:
    df = load_train_full()

    # Branch A frame (catboost_tuned's native-categorical representation, Stage 13's own).
    combined_a, feature_cols_a, cat_idx = build_tuning_frame(df)
    X_a, y = combined_a[feature_cols_a].values, combined_a[TARGET_COL].values

    # Branch B frame (the other three candidates' one-hot/target-encoded representation).
    X_full, _ = build_feature_matrix(df, fit=True, target_col=TARGET_COL)
    branch_b_df = branch_b_encoded(X_full, target_col=TARGET_COL)
    combined_b = X_full.copy()
    for col in branch_b_df.columns:
        combined_b[col] = branch_b_df[col].values
    feature_cols_b = list(branch_b_df.columns)
    X_b = combined_b[feature_cols_b].values

    catboost_scores, blend_scores = [], []
    # non_locked_tier2_splits needs `spatial_cluster` (added by build_feature_matrix, in
    # combined_a) — passing raw `df` here raises KeyError: 'spatial_cluster' not found.
    splits = non_locked_tier2_splits(
        combined_a, spatial_cluster_col="spatial_cluster", random_state=RANDOM_STATE
    )
    for split in splits:
        y_val = y[split.val_idx]

        catboost_pred = (
            cb.CatBoostClassifier(**CATBOOST_TUNED_PARAMS, cat_features=cat_idx, verbose=False)
            .fit(X_a[split.train_idx], y[split.train_idx])
            .predict_proba(X_a[split.val_idx])[:, 1]
        )

        oof_preds = {"catboost_tuned": catboost_pred}
        for name, model in make_branch_b_models().items():
            model.fit(X_b[split.train_idx], y[split.train_idx])
            oof_preds[name] = model.predict_proba(X_b[split.val_idx])[:, 1]

        catboost_scores.append(competition_score(y_val, catboost_pred))
        blend_scores.append(competition_score(y_val, weighted_average(oof_preds, BLEND_WEIGHTS)))
        print(
            f"fold {split.repeat}/{split.fold}: "
            f"catboost={catboost_scores[-1]:.4f}  blend={blend_scores[-1]:.4f}"
        )

    catboost_scores, blend_scores = np.array(catboost_scores), np.array(blend_scores)
    diffs = blend_scores - catboost_scores

    print("\n=== Stage 14.3 — 24-fold paired comparison ===")
    print(f"catboost_tuned : mean={catboost_scores.mean():.4f}  std={catboost_scores.std():.4f}")
    print(f"blend          : mean={blend_scores.mean():.4f}  std={blend_scores.std():.4f}")
    print(f"paired diff (blend - catboost): mean={diffs.mean():.4f}  std={diffs.std():.4f}")
    print(f"folds where blend beat catboost: {(diffs > 0).sum()}/{len(diffs)}")


if __name__ == "__main__":
    main()
