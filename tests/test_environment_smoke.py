"""
Environment smoke test — Stage 1.

Not a placeholder: this is what actually verifies the pinned dependency set is
mutually compatible on this machine, rather than hoping the version ranges in
pyproject.toml resolve cleanly forever. Run this immediately after
`pip install -e ".[all]"`, before writing any pipeline code.

If something fails here, fix the pin in pyproject.toml now — it is far cheaper
than discovering a conflict in week 5 while tuning models.
"""

import numpy as np
import pandas as pd


def test_core_imports_and_versions():
    import matplotlib
    import pandera
    import scipy
    import seaborn
    import sklearn

    versions = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "pandera": pandera.__version__,
        "matplotlib": matplotlib.__version__,
        "seaborn": seaborn.__version__,
    }
    print("\nCore versions:", versions)
    assert np.__version__ < "2.2", "numpy must stay below 2.2 for SHAP/numba compatibility"


def test_model_zoo_imports_and_versions():
    import catboost
    import lightgbm
    import optuna
    import shap
    import xgboost

    versions = {
        "lightgbm": lightgbm.__version__,
        "xgboost": xgboost.__version__,
        "catboost": catboost.__version__,
        "optuna": optuna.__version__,
        "shap": shap.__version__,
    }
    print("\nModel zoo versions:", versions)


def test_tracking_and_serving_imports():
    import fastapi
    import mlflow
    import pydantic
    import streamlit

    versions = {
        "mlflow": mlflow.__version__,
        "pydantic": pydantic.__version__,
        "fastapi": fastapi.__version__,
        "streamlit": streamlit.__version__,
    }
    print("\nTracking/serving versions:", versions)


def _toy_data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def test_lightgbm_fit_predict_roundtrip():
    import lightgbm as lgb

    X, y = _toy_data()
    model = lgb.LGBMClassifier(n_estimators=10, verbosity=-1)
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (200,)


def test_xgboost_fit_predict_roundtrip():
    import xgboost as xgb

    X, y = _toy_data()
    model = xgb.XGBClassifier(n_estimators=10, eval_metric="logloss")
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (200,)


def test_catboost_fit_predict_roundtrip():
    from catboost import CatBoostClassifier

    X, y = _toy_data()
    model = CatBoostClassifier(iterations=10, verbose=False)
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (200,)


def test_sklearn_fit_predict_roundtrip():
    from sklearn.linear_model import LogisticRegression

    X, y = _toy_data()
    model = LogisticRegression().fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (200,)


def test_shap_roundtrip_on_lightgbm():
    import lightgbm as lgb
    import shap

    X, y = _toy_data()
    model = lgb.LGBMClassifier(n_estimators=10, verbosity=-1).fit(X, y)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X[:20])
    assert values is not None


def test_mlflow_roundtrip(tmp_path):
    import mlflow

    # MLflow 3.x put the plain filesystem tracking backend ("./mlruns") into
    # maintenance mode and pushes toward a real backend store — this is a real
    # finding from running this smoke test, not a hypothetical, so the project
    # standardizes on SQLite here rather than opting back into the deprecated
    # file store. See docs/PROJECT_BLUEPRINT.md Stage 1 recorded decisions.
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow_smoke_test.db")
    # Fluent API caches the active experiment id globally per-process — earlier
    # tests (test_tracking.py, test_baselines.py) set it against the real
    # mlflow.db, and that stale id doesn't exist in this fresh tmp_path store.
    # Pin the experiment explicitly so this test doesn't depend on suite order.
    mlflow.set_experiment("smoke_test")
    with mlflow.start_run():
        mlflow.log_param("smoke_test", True)
        mlflow.log_metric("dummy_score", 0.5)


def test_pandera_schema_roundtrip():
    import pandera.pandas as pa

    df = pd.DataFrame({"age": [1, 2, 3]})
    schema = pa.DataFrameSchema({"age": pa.Column(int, pa.Check.ge(0))})
    schema.validate(df)
