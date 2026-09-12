"""Smoke test for MLflow tracking setup (Stage 10.1).

Verifies configure_tracking() points MLflow at the project's sqlite store and
that a run actually gets logged there, per the blueprint's Stage 1 recorded
decision (file:// backend is deprecated in MLflow 3.x).
"""

import mlflow

from climate_health.tracking import REPO_ROOT, TRACKING_URI, configure_tracking


def test_configure_tracking_sets_sqlite_uri():
    configure_tracking()
    assert mlflow.get_tracking_uri() == TRACKING_URI
    assert TRACKING_URI.startswith("sqlite:///")


def test_smoke_run_is_logged_to_mlflow_db():
    configure_tracking()
    with mlflow.start_run(run_name="stage10_smoke_test") as run:
        mlflow.log_param("smoke_test", True)

    db_path = REPO_ROOT / "mlflow.db"
    assert db_path.exists(), "mlflow.db was not created by the smoke run"

    client = mlflow.tracking.MlflowClient()
    logged = client.get_run(run.info.run_id)
    assert logged.data.params.get("smoke_test") == "True"


if __name__ == "__main__":
    test_configure_tracking_sets_sqlite_uri()
    test_smoke_run_is_logged_to_mlflow_db()
    print("OK: mlflow tracking smoke test passed")
