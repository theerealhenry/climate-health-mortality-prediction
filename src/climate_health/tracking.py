"""MLflow tracking setup for the climate-health pipeline.

Every stage from Phase 4 onward (baselines, model zoo, tuning, ensembling) logs
its runs through MLflow. MLflow 3.x deprecated the plain filesystem (`file://`)
tracking backend, so this project standardizes on a SQLite store at the repo
root instead: `sqlite:///mlflow.db` (recorded decision, see
docs/PROJECT_BLUEPRINT.md, Stage 1). `mlflow.db` and `mlflow_artifacts/` are
gitignored, same as `mlruns/`.

Call `configure_tracking()` explicitly before any `mlflow.start_run()` — deliberately
not run at import time, since an import-time side effect (creating mlflow.db,
touching disk) is surprising for anything that imports this module incidentally
(e.g. pytest collection). It is idempotent — calling it more than once just
re-sets the same URI.
"""

from pathlib import Path

import mlflow

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKING_URI = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"
DEFAULT_EXPERIMENT = "climate-health"


def configure_tracking(experiment_name: str = DEFAULT_EXPERIMENT) -> None:
    """Point MLflow at the project's SQLite tracking store and select the
    experiment every stage logs into. Safe to call repeatedly."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment_name)
