"""Prints every MLflow run's metrics as a table, sorted by tier2_mean.

Why this exists: MLflow's own web UI (mlflow 3.15.2, Windows) has a confirmed,
currently-open upstream bug where the page loads but never renders — see
https://github.com/mlflow/mlflow/issues/18620 and
https://github.com/mlflow/mlflow/issues/25747. Ruled out on this machine: wrong
JS MIME type (verified correct via a direct header check), browser/extension
interference (reproduced in both Chrome and Edge, including incognito), and the
disabled-by-default `/workspaces` endpoint (reproduced with `--enable-workspaces`
too). Rather than block on an unresolved third-party bug, this script reads the
same SQLite tracking store directly — no server, no browser, nothing that can
render blank.

Usage: python scripts/mlflow_report.py [experiment_name]
"""

from __future__ import annotations

import sys

import mlflow

from climate_health.tracking import DEFAULT_EXPERIMENT, TRACKING_URI

METRIC_COLUMNS = ("tier1_mean", "tier1_std", "tier2_mean", "tier2_std", "tier3_score")


def main(experiment_name: str = DEFAULT_EXPERIMENT) -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"No experiment named {experiment_name!r} found at {TRACKING_URI}")
        return

    runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"])
    # Keep only the most recent run per model name — repeated test/notebook runs
    # log duplicates under the same name, and only the latest reflects current code.
    latest_by_model: dict[str, object] = {}
    for run in runs:
        name = run.data.params.get("model", run.info.run_name)
        if name not in latest_by_model:
            latest_by_model[name] = run

    header = f"{'model':<25}" + "".join(f"{c:>14}" for c in METRIC_COLUMNS)
    print(header)
    print("-" * len(header))
    for name, run in sorted(
        latest_by_model.items(),
        key=lambda kv: kv[1].data.metrics.get("tier2_mean", -1),
        reverse=True,
    ):
        row = f"{name:<25}"
        for col in METRIC_COLUMNS:
            val = run.data.metrics.get(col)
            row += f"{val:>14.4f}" if val is not None else f"{'—':>14}"
        print(row)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EXPERIMENT)
