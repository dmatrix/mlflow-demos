"""
MLflow setup for the devconnect demo agents.
"""

import mlflow
from typing import Optional


def setup_mlflow_tracking(
    experiment_name: str,
    tracking_uri: Optional[str] = "http://localhost:5000",
    enable_autolog: bool = True,
) -> None:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if enable_autolog:
        try:
            mlflow.openai.autolog()
        except Exception:
            pass
    mlflow.set_experiment(experiment_name)
