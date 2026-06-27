"""
Automated MLflow retraining pipeline.
Can be triggered on a schedule (cron) or via API endpoint.
"""

from pathlib import Path
import pandas as pd
import mlflow
from loguru import logger
from datetime import datetime

from data.processing.preprocessor import load_children_master, clean_children
from data.processing.feature_engineering import engineer_dropout_features
from models.dropout_prediction.predictor import run_training_pipeline


def retrain(experiment_name: str = "EduSentinel_Dropout_Prediction") -> dict:
    logger.info("=== EduSentinel Dropout Model Retraining Pipeline ===")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")

    children_raw = load_children_master()
    children_clean = clean_children(children_raw)
    children_feat = engineer_dropout_features(children_clean)

    metrics = run_training_pipeline(children_feat, experiment_name=experiment_name)

    logger.success(f"Retraining complete. Metrics: {metrics}")
    return metrics


def list_experiments() -> list[dict]:
    mlflow.set_tracking_uri("./mlruns")
    client = mlflow.tracking.MlflowClient()
    exps = client.search_experiments()
    return [
        {
            "name": e.name,
            "experiment_id": e.experiment_id,
            "lifecycle_stage": e.lifecycle_stage,
        }
        for e in exps
    ]


def get_best_run(experiment_name: str = "EduSentinel_Dropout_Prediction") -> dict | None:
    mlflow.set_tracking_uri("./mlruns")
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if not exp:
        return None
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.test_roc_auc DESC"],
        max_results=1,
    )
    if not runs:
        return None
    r = runs[0]
    return {
        "run_id": r.info.run_id,
        "start_time": r.info.start_time,
        "metrics": r.data.metrics,
        "params": r.data.params,
    }


if __name__ == "__main__":
    retrain()
