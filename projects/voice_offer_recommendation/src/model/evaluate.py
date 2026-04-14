"""Evaluate recommendation model on test/holdout data."""

import json
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

import joblib


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parents[2] / "configs" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Calculate precision@k."""
    if len(y_pred) == 0:
        return 0.0
    top_k = y_pred[:k]
    relevant = np.sum(y_true[np.isin(y_true, top_k)])
    return relevant / k


def recall_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Calculate recall@k."""
    if len(y_true) == 0:
        return 0.0
    top_k = y_pred[:k]
    relevant = np.sum(y_true[np.isin(y_true, top_k)])
    return relevant / len(y_true)


def evaluate_model() -> dict:
    """Load model and evaluate on holdout data."""
    config = load_config()
    mlflow_config = config["mlflow"]
    paths = config["paths"]
    base_path = Path(__file__).parents[2]

    # Set MLflow tracking URI
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", f"file://{base_path}/mlruns")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])

    # Load model
    model_path = base_path / paths["model"]
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    # Load training data and create a temporal/train-test split
    train_path = base_path / paths["training_features"]
    print(f"Loading data from {train_path}...")
    df = pd.read_parquet(train_path)

    # Simple split by agents (20% holdout agents)
    agent_ids = df["agent_id"].unique()
    train_agents, test_agents = train_test_split(
        agent_ids, test_size=config["data"]["test_size"], random_state=42
    )

    test_df = df[df["agent_id"].isin(test_agents)]
    print(
        f"Evaluating on {len(test_agents)} holdout agents ({len(test_df)} interactions)"
    )

    # Evaluate each test agent
    k_values = [5, 10]
    results = {f"precision@{k}": [] for k in k_values}
    results.update({f"recall@{k}": [] for k in k_values})

    for agent_id in test_agents[:100]:  # Limit to 100 agents for speed
        agent_data = test_df[test_df["agent_id"] == agent_id]
        y_true = agent_data[agent_data["purchased"] == 1]["product_id"].values

        if len(y_true) == 0:
            continue

        predictions = model.predict(agent_id)
        y_pred = predictions["product_id"].values

        for k in k_values:
            results[f"precision@{k}"].append(precision_at_k(y_true, y_pred, k))
            results[f"recall@{k}"].append(recall_at_k(y_true, y_pred, k))

    # Aggregate metrics
    test_metrics = {}
    for key, values in results.items():
        test_metrics[key] = float(np.mean(values)) if values else 0.0

    print(f"\n{'=' * 40}")
    print("TEST RESULTS")
    print(f"{'=' * 40}")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")

    # Log to MLflow
    with mlflow.start_run():
        for metric_name, metric_value in test_metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        mlflow.set_tag("stage", "evaluation")

        # Save evaluation report
        report_lines = [
            "Voice Offer Recommendation - Evaluation Report\n",
            "=" * 40 + "\n",
        ]
        for key, value in test_metrics.items():
            report_lines.append(f"{key}: {value:.4f}\n")

        report_path = base_path / "metrics" / "evaluation_report.txt"
        with open(report_path, "w") as f:
            f.writelines(report_lines)
        mlflow.log_artifact(str(report_path))

    # Update metrics JSON
    metrics_path = base_path / paths["metrics"]
    with open(metrics_path) as f:
        metrics = json.load(f)

    metrics.update(test_metrics)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    evaluate_model()
