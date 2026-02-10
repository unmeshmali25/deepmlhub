"""Evaluate model on test data with MLflow tracking."""

import json
import yaml
import os
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

import mlflow


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parents[2] / "configs" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def evaluate_model() -> dict:
    """Load model and evaluate on test data."""
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

    # Load test data
    test_path = base_path / paths["test_data"]
    print(f"Loading test data from {test_path}...")
    test_df = pd.read_csv(test_path)

    X_test = test_df.drop("target", axis=1)
    y_test = test_df["target"]

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    test_metrics = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred, average="weighted"),
        "test_precision": precision_score(y_test, y_pred, average="weighted"),
        "test_recall": recall_score(y_test, y_pred, average="weighted"),
    }

    print(f"\n{'=' * 40}")
    print("TEST RESULTS")
    print(f"{'=' * 40}")
    print(f"Accuracy:  {test_metrics['test_accuracy']:.4f}")
    print(f"F1 Score:  {test_metrics['test_f1']:.4f}")
    print(f"Precision: {test_metrics['test_precision']:.4f}")
    print(f"Recall:    {test_metrics['test_recall']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Log to MLflow
    with mlflow.start_run():
        for metric_name, metric_value in test_metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        mlflow.set_tag("stage", "evaluation")

        # Save classification report as artifact
        report = classification_report(y_test, y_pred)
        report_path = base_path / "metrics" / "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(str(report_path))

        # Save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_path = base_path / "metrics" / "confusion_matrix.txt"
        with open(cm_path, "w") as f:
            f.write(str(cm))
        mlflow.log_artifact(str(cm_path))

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
