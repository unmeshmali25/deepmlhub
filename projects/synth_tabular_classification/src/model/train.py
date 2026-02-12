"""Train a classification model with MLflow tracking."""

import json
import yaml
import os
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import mlflow
import mlflow.sklearn


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parents[2] / "configs" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_mlflow_tracking_uri(config: dict, base_path: Path) -> str:
    """Get MLflow tracking URI from config, env var, or default to local."""
    # Priority: 1) Environment variable, 2) Config file, 3) Local default
    mlflow_config = config.get("mlflow", {})
    return os.getenv(
        "MLFLOW_TRACKING_URI",
        mlflow_config.get("tracking_uri", f"file://{base_path}/mlruns"),
    )


def train_model() -> tuple:
    """Train model on training data with MLflow tracking."""
    config = load_config()
    model_config = config["model"]
    mlflow_config = config["mlflow"]
    paths = config["paths"]
    base_path = Path(__file__).parents[2]

    # Set MLflow tracking URI
    mlflow_uri = get_mlflow_tracking_uri(config, base_path)
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])
    print(f"MLflow tracking URI: {mlflow_uri}")

    # Load training data
    train_path = base_path / paths["train_data"]
    print(f"Loading training data from {train_path}...")
    train_df = pd.read_csv(train_path)

    X_train = train_df.drop("target", axis=1)
    y_train = train_df["target"]

    # Start MLflow run
    with mlflow.start_run() as run:
        print(f"MLflow run ID: {run.info.run_id}")

        # Log parameters
        mlflow.log_param("model_type", model_config["type"])
        mlflow.log_param("n_estimators", model_config["n_estimators"])
        mlflow.log_param("max_depth", model_config["max_depth"])
        mlflow.log_param("n_samples", len(train_df))
        mlflow.log_param("n_features", X_train.shape[1])

        # Create and train model
        print(f"Training {model_config['type']} model...")
        model = RandomForestClassifier(
            n_estimators=model_config["n_estimators"],
            max_depth=model_config["max_depth"],
            random_state=model_config["random_seed"],
            n_jobs=-1,  # Use all CPUs
        )
        model.fit(X_train, y_train)

        # Calculate training metrics
        y_pred = model.predict(X_train)
        metrics = {
            "train_accuracy": accuracy_score(y_train, y_pred),
            "train_f1": f1_score(y_train, y_pred, average="weighted"),
            "train_precision": precision_score(y_train, y_pred, average="weighted"),
            "train_recall": recall_score(y_train, y_pred, average="weighted"),
            "n_samples": len(train_df),
            "n_features": X_train.shape[1],
        }

        # Log metrics to MLflow
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                mlflow.log_metric(metric_name, metric_value)

        print(f"Training accuracy: {metrics['train_accuracy']:.4f}")
        print(f"Training F1 score: {metrics['train_f1']:.4f}")

        # Log model to MLflow
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=mlflow_config["experiment_name"],
        )

        # Save model locally for DVC
        model_path = base_path / paths["model"]
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

        # Save metrics JSON for DVC
        metrics_path = base_path / paths["metrics"]
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # Log metrics file as artifact
        mlflow.log_artifact(str(metrics_path))

    return model, metrics


if __name__ == "__main__":
    train_model()
