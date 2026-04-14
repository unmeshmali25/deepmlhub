"""Train a recommendation model with MLflow tracking."""

import json
import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import yaml

from src.model.recommendation_algorithm import get_recommender


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parents[2] / "configs" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_mlflow_tracking_uri(config: dict, base_path: Path) -> str:
    """Get MLflow tracking URI from config, env var, or default to local."""
    mlflow_config = config.get("mlflow", {})
    return os.getenv(
        "MLFLOW_TRACKING_URI",
        mlflow_config.get("tracking_uri", f"file://{base_path}/mlruns"),
    )


def train_model() -> tuple:
    """Train recommendation model on training data with MLflow tracking."""
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
    train_path = base_path / paths["training_features"]
    print(f"Loading training data from {train_path}...")
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}. "
            "Run: python -m src.data.fetch_features"
        )

    train_df = pd.read_parquet(train_path)
    print(f"Loaded {len(train_df)} training rows")

    # Start MLflow run
    with mlflow.start_run() as run:
        print(f"MLflow run ID: {run.info.run_id}")

        # Log parameters
        mlflow.log_param("model_type", model_config["type"])
        mlflow.log_param("n_samples", len(train_df))
        mlflow.log_param("n_features", len(train_df.columns))

        # Get algorithm-specific params
        algo_params = model_config.get(model_config["type"], {})
        for param_name, param_value in algo_params.items():
            mlflow.log_param(param_name, param_value)

        # Create and train model
        print(f"Training {model_config['type']} model...")
        model = get_recommender(model_config["type"], **algo_params)
        model.fit(train_df)

        # Calculate a simple training metric (coverage = fraction of products scored)
        sample_agent = train_df["agent_id"].iloc[0]
        predictions = model.predict(sample_agent)
        coverage = len(predictions) / train_df["product_id"].nunique()

        metrics = {
            "train_coverage": float(coverage),
            "n_samples": len(train_df),
            "n_products": train_df["product_id"].nunique(),
            "n_agents": train_df["agent_id"].nunique(),
        }

        # Log metrics to MLflow
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(metric_name, metric_value)

        print(f"Training coverage: {metrics['train_coverage']:.4f}")

        # Save model locally for DVC
        model_path = base_path / paths["model"]
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

        # Log model to MLflow
        mlflow.log_artifact(str(model_path))

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
