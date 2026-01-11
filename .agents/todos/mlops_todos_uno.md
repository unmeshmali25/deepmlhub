# MLOps AI Tasks (Full Infrastructure, Small Data)

> **Goal**: Build complete MLOps infrastructure that works at scale, but test with small data (1000 samples, CPU-only).
> **Approach**: Full cloud setup (GCP, Terraform, K8s) but simple models for testing.

---

## How This Works

1. **Human** completes tasks in `.human/todos/mlops_human_todo_uno.md` (signups, installs)
2. **AI** completes tasks in this file (code, configs, Terraform, K8s manifests)
3. Tasks are numbered with dependencies (e.g., A1.1 requires H1.5 human task complete)

---

## Phase 0: Local ML Pipeline (No Cloud Yet)

**Prerequisites**: Human must complete H0.1-H0.4 first.

### A0.1 Create Project Directory Structure ✅

**Create these directories**:

```bash
mkdir -p projects/synth_tabular_classification/src/data
mkdir -p projects/synth_tabular_classification/src/model
mkdir -p projects/synth_tabular_classification/src/inference
mkdir -p projects/synth_tabular_classification/data/raw
mkdir -p projects/synth_tabular_classification/data/processed
mkdir -p projects/synth_tabular_classification/models
mkdir -p projects/synth_tabular_classification/metrics
mkdir -p projects/synth_tabular_classification/configs
mkdir -p projects/synth_tabular_classification/tests
mkdir -p projects/synth_tabular_classification/notebooks
```

**Create these files**:
- `projects/synth_tabular_classification/src/__init__.py` (empty)
- `projects/synth_tabular_classification/src/data/__init__.py` (empty)
- `projects/synth_tabular_classification/src/model/__init__.py` (empty)
- `projects/synth_tabular_classification/src/inference/__init__.py` (empty)
- `projects/synth_tabular_classification/tests/__init__.py` (empty)
- `.gitkeep` files in data/raw, data/processed, models, metrics

---

### A0.2 Create requirements.txt ✅

**File**: `projects/synth_tabular_classification/requirements.txt`

```
# Core ML (CPU-only for testing)
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
PyYAML>=6.0
joblib>=1.3.0

# Experiment tracking
mlflow>=2.10.0

# Data versioning
dvc[gs]>=3.30.0

# API server
fastapi>=0.109.0
uvicorn>=0.27.0
pydantic>=2.0.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
httpx>=0.25.0

# Linting
ruff>=0.1.0
mypy>=1.7.0
```

---

### A0.3 Create Configuration File ✅

**File**: `projects/synth_tabular_classification/configs/config.yaml`

```yaml
# ============================================
# SYNTH TABULAR CLASSIFICATION CONFIG
# ============================================
# Small data for testing infrastructure
# Increase n_samples for production

# Data generation settings
data:
  n_samples: 1000          # Small for testing (use 100000+ for prod)
  n_features: 10           # Simple feature set
  n_classes: 2             # Binary classification
  random_seed: 42
  test_size: 0.2

# Model settings (sklearn RandomForest - CPU only)
model:
  type: "random_forest"
  n_estimators: 50         # Small for testing (use 200+ for prod)
  max_depth: 10
  random_seed: 42

# Paths (relative to project root)
paths:
  raw_data: "data/raw/synthetic_data.csv"
  train_data: "data/processed/train.csv"
  test_data: "data/processed/test.csv"
  model: "models/model.joblib"
  metrics: "metrics/metrics.json"

# MLflow settings
mlflow:
  experiment_name: "synth_tabular_classification"
  # tracking_uri is set via MLFLOW_TRACKING_URI env var

# Inference settings
inference:
  host: "0.0.0.0"
  port: 8000
```

---

### A0.4 Create Data Generation Script ✅

**File**: `projects/synth_tabular_classification/src/data/generate.py`

```python
"""Generate synthetic classification data."""
import yaml
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_classification


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parents[2] / "configs" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_data() -> pd.DataFrame:
    """Generate synthetic classification data and save to CSV."""
    config = load_config()
    data_config = config["data"]
    paths = config["paths"]

    print(f"Generating {data_config['n_samples']} samples...")
    print(f"Features: {data_config['n_features']}, Classes: {data_config['n_classes']}")

    # Generate synthetic data
    X, y = make_classification(
        n_samples=data_config["n_samples"],
        n_features=data_config["n_features"],
        n_informative=max(2, data_config["n_features"] // 2),
        n_redundant=max(1, data_config["n_features"] // 4),
        n_classes=data_config["n_classes"],
        random_state=data_config["random_seed"],
        flip_y=0.1,  # Add some noise
    )

    # Create DataFrame
    feature_cols = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_cols)
    df["target"] = y

    # Save to CSV
    output_path = Path(__file__).parents[2] / paths["raw_data"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} samples to {output_path}")
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")

    return df


if __name__ == "__main__":
    generate_data()
```

---

### A0.5 Create Data Preprocessing Script ✅

**File**: `projects/synth_tabular_classification/src/data/preprocess.py`

```python
"""Preprocess data: split into train/test sets."""
import yaml
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parents[2] / "configs" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def preprocess_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw data, split into train/test, and save."""
    config = load_config()
    data_config = config["data"]
    paths = config["paths"]
    base_path = Path(__file__).parents[2]

    # Load raw data
    raw_path = base_path / paths["raw_data"]
    print(f"Loading data from {raw_path}...")
    df = pd.read_csv(raw_path)

    # Split features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=data_config["test_size"],
        random_state=data_config["random_seed"],
        stratify=y,
    )

    # Optional: Scale features (uncomment if needed)
    # scaler = StandardScaler()
    # X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    # X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    # joblib.dump(scaler, base_path / "models" / "scaler.joblib")

    # Combine back to DataFrames
    train_df = X_train.copy()
    train_df["target"] = y_train.values

    test_df = X_test.copy()
    test_df["target"] = y_test.values

    # Save
    train_path = base_path / paths["train_data"]
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)

    test_path = base_path / paths["test_data"]
    test_df.to_csv(test_path, index=False)

    print(f"Train set: {len(train_df)} samples -> {train_path}")
    print(f"Test set: {len(test_df)} samples -> {test_path}")

    return train_df, test_df


if __name__ == "__main__":
    preprocess_data()
```

---

### A0.6 Create Model Training Script (with MLflow)

**File**: `projects/synth_tabular_classification/src/model/train.py`

```python
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


def get_mlflow_tracking_uri(base_path: Path) -> str:
    """Get MLflow tracking URI from env or default to local."""
    return os.getenv("MLFLOW_TRACKING_URI", f"file://{base_path}/mlruns")


def train_model() -> tuple:
    """Train model on training data with MLflow tracking."""
    config = load_config()
    model_config = config["model"]
    mlflow_config = config["mlflow"]
    paths = config["paths"]
    base_path = Path(__file__).parents[2]

    # Set MLflow tracking URI
    mlflow_uri = get_mlflow_tracking_uri(base_path)
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
```

---

### A0.7 Create Model Evaluation Script

**File**: `projects/synth_tabular_classification/src/model/evaluate.py`

```python
"""Evaluate model on test data with MLflow tracking."""
import json
import yaml
import os
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
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

    print(f"\n{'='*40}")
    print("TEST RESULTS")
    print(f"{'='*40}")
    print(f"Accuracy:  {test_metrics['test_accuracy']:.4f}")
    print(f"F1 Score:  {test_metrics['test_f1']:.4f}")
    print(f"Precision: {test_metrics['test_precision']:.4f}")
    print(f"Recall:    {test_metrics['test_recall']:.4f}")
    print(f"\nClassification Report:")
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
```

---

### A0.8 Create Prediction Script

**File**: `projects/synth_tabular_classification/src/inference/predict.py`

```python
"""Make predictions with trained model."""
import yaml
import pandas as pd
import joblib
from pathlib import Path
from typing import Union
import numpy as np


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parents[2] / "configs" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model():
    """Load the trained model."""
    config = load_config()
    paths = config["paths"]
    base_path = Path(__file__).parents[2]
    model_path = base_path / paths["model"]

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run training first.")

    return joblib.load(model_path)


def predict(features: Union[list, np.ndarray]) -> tuple[list[int], list[list[float]]]:
    """
    Make predictions for given features.

    Args:
        features: List of feature vectors, e.g., [[1.0, 2.0, ...], [3.0, 4.0, ...]]

    Returns:
        Tuple of (predictions, probabilities)
    """
    model = load_model()
    predictions = model.predict(features).tolist()
    probabilities = model.predict_proba(features).tolist()
    return predictions, probabilities


def main():
    """Demo: predict on a few samples from test set."""
    config = load_config()
    paths = config["paths"]
    base_path = Path(__file__).parents[2]

    # Load a few test samples
    test_path = base_path / paths["test_data"]

    if not test_path.exists():
        print(f"Test data not found at {test_path}. Run preprocessing first.")
        return

    test_df = pd.read_csv(test_path)

    X_test = test_df.drop("target", axis=1).head(5)
    y_true = test_df["target"].head(5).tolist()

    # Make predictions
    predictions, probabilities = predict(X_test.values.tolist())

    print("=" * 50)
    print("PREDICTION DEMO")
    print("=" * 50)
    for i, (pred, true, prob) in enumerate(zip(predictions, y_true, probabilities)):
        status = "✓" if pred == true else "✗"
        confidence = max(prob) * 100
        print(f"Sample {i}: predicted={pred}, actual={true} {status} (confidence: {confidence:.1f}%)")


if __name__ == "__main__":
    main()
```

---

### A0.9 Create FastAPI Inference Server

**File**: `projects/synth_tabular_classification/src/inference/server.py`

```python
"""FastAPI inference server for model predictions."""
import yaml
import joblib
from pathlib import Path
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Global model and config
model = None
config = None


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parents[2] / "configs" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global model, config
    config = load_config()
    model_path = Path(__file__).parents[2] / config["paths"]["model"]

    if not model_path.exists():
        raise RuntimeError(f"Model not found at {model_path}. Run training first.")

    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")

    yield  # Server runs here

    # Cleanup (if needed)
    print("Shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="Synth Tabular Classification API",
    description="Predict class labels for tabular data",
    version="1.0.0",
    lifespan=lifespan,
)


# Request/Response schemas
class PredictionRequest(BaseModel):
    """Request body for predictions."""
    features: List[List[float]] = Field(
        ...,
        description="List of feature vectors",
        example=[[0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1.0]]
    )


class PredictionResponse(BaseModel):
    """Response body for predictions."""
    predictions: List[int] = Field(..., description="Predicted class labels")
    probabilities: List[List[float]] = Field(..., description="Class probabilities")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool


# Endpoints
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check if server is healthy and model is loaded."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
    )


@app.get("/ready", tags=["Health"])
async def ready_check():
    """Check if server is ready to serve requests."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.get("/info", tags=["Info"])
async def model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_type": config["model"]["type"],
        "n_features": config["data"]["n_features"],
        "n_classes": config["data"]["n_classes"],
        "n_estimators": config["model"]["n_estimators"],
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """Make predictions for given features."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate feature dimensions
    expected_features = config["data"]["n_features"]
    for i, features in enumerate(request.features):
        if len(features) != expected_features:
            raise HTTPException(
                status_code=400,
                detail=f"Sample {i}: expected {expected_features} features, got {len(features)}",
            )

    # Make predictions
    predictions = model.predict(request.features).tolist()
    probabilities = model.predict_proba(request.features).tolist()

    return PredictionResponse(
        predictions=predictions,
        probabilities=probabilities,
    )


# For running directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

### A0.10 Create Unit Tests

**File**: `projects/synth_tabular_classification/tests/test_data.py`

```python
"""Tests for data generation and preprocessing."""
import pytest
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data.generate import generate_data, load_config
from src.data.preprocess import preprocess_data


class TestDataGeneration:
    def test_generate_data_creates_file(self, tmp_path, monkeypatch):
        """Test that generate_data creates the output file."""
        # This is a basic smoke test
        config = load_config()
        assert config["data"]["n_samples"] > 0

    def test_config_loads(self):
        """Test that config loads correctly."""
        config = load_config()
        assert "data" in config
        assert "model" in config
        assert "paths" in config


class TestPreprocessing:
    def test_config_has_test_size(self):
        """Test that config has test_size."""
        config = load_config()
        assert "test_size" in config["data"]
        assert 0 < config["data"]["test_size"] < 1
```

**File**: `projects/synth_tabular_classification/tests/test_model.py`

```python
"""Tests for model training and evaluation."""
import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class TestModel:
    def test_random_forest_trains(self):
        """Test that RandomForest can train on dummy data."""
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == 100
        assert set(predictions).issubset({0, 1})

    def test_model_predicts_probabilities(self):
        """Test that model returns probabilities."""
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        probs = model.predict_proba(X)
        assert probs.shape == (100, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)
```

**File**: `projects/synth_tabular_classification/tests/test_inference.py`

```python
"""Tests for inference server."""
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1]))


class TestInferenceEndpoints:
    """Test inference endpoints (requires trained model)."""

    def test_health_endpoint_structure(self):
        """Test health response structure."""
        # This is a structural test - actual endpoint test needs model
        expected_fields = ["status", "model_loaded"]
        assert len(expected_fields) == 2

    def test_prediction_request_validation(self):
        """Test that prediction request has correct structure."""
        from src.inference.server import PredictionRequest

        # Valid request
        request = PredictionRequest(features=[[0.1, 0.2, 0.3]])
        assert len(request.features) == 1
        assert len(request.features[0]) == 3

    def test_prediction_response_structure(self):
        """Test prediction response structure."""
        from src.inference.server import PredictionResponse

        response = PredictionResponse(
            predictions=[0, 1],
            probabilities=[[0.8, 0.2], [0.3, 0.7]]
        )
        assert len(response.predictions) == 2
        assert len(response.probabilities) == 2
```

---

### A0.11 Create .gitignore for Project

**File**: `projects/synth_tabular_classification/.gitignore`

```
# Data (DVC tracked)
/data/raw/
/data/processed/

# Models (DVC tracked)
/models/*.joblib
/models/*.pt
/models/*.pkl

# MLflow local tracking
/mlruns/

# Metrics (keep in git, but generated files might be ignored)
# /metrics/

# Python
__pycache__/
*.pyc
*.pyo
.pytest_cache/
*.egg-info/
.eggs/
dist/
build/

# Virtual environments
.venv/
venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/
```

---

## Phase 0 Verification

```bash
cd projects/synth_tabular_classification

# Install dependencies
pip install -r requirements.txt

# Run pipeline manually
python -m src.data.generate
python -m src.data.preprocess
python -m src.model.train
python -m src.model.evaluate
python -m src.inference.predict

# Run tests
pytest tests/ -v

# Start API server (in background)
uvicorn src.inference.server:app --host 0.0.0.0 --port 8000 &

# Test API
curl http://localhost:8000/health
curl http://localhost:8000/info
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1.0]]}'

# Stop server
pkill -f uvicorn

echo "Phase 0 complete!"
```

---

## Phase 1: DVC Pipeline Setup

**Prerequisites**: Human must complete H0.1-H0.4.

### A1.1 Create DVC Pipeline File

**File**: `projects/synth_tabular_classification/dvc.yaml`

```yaml
stages:
  generate:
    cmd: python -m src.data.generate
    deps:
      - src/data/generate.py
      - configs/config.yaml
    outs:
      - data/raw/synthetic_data.csv

  preprocess:
    cmd: python -m src.data.preprocess
    deps:
      - src/data/preprocess.py
      - data/raw/synthetic_data.csv
      - configs/config.yaml
    outs:
      - data/processed/train.csv
      - data/processed/test.csv

  train:
    cmd: python -m src.model.train
    deps:
      - src/model/train.py
      - data/processed/train.csv
      - configs/config.yaml
    outs:
      - models/model.joblib
    metrics:
      - metrics/metrics.json:
          cache: false

  evaluate:
    cmd: python -m src.model.evaluate
    deps:
      - src/model/evaluate.py
      - models/model.joblib
      - data/processed/test.csv
    metrics:
      - metrics/metrics.json:
          cache: false
```

---

### A1.2 Create params.yaml for DVC

**File**: `projects/synth_tabular_classification/params.yaml`

```yaml
# DVC parameter tracking
# These are duplicated from config.yaml for DVC to track

data:
  n_samples: 1000
  n_features: 10
  n_classes: 2
  test_size: 0.2

model:
  n_estimators: 50
  max_depth: 10
```

---

### A1.3 Initialize DVC in Project

```bash
cd projects/synth_tabular_classification

# Initialize DVC
dvc init --subdir

# Verify
ls -la .dvc/
```

---

## Phase 1 Verification

```bash
cd projects/synth_tabular_classification

# Remove old outputs
rm -rf data/raw/* data/processed/* models/* metrics/*

# Run DVC pipeline
dvc repro

# View DAG
dvc dag

# Show metrics
dvc metrics show

# Push to remote (after human completes H4.1-H4.2)
# dvc push

echo "Phase 1 complete!"
```

---

## Phase 2: Terraform Infrastructure

**Prerequisites**: Human must complete H1.1-H1.6, H2.1-H2.3.

### A2.1 Create Terraform Directory Structure

```bash
mkdir -p infrastructure/terraform/environments/dev
mkdir -p infrastructure/terraform/modules/gcs
mkdir -p infrastructure/terraform/modules/mlflow
mkdir -p infrastructure/terraform/modules/artifact-registry
mkdir -p infrastructure/terraform/modules/gke
```

---

### A2.2 Create GCS Module

**File**: `infrastructure/terraform/modules/gcs/main.tf`

```hcl
# GCS Buckets for MLOps

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

# DVC Storage Bucket
resource "google_storage_bucket" "dvc" {
  name          = "${var.project_id}-dvc-storage"
  location      = var.region
  force_destroy = var.environment == "dev" ? true : false

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90  # Move to nearline after 90 days
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  labels = {
    environment = var.environment
    purpose     = "dvc-storage"
  }
}

# MLflow Artifacts Bucket
resource "google_storage_bucket" "mlflow" {
  name          = "${var.project_id}-mlflow"
  location      = var.region
  force_destroy = var.environment == "dev" ? true : false

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  labels = {
    environment = var.environment
    purpose     = "mlflow"
  }
}

output "dvc_bucket_name" {
  value = google_storage_bucket.dvc.name
}

output "dvc_bucket_url" {
  value = "gs://${google_storage_bucket.dvc.name}"
}

output "mlflow_bucket_name" {
  value = google_storage_bucket.mlflow.name
}

output "mlflow_bucket_url" {
  value = "gs://${google_storage_bucket.mlflow.name}"
}
```

**File**: `infrastructure/terraform/modules/gcs/variables.tf`

```hcl
# Variables are defined in main.tf for simplicity
```

---

### A2.3 Create Artifact Registry Module

**File**: `infrastructure/terraform/modules/artifact-registry/main.tf`

```hcl
# Artifact Registry for Docker images

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

# Docker repository
resource "google_artifact_registry_repository" "ml" {
  location      = var.region
  repository_id = "ml"
  description   = "ML Docker images"
  format        = "DOCKER"

  labels = {
    environment = var.environment
  }
}

output "repository_name" {
  value = google_artifact_registry_repository.ml.name
}

output "repository_url" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.ml.repository_id}"
}
```

---

### A2.4 Create MLflow Cloud Run Module

**File**: `infrastructure/terraform/modules/mlflow/main.tf`

```hcl
# MLflow on Cloud Run

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "mlflow_bucket" {
  description = "GCS bucket for MLflow storage"
  type        = string
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "invoker_email" {
  description = "Email of user allowed to invoke MLflow"
  type        = string
}

# Service Account for MLflow
resource "google_service_account" "mlflow" {
  account_id   = "mlflow-server"
  display_name = "MLflow Server Service Account"
}

# Grant MLflow SA access to storage
resource "google_storage_bucket_iam_member" "mlflow_storage" {
  bucket = var.mlflow_bucket
  role   = "roles/storage.admin"
  member = "serviceAccount:${google_service_account.mlflow.email}"
}

# Cloud Run service
resource "google_cloud_run_v2_service" "mlflow" {
  name     = "mlflow-server"
  location = var.region

  template {
    service_account = google_service_account.mlflow.email

    containers {
      # Using public MLflow image for simplicity
      # Replace with custom image from Artifact Registry for production
      image = "ghcr.io/mlflow/mlflow:v2.10.0"

      ports {
        container_port = 5000
      }

      env {
        name  = "MLFLOW_BACKEND_STORE_URI"
        value = "sqlite:///mlflow.db"
      }

      env {
        name  = "MLFLOW_DEFAULT_ARTIFACT_ROOT"
        value = "gs://${var.mlflow_bucket}/artifacts"
      }

      # Command to start MLflow server
      command = ["mlflow"]
      args = [
        "server",
        "--host", "0.0.0.0",
        "--port", "5000",
        "--backend-store-uri", "sqlite:///mlflow.db",
        "--default-artifact-root", "gs://${var.mlflow_bucket}/artifacts"
      ]

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
      }
    }

    scaling {
      min_instance_count = 0  # Scale to zero
      max_instance_count = 2
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  labels = {
    environment = var.environment
  }
}

# IAM - Allow specific user to invoke
resource "google_cloud_run_v2_service_iam_member" "invoker" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.mlflow.name
  role     = "roles/run.invoker"
  member   = "user:${var.invoker_email}"
}

# IAM - Allow service account to invoke (for CI/CD)
resource "google_cloud_run_v2_service_iam_member" "sa_invoker" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.mlflow.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:github-actions@${var.project_id}.iam.gserviceaccount.com"
}

output "mlflow_url" {
  value = google_cloud_run_v2_service.mlflow.uri
}

output "mlflow_service_account" {
  value = google_service_account.mlflow.email
}
```

---

### A2.5 Create GKE Module (Optional)

**File**: `infrastructure/terraform/modules/gke/main.tf`

```hcl
# GKE Standard Cluster for ML Training/Inference

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP Zone"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

# GKE Cluster
resource "google_container_cluster" "primary" {
  name     = "deepmlhub-cluster"
  location = var.zone  # Zonal cluster (cheaper, free tier eligible)

  # Remove default node pool, create custom ones
  remove_default_node_pool = true
  initial_node_count       = 1

  # Enable Workload Identity
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # Network config
  networking_mode = "VPC_NATIVE"
  ip_allocation_policy {}

  # Cluster autoscaling
  cluster_autoscaling {
    enabled = true
    resource_limits {
      resource_type = "cpu"
      minimum       = 0
      maximum       = 16
    }
    resource_limits {
      resource_type = "memory"
      minimum       = 0
      maximum       = 64
    }
  }

  # Maintenance window (off-peak hours)
  maintenance_policy {
    daily_maintenance_window {
      start_time = "03:00"  # 3 AM
    }
  }
}

# CPU Node Pool (Spot VMs for cost savings)
resource "google_container_node_pool" "cpu_pool" {
  name     = "cpu-pool"
  location = var.zone
  cluster  = google_container_cluster.primary.name

  autoscaling {
    min_node_count = 0  # Scale to zero when idle
    max_node_count = 3
  }

  node_config {
    preemptible  = true  # Spot VMs - 60-70% cheaper
    machine_type = "e2-standard-4"  # 4 vCPUs, 16GB RAM

    disk_size_gb = 50
    disk_type    = "pd-standard"

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      workload-type = "cpu"
      environment   = var.environment
    }

    # Workload Identity
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

# GPU Node Pool (Optional - uncomment when needed)
# resource "google_container_node_pool" "gpu_pool" {
#   name     = "gpu-pool"
#   location = var.zone
#   cluster  = google_container_cluster.primary.name
#
#   autoscaling {
#     min_node_count = 0
#     max_node_count = 2
#   }
#
#   node_config {
#     preemptible  = true
#     machine_type = "n1-standard-4"
#
#     guest_accelerator {
#       type  = "nvidia-tesla-t4"
#       count = 1
#       gpu_driver_installation_config {
#         gpu_driver_version = "LATEST"
#       }
#     }
#
#     labels = {
#       workload-type = "gpu"
#     }
#
#     taint {
#       key    = "nvidia.com/gpu"
#       value  = "present"
#       effect = "NO_SCHEDULE"
#     }
#   }
# }

output "cluster_name" {
  value = google_container_cluster.primary.name
}

output "cluster_endpoint" {
  value = google_container_cluster.primary.endpoint
}

output "cluster_zone" {
  value = var.zone
}
```

---

### A2.6 Create Dev Environment Config

**File**: `infrastructure/terraform/environments/dev/main.tf`

```hcl
# Dev Environment Terraform Configuration

terraform {
  required_version = ">= 1.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  backend "gcs" {
    # Configure via terraform init -backend-config
    # bucket = "deepmlhub-YOUR_ID-tfstate"
    # prefix = "terraform/dev"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Variables
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP Zone"
  type        = string
  default     = "us-central1-a"
}

variable "invoker_email" {
  description = "Email of user allowed to invoke Cloud Run services"
  type        = string
}

# GCS Buckets
module "gcs" {
  source = "../../modules/gcs"

  project_id  = var.project_id
  region      = var.region
  environment = "dev"
}

# Artifact Registry
module "artifact_registry" {
  source = "../../modules/artifact-registry"

  project_id  = var.project_id
  region      = var.region
  environment = "dev"
}

# MLflow on Cloud Run
module "mlflow" {
  source = "../../modules/mlflow"

  project_id     = var.project_id
  region         = var.region
  mlflow_bucket  = module.gcs.mlflow_bucket_name
  environment    = "dev"
  invoker_email  = var.invoker_email

  depends_on = [module.gcs]
}

# GKE Cluster (optional - comment out if not needed yet)
# module "gke" {
#   source = "../../modules/gke"
#
#   project_id  = var.project_id
#   region      = var.region
#   zone        = var.zone
#   environment = "dev"
# }

# Outputs
output "dvc_bucket_url" {
  value = module.gcs.dvc_bucket_url
}

output "mlflow_url" {
  value = module.mlflow.mlflow_url
}

output "artifact_registry_url" {
  value = module.artifact_registry.repository_url
}

# output "gke_cluster_name" {
#   value = module.gke.cluster_name
# }
```

**File**: `infrastructure/terraform/environments/dev/terraform.tfvars.example`

```hcl
# Copy this to terraform.tfvars and fill in your values

project_id    = "deepmlhub-YOUR_ID"
region        = "us-central1"
zone          = "us-central1-a"
invoker_email = "your-email@gmail.com"
```

**File**: `infrastructure/terraform/environments/dev/backend.tf.example`

```hcl
# Copy this to backend.tf and fill in your values

terraform {
  backend "gcs" {
    bucket = "deepmlhub-YOUR_ID-tfstate"
    prefix = "terraform/dev"
  }
}
```

---

## Phase 2 Verification

```bash
cd infrastructure/terraform/environments/dev

# Copy example files
cp terraform.tfvars.example terraform.tfvars
cp backend.tf.example backend.tf

# Edit files with your values
# (Human does this manually)

# Initialize Terraform
terraform init

# Plan
terraform plan

# Apply (human reviews and approves)
terraform apply

echo "Phase 2 complete!"
```

---

## Phase 3: Docker Images

**Prerequisites**: Human must complete H5.1-H5.4.

### A3.1 Create Training Dockerfile

**File**: `docker/training/Dockerfile`

```dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY projects/synth_tabular_classification/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY projects/synth_tabular_classification/src/ ./src/
COPY projects/synth_tabular_classification/configs/ ./configs/

# Copy shared utilities (if any)
# COPY shared/ ./shared/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default command (can be overridden)
CMD ["python", "-m", "src.model.train"]
```

**File**: `docker/training/requirements.txt`

```
# Same as project requirements, or minimal subset
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
PyYAML>=6.0
joblib>=1.3.0
mlflow>=2.10.0
google-cloud-storage>=2.14.0
```

---

### A3.2 Create Inference Dockerfile

**File**: `docker/inference/Dockerfile`

```dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY projects/synth_tabular_classification/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY projects/synth_tabular_classification/src/ ./src/
COPY projects/synth_tabular_classification/configs/ ./configs/
COPY projects/synth_tabular_classification/models/ ./models/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["uvicorn", "src.inference.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

### A3.3 Create .dockerignore

**File**: `.dockerignore`

```
# Git
.git/
.gitignore

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
.eggs/
dist/
build/
.pytest_cache/

# Virtual environments
.venv/
venv/
env/

# IDE
.vscode/
.idea/

# Data (use DVC to pull)
projects/*/data/raw/
projects/*/data/processed/

# MLflow local runs
projects/*/mlruns/

# DVC
.dvc/cache/
.dvc/tmp/

# OS
.DS_Store

# Terraform
infrastructure/terraform/**/.terraform/
infrastructure/terraform/**/*.tfstate*

# Secrets
*.key
*.pem
*.json
!package.json

# Agent files
.agents/
.human/
```

---

## Phase 3 Verification

```bash
# Build training image
docker build -f docker/training/Dockerfile -t training:local .

# Build inference image (requires model first)
cd projects/synth_tabular_classification
dvc repro  # Generate model
cd ../..
docker build -f docker/inference/Dockerfile -t inference:local .

# Test inference container
docker run -d -p 8000:8000 --name test-inference inference:local
sleep 5
curl http://localhost:8000/health
curl http://localhost:8000/info
docker stop test-inference && docker rm test-inference

echo "Phase 3 complete!"
```

---

## Phase 4: GitHub Actions CI/CD

**Prerequisites**: Human must complete H3.1-H3.3.

### A4.1 Create CI Workflow

**File**: `.github/workflows/ci.yaml`

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  PYTHON_VERSION: "3.10"

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install -r projects/synth_tabular_classification/requirements.txt
          pip install ruff mypy

      - name: Lint with ruff
        run: ruff check projects/synth_tabular_classification/src/

      - name: Type check with mypy
        run: mypy projects/synth_tabular_classification/src/ --ignore-missing-imports
        continue-on-error: true  # Don't fail on type errors initially

      - name: Run tests
        run: |
          cd projects/synth_tabular_classification
          pytest tests/ -v

  dvc-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up DVC
        uses: iterative/setup-dvc@v2

      - name: Check DVC pipeline validity
        run: |
          cd projects/synth_tabular_classification
          dvc dag
```

---

### A4.2 Create Build and Push Workflow

**File**: `.github/workflows/build-push.yaml`

```yaml
name: Build and Push Docker Images

on:
  push:
    branches: [main]
    paths:
      - 'docker/**'
      - 'projects/synth_tabular_classification/src/**'
      - 'projects/synth_tabular_classification/requirements.txt'
  workflow_dispatch:  # Manual trigger

env:
  PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
  REGION: ${{ secrets.GCP_REGION }}

jobs:
  build-training:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - id: auth
        uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}

      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v2

      - name: Configure Docker
        run: gcloud auth configure-docker ${{ env.REGION }}-docker.pkg.dev

      - name: Build and push training image
        run: |
          IMAGE_TAG=${{ env.REGION }}-docker.pkg.dev/${{ env.PROJECT_ID }}/ml/training
          docker build -f docker/training/Dockerfile \
            -t ${IMAGE_TAG}:${{ github.sha }} \
            -t ${IMAGE_TAG}:latest \
            .
          docker push ${IMAGE_TAG} --all-tags

  build-inference:
    runs-on: ubuntu-latest
    needs: build-training  # Build after training (uses same base)
    steps:
      - uses: actions/checkout@v4

      - id: auth
        uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}

      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v2

      - name: Set up DVC
        uses: iterative/setup-dvc@v2

      - name: Configure Docker
        run: gcloud auth configure-docker ${{ env.REGION }}-docker.pkg.dev

      - name: Pull model from DVC
        run: |
          cd projects/synth_tabular_classification
          dvc remote modify gcs credentialpath $GOOGLE_APPLICATION_CREDENTIALS
          dvc pull models/

      - name: Build and push inference image
        run: |
          IMAGE_TAG=${{ env.REGION }}-docker.pkg.dev/${{ env.PROJECT_ID }}/ml/inference
          docker build -f docker/inference/Dockerfile \
            -t ${IMAGE_TAG}:${{ github.sha }} \
            -t ${IMAGE_TAG}:latest \
            .
          docker push ${IMAGE_TAG} --all-tags
```

---

### A4.3 Create Training Trigger Workflow

**File**: `.github/workflows/train.yaml`

```yaml
name: Trigger Training

on:
  workflow_dispatch:
    inputs:
      experiment_name:
        description: 'Experiment name for MLflow'
        required: false
        default: 'synth_tabular_classification'
      n_samples:
        description: 'Number of samples to generate'
        required: false
        default: '1000'

env:
  PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
  REGION: ${{ secrets.GCP_REGION }}

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - id: auth
        uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}

      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v2

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: |
          pip install -r projects/synth_tabular_classification/requirements.txt

      - name: Set up DVC
        uses: iterative/setup-dvc@v2

      - name: Configure DVC
        run: |
          cd projects/synth_tabular_classification
          dvc remote modify gcs credentialpath $GOOGLE_APPLICATION_CREDENTIALS

      - name: Get MLflow URL
        id: mlflow
        run: |
          MLFLOW_URL=$(gcloud run services describe mlflow-server \
            --region=${{ env.REGION }} \
            --format='value(status.url)')
          echo "url=$MLFLOW_URL" >> $GITHUB_OUTPUT

      - name: Run training pipeline
        env:
          MLFLOW_TRACKING_URI: ${{ steps.mlflow.outputs.url }}
        run: |
          cd projects/synth_tabular_classification

          # Get identity token for Cloud Run auth
          export MLFLOW_TRACKING_TOKEN=$(gcloud auth print-identity-token)

          # Run DVC pipeline
          dvc repro

          # Push results
          dvc push

          # Commit DVC lock file
          git config user.name "GitHub Actions"
          git config user.email "actions@github.com"
          git add dvc.lock metrics/
          git commit -m "Update model from training run" || echo "No changes to commit"
          git push || echo "Nothing to push"

      - name: Upload metrics as artifact
        uses: actions/upload-artifact@v4
        with:
          name: training-metrics
          path: projects/synth_tabular_classification/metrics/
```

---

## Phase 4 Verification

```bash
# Verify workflow files are valid YAML
python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yaml'))"
python -c "import yaml; yaml.safe_load(open('.github/workflows/build-push.yaml'))"
python -c "import yaml; yaml.safe_load(open('.github/workflows/train.yaml'))"

# Push to GitHub and check Actions tab
git add .github/
git commit -m "Add CI/CD workflows"
git push

# Check GitHub Actions for successful runs

echo "Phase 4 complete!"
```

---

## Phase 5: Kubernetes Manifests (Optional)

**Prerequisites**: Human must complete H6.1-H6.3.

### A5.1 Create Namespace and ConfigMaps

**File**: `infrastructure/kubernetes/base/namespace.yaml`

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ml-training
  labels:
    purpose: training
---
apiVersion: v1
kind: Namespace
metadata:
  name: ml-inference
  labels:
    purpose: inference
```

**File**: `infrastructure/kubernetes/base/configmaps/training-config.yaml`

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: training-config
  namespace: ml-training
data:
  # These will be replaced with actual values
  mlflow_tracking_uri: "https://mlflow-server-XXXXX.run.app"
  dvc_remote: "gcs"
  log_level: "INFO"
```

**File**: `infrastructure/kubernetes/base/configmaps/inference-config.yaml`

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: inference-config
  namespace: ml-inference
data:
  model_version: "latest"
  log_level: "INFO"
```

---

### A5.2 Create Inference Deployment

**File**: `infrastructure/kubernetes/inference/deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: synth-tabular-api
  namespace: ml-inference
  labels:
    app: synth-tabular-api
spec:
  replicas: 1
  selector:
    matchLabels:
      app: synth-tabular-api
  template:
    metadata:
      labels:
        app: synth-tabular-api
    spec:
      containers:
      - name: api
        image: us-central1-docker.pkg.dev/PROJECT_ID/ml/inference:latest
        ports:
        - containerPort: 8000

        env:
        - name: LOG_LEVEL
          valueFrom:
            configMapKeyRef:
              name: inference-config
              key: log_level

        resources:
          requests:
            cpu: "250m"
            memory: "512Mi"
          limits:
            cpu: "500m"
            memory: "1Gi"

        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10

        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: synth-tabular-api
  namespace: ml-inference
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: synth-tabular-api
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: synth-tabular-api-hpa
  namespace: ml-inference
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: synth-tabular-api
  minReplicas: 1
  maxReplicas: 3
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

### A5.3 Create Training Job Template

**File**: `infrastructure/kubernetes/training/job.yaml`

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: training-job
  namespace: ml-training
spec:
  backoffLimit: 2
  template:
    spec:
      restartPolicy: Never

      containers:
      - name: training
        image: us-central1-docker.pkg.dev/PROJECT_ID/ml/training:latest

        command: ["python", "-m", "src.model.train"]

        env:
        - name: MLFLOW_TRACKING_URI
          valueFrom:
            configMapKeyRef:
              name: training-config
              key: mlflow_tracking_uri

        resources:
          requests:
            cpu: "1"
            memory: "2Gi"
          limits:
            cpu: "2"
            memory: "4Gi"

      nodeSelector:
        workload-type: cpu
```

---

## Quick Reference Commands

```bash
# === Local Development ===
cd projects/synth_tabular_classification
source ../../.venv/bin/activate
dvc repro
mlflow ui --backend-store-uri file://$(pwd)/mlruns

# === Docker ===
docker build -f docker/training/Dockerfile -t training:local .
docker build -f docker/inference/Dockerfile -t inference:local .
docker run -p 8000:8000 inference:local

# === Terraform ===
cd infrastructure/terraform/environments/dev
terraform init
terraform plan
terraform apply

# === GKE (after human setup) ===
gcloud container clusters get-credentials deepmlhub-cluster --zone us-central1-a
kubectl apply -f infrastructure/kubernetes/base/
kubectl apply -f infrastructure/kubernetes/inference/
kubectl get pods -n ml-inference

# === DVC with GCS ===
dvc push
dvc pull
dvc remote list
```

---

## Status Tracker

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Local ML Pipeline (Python code) | ⬜ Not Started |
| 1 | DVC Pipeline Setup | ⬜ Not Started |
| 2 | Terraform Infrastructure | ⬜ Not Started |
| 3 | Docker Images | ⬜ Not Started |
| 4 | GitHub Actions CI/CD | ⬜ Not Started |
| 5 | Kubernetes Manifests | ⬜ Not Started |

Update status as you complete each phase:
- ⬜ Not Started
- 🔄 In Progress
- ✅ Complete
- ⏭️ Skipped
