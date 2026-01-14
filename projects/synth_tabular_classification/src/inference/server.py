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
        example=[[0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1.0]],
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
