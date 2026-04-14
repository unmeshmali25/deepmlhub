"""FastAPI inference server for voice offer recommendations."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import joblib
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.inference.feast_client import get_online_features
from src.inference.predict import RecommendationPredictor

# Global state
model = None
predictor = None


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parents[2] / "configs" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and predictor on startup."""
    global model, predictor
    config = load_config()
    paths = config["paths"]
    base_path = Path(__file__).parents[2]
    model_path = base_path / paths["model"]

    if not model_path.exists():
        print(f"Warning: Model not found at {model_path}")
        model = None
        predictor = None
    else:
        model = joblib.load(model_path)
        predictor = RecommendationPredictor(model)
        print(f"Loaded model from {model_path}")

    yield

    # Cleanup on shutdown
    model = None
    predictor = None


app = FastAPI(
    title="Voice Offer Recommendation API",
    version="0.1.0",
    lifespan=lifespan,
)


class RecommendRequest(BaseModel):
    agent_id: int
    top_k: Optional[int] = 10
    product_ids: Optional[List[int]] = None


class RecommendResponse(BaseModel):
    agent_id: int
    recommendations: List[dict]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
    )


@app.get("/ready")
def readiness_check():
    """Readiness probe for Kubernetes."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    """Get product recommendations for an agent."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        recommendations = predictor.predict(
            agent_id=request.agent_id,
            top_k=request.top_k,
            product_ids=request.product_ids,
        )
        return RecommendResponse(
            agent_id=request.agent_id,
            recommendations=recommendations,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/features/{agent_id}")
def get_features(agent_id: int):
    """Debug endpoint to view online features for an agent."""
    try:
        features = get_online_features(agent_id)
        return {"agent_id": agent_id, "features": features}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    config = load_config()
    inference_config = config["inference"]
    uvicorn.run(
        app,
        host=inference_config["host"],
        port=inference_config["port"],
    )
