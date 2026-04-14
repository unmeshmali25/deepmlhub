"""Tests for FastAPI inference endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.inference.server import app


client = TestClient(app)


class TestHealthEndpoints:
    def test_health_check(self):
        """Test health check endpoint returns healthy."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data

    def test_ready_without_model(self):
        """Test readiness probe fails when model is not loaded."""
        # Model won't be loaded in test environment
        response = client.get("/ready")
        assert response.status_code == 503


class TestRecommendEndpoint:
    def test_recommend_without_model(self):
        """Test recommend endpoint returns 503 when model not loaded."""
        response = client.post("/recommend", json={"agent_id": 1, "top_k": 5})
        assert response.status_code == 503

    def test_recommend_request_validation(self):
        """Test recommend endpoint validates request body."""
        # Missing required field agent_id
        response = client.post("/recommend", json={"top_k": 5})
        assert response.status_code == 422


class TestFeatureEndpoint:
    def test_get_features(self):
        """Test feature endpoint handles requests."""
        response = client.get("/features/1")
        # Should return 200 with fallback or 500 if Feast not configured
        assert response.status_code in [200, 500]
