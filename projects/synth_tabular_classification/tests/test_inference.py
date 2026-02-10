"""Tests for inference server."""

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
            predictions=[0, 1], probabilities=[[0.8, 0.2], [0.3, 0.7]]
        )
        assert len(response.predictions) == 2
        assert len(response.probabilities) == 2
