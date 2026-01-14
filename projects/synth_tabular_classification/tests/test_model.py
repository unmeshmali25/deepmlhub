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
