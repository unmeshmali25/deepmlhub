"""Tests for recommendation algorithms."""

import numpy as np
import pandas as pd
import pytest

from src.model.recommendation_algorithm import (
    CollaborativeFilteringRecommender,
    NeuralCollaborativeFilteringRecommender,
    SimpleRuleRecommender,
    get_recommender,
)


class TestSimpleRuleRecommender:
    def test_fit_and_predict(self):
        """Test that SimpleRuleRecommender can fit and predict."""
        df = pd.DataFrame(
            {
                "agent_id": [1, 1, 1, 2, 2],
                "product_id": [101, 102, 103, 101, 104],
                "order_count_7d": [10, 5, 20, 10, 1],
                "avg_discount": [0.1, 0.2, 0.05, 0.1, 0.0],
                "times_viewed": [2, 0, 1, 5, 0],
                "times_added_to_cart": [1, 0, 0, 2, 0],
                "times_purchased": [1, 0, 0, 1, 0],
            }
        )

        model = SimpleRuleRecommender()
        model.fit(df)

        predictions = model.predict(agent_id=1)
        assert len(predictions) > 0
        assert "product_id" in predictions.columns
        assert "score" in predictions.columns
        assert predictions["score"].is_monotonic_decreasing

    def test_recommend_returns_top_k(self):
        """Test that recommend() returns exactly top_k items."""
        df = pd.DataFrame(
            {
                "agent_id": [1, 1, 1, 2, 2],
                "product_id": [101, 102, 103, 101, 104],
                "order_count_7d": [10, 5, 20, 10, 1],
                "avg_discount": [0.1, 0.2, 0.05, 0.1, 0.0],
                "times_viewed": [2, 0, 1, 5, 0],
                "times_added_to_cart": [1, 0, 0, 2, 0],
                "times_purchased": [1, 0, 0, 1, 0],
            }
        )

        model = SimpleRuleRecommender()
        model.fit(df)

        recs = model.recommend(agent_id=1, top_k=2)
        assert len(recs) == 2
        assert all(isinstance(pid, (int, np.integer)) for pid in recs)

    def test_predict_with_product_filter(self):
        """Test that predict respects product_ids filter."""
        df = pd.DataFrame(
            {
                "agent_id": [1, 1, 1, 2, 2],
                "product_id": [101, 102, 103, 101, 104],
                "order_count_7d": [10, 5, 20, 10, 1],
                "avg_discount": [0.1, 0.2, 0.05, 0.1, 0.0],
                "times_viewed": [2, 0, 1, 5, 0],
                "times_added_to_cart": [1, 0, 0, 2, 0],
                "times_purchased": [1, 0, 0, 1, 0],
            }
        )

        model = SimpleRuleRecommender()
        model.fit(df)

        predictions = model.predict(agent_id=1, product_ids=[101, 102])
        assert len(predictions) == 2
        assert set(predictions["product_id"]) == {101, 102}

    def test_unfitted_raises(self):
        """Test that predict raises if model is not fitted."""
        model = SimpleRuleRecommender()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(agent_id=1)


class TestModelRegistry:
    def test_get_recommender_simple_rule(self):
        """Test factory function for simple rule recommender."""
        model = get_recommender("simple_rule", top_k=5)
        assert isinstance(model, SimpleRuleRecommender)
        assert model.top_k == 5

    def test_get_recommender_unknown_type(self):
        """Test factory function raises on unknown type."""
        with pytest.raises(ValueError, match="Unknown model type"):
            get_recommender("unknown_algorithm")


class TestPlaceholderModels:
    def test_collaborative_filtering_not_implemented(self):
        """Test that CF placeholder raises NotImplementedError."""
        model = CollaborativeFilteringRecommender()
        df = pd.DataFrame({"agent_id": [1], "product_id": [101]})
        with pytest.raises(NotImplementedError):
            model.fit(df)

    def test_neural_cf_not_implemented(self):
        """Test that neural CF placeholder raises NotImplementedError."""
        model = NeuralCollaborativeFilteringRecommender()
        df = pd.DataFrame({"agent_id": [1], "product_id": [101]})
        with pytest.raises(NotImplementedError):
            model.fit(df)
