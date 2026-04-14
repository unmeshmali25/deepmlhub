"""Recommendation algorithm placeholder with extensible interface.

This module defines the base recommender interface and a simple rule-based
implementation for testing the MLOps pipeline. It is designed to be easily
swapped for collaborative filtering or neural collaborative filtering later.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import pandas as pd


class BaseRecommender(ABC):
    """Abstract base class for all recommendation algorithms.

    # Sensor: Accepts feature vectors and agent/product IDs
    # Decision Maker: Outputs ranked product recommendations
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseRecommender":
        """Train the recommender on historical interaction data.

        Args:
            df: DataFrame with agent_id, product_id, features, and target.

        Returns:
            self for method chaining.
        """
        pass

    @abstractmethod
    def predict(
        self, agent_id: int, product_ids: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Predict scores for a single agent and a list of products.

        Args:
            agent_id: The agent to generate predictions for.
            product_ids: List of candidate products. If None, use all known products.

        Returns:
            DataFrame with columns [product_id, score] sorted by score descending.
        """
        pass

    def recommend(self, agent_id: int, top_k: int = 10) -> List[int]:
        """Return top-k product IDs for the given agent.

        Args:
            agent_id: The agent to recommend products to.
            top_k: Number of recommendations to return.

        Returns:
            List of top-k product IDs.
        """
        predictions = self.predict(agent_id)
        return predictions["product_id"].head(top_k).tolist()


class SimpleRuleRecommender(BaseRecommender):
    """Basic rule-based recommender for pipeline testing.

    Scores products using a weighted combination of:
    - Discount level (higher discount = higher score)
    - Product popularity (more orders = higher score)
    - Interaction recency (recent views/carts = higher score)

    # Decision Maker: Simple weighted heuristic
    # TODO: Replace with collaborative filtering or neural CF in Sprint 10
    """

    def __init__(
        self,
        top_k: int = 10,
        discount_weight: float = 0.3,
        popularity_weight: float = 0.2,
        recency_weight: float = 0.5,
    ):
        self.top_k = top_k
        self.discount_weight = discount_weight
        self.popularity_weight = popularity_weight
        self.recency_weight = recency_weight
        self.product_catalog: Optional[pd.DataFrame] = None
        self.agent_interactions: Optional[pd.DataFrame] = None

    def fit(self, df: pd.DataFrame) -> "SimpleRuleRecommender":
        """Build product catalog and agent interaction lookup from training data."""
        # Extract unique product features
        product_cols = [
            c for c in df.columns if c.startswith(("order_count", "avg_discount"))
        ]
        if not product_cols:
            product_cols = ["product_id"]  # Fallback

        self.product_catalog = (
            df[["product_id"] + product_cols].drop_duplicates("product_id").copy()
        )

        # Extract agent-product interactions
        interaction_cols = [c for c in df.columns if c.startswith(("times_", "last_"))]
        self.agent_interactions = df[
            ["agent_id", "product_id"] + interaction_cols
        ].copy()

        print(f"SimpleRuleRecommender fitted on {len(df)} rows")
        print(f"Product catalog size: {len(self.product_catalog)}")

        return self

    def predict(
        self, agent_id: int, product_ids: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Score all products for the given agent."""
        if self.product_catalog is None:
            raise RuntimeError("Recommender has not been fitted yet.")

        candidates = self.product_catalog.copy()

        if product_ids is not None:
            candidates = candidates[candidates["product_id"].isin(product_ids)]

        # Merge agent interactions (if any)
        if self.agent_interactions is not None:
            agent_hist = self.agent_interactions[
                self.agent_interactions["agent_id"] == agent_id
            ]
            candidates = candidates.merge(agent_hist, on="product_id", how="left")

        # Fill missing interaction features with 0
        for col in ["times_viewed", "times_added_to_cart", "times_purchased"]:
            if col in candidates.columns:
                candidates[col] = candidates[col].fillna(0)

        # Compute score
        # Discount score
        if "avg_discount" in candidates.columns:
            discount_score = candidates["avg_discount"] / (
                candidates["avg_discount"].max() + 1e-8
            )
        else:
            discount_score = 0.0

        # Popularity score
        if "order_count_7d" in candidates.columns:
            popularity_score = candidates["order_count_7d"] / (
                candidates["order_count_7d"].max() + 1e-8
            )
        else:
            popularity_score = 0.0

        # Recency score (interaction recency proxy)
        recency_score = 0.0
        if "times_viewed" in candidates.columns:
            recency_score += candidates["times_viewed"] / (
                candidates["times_viewed"].max() + 1e-8
            )
        if "times_added_to_cart" in candidates.columns:
            recency_score += candidates["times_added_to_cart"] / (
                candidates["times_added_to_cart"].max() + 1e-8
            )

        candidates["score"] = (
            self.discount_weight * discount_score
            + self.popularity_weight * popularity_score
            + self.recency_weight * recency_score
        )

        # Break ties with random noise for deterministic-ish but varied results
        candidates["score"] += np.random.uniform(0, 0.001, size=len(candidates))

        return candidates.sort_values("score", ascending=False)[
            ["product_id", "score"]
        ].reset_index(drop=True)


class CollaborativeFilteringRecommender(BaseRecommender):
    """Placeholder for matrix factorization / collaborative filtering.

    # TODO: Implement in Sprint 10
    """

    def __init__(self, embedding_dim: int = 64, regularization: float = 0.01):
        self.embedding_dim = embedding_dim
        self.regularization = regularization

    def fit(self, df: pd.DataFrame) -> "CollaborativeFilteringRecommender":
        raise NotImplementedError(
            "CollaborativeFilteringRecommender not implemented yet. "
            "Use SimpleRuleRecommender for pipeline testing."
        )

    def predict(
        self, agent_id: int, product_ids: Optional[List[int]] = None
    ) -> pd.DataFrame:
        raise NotImplementedError("Not implemented yet.")


class NeuralCollaborativeFilteringRecommender(BaseRecommender):
    """Placeholder for neural collaborative filtering.

    # TODO: Implement in Sprint 11
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_layers: Optional[List[int]] = None,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 10,
        batch_size: int = 256,
    ):
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers or [256, 128, 64]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, df: pd.DataFrame) -> "NeuralCollaborativeFilteringRecommender":
        raise NotImplementedError(
            "NeuralCollaborativeFilteringRecommender not implemented yet. "
            "Use SimpleRuleRecommender for pipeline testing."
        )

    def predict(
        self, agent_id: int, product_ids: Optional[List[int]] = None
    ) -> pd.DataFrame:
        raise NotImplementedError("Not implemented yet.")


# Registry for easy model instantiation from config
MODEL_REGISTRY = {
    "simple_rule": SimpleRuleRecommender,
    "collaborative_filtering": CollaborativeFilteringRecommender,
    "neural_cf": NeuralCollaborativeFilteringRecommender,
}


def get_recommender(model_type: str, **kwargs) -> BaseRecommender:
    """Factory function to instantiate a recommender by type."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_type](**kwargs)
