"""Prediction logic for the recommendation inference server."""

from typing import List, Optional


from src.inference.feast_client import get_online_features


class RecommendationPredictor:
    """Wrapper around the recommendation model for inference.

    # Sensor: Fetches online features from Feast
    # Decision Maker: Calls model.predict() and formats results
    """

    def __init__(self, model):
        self.model = model

    def predict(
        self, agent_id: int, top_k: int = 10, product_ids: Optional[List[int]] = None
    ) -> List[dict]:
        """Generate recommendations for an agent.

        Args:
            agent_id: The agent to recommend products to.
            top_k: Number of recommendations to return.
            product_ids: Optional list of candidate products to score.

        Returns:
            List of recommendation dicts with product_id and score.
        """
        # Fetch online features (for logging/debugging, model may not need them)
        _ = get_online_features(agent_id)

        # Get predictions from model
        predictions = self.model.predict(agent_id, product_ids=product_ids)
        top_predictions = predictions.head(top_k)

        return [
            {"product_id": int(row["product_id"]), "score": float(row["score"])}
            for _, row in top_predictions.iterrows()
        ]
