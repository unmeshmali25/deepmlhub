"""Feast online feature client for inference."""

from pathlib import Path

import yaml

try:
    from feast import FeatureStore
except ImportError:
    FeatureStore = None


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parents[2] / "configs" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_online_features(agent_id: int) -> dict:
    """Fetch online features for a single agent from Feast.

    Uses Supabase PostgreSQL as the online store (zero additional cost).
    """
    config = load_config()
    feast_config = config["feast"]

    if FeatureStore is None:
        print("Feast is not installed. Returning fallback features.")
        return {"agent_id": agent_id, "status": "fallback_no_feast"}

    try:
        store = FeatureStore(repo_path=feast_config["repo_path"])

        feature_refs = []
        for view_name in feast_config["feature_views"]:
            fv = store.get_feature_view(view_name)
            for feature in fv.features:
                feature_refs.append(f"{view_name}:{feature.name}")

        features = store.get_online_features(
            features=feature_refs,
            entity_rows=[{"agent_id": agent_id}],
        ).to_dict()

        return features
    except Exception as e:
        # Fallback: return empty features if Feast is not fully configured
        print(f"Feast online feature fetch failed: {e}")
        return {"agent_id": agent_id, "status": "fallback_empty"}
