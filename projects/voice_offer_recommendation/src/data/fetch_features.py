"""Fetch training features from Feast offline store or Supabase directly."""

from pathlib import Path

import pandas as pd
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


def fetch_features_from_feast(config: dict) -> pd.DataFrame:
    """Fetch historical features from Feast offline store."""
    if FeatureStore is None:
        raise ImportError(
            "Feast is not installed. Use fetch_features_from_supabase() instead."
        )

    feast_config = config["feast"]
    store = FeatureStore(repo_path=feast_config["repo_path"])

    # Build feature references
    feature_refs = []
    for view_name in feast_config["feature_views"]:
        # Get all features from this view
        feature_service = store.get_feature_view(view_name)
        for feature in feature_service.features:
            feature_refs.append(f"{view_name}:{feature.name}")

    # Create entity dataframe
    # For training, we need all agent-product pairs with timestamps
    # This is a simplified version - in production, you'd query your
    # actual transactions table for entity keys and timestamps
    print("Fetching historical features from Feast...")
    print(f"Feature references: {feature_refs}")

    # TODO: Replace with actual entity dataframe from Supabase
    # For now, return a dummy dataframe to allow pipeline testing
    raise NotImplementedError(
        "fetch_features_from_feast requires an entity dataframe from your source data. "
        "Use fetch_features_from_supabase() as a fallback until Feast offline store is fully configured."
    )


def fetch_features_from_supabase(config: dict) -> pd.DataFrame:
    """Fetch training data directly from Supabase as a fallback."""
    from supabase_client import supabase

    print("Fetching training data directly from Supabase...")

    # Query agent features
    agent_response = supabase.table("agent_stats").select("*").execute()
    agent_df = pd.DataFrame(agent_response.data)

    # Query product features
    product_response = supabase.table("product_stats").select("*").execute()
    product_df = pd.DataFrame(product_response.data)

    # Query interactions (our target data)
    interaction_response = (
        supabase.table("agent_product_interactions").select("*").execute()
    )
    interaction_df = pd.DataFrame(interaction_response.data)

    # Merge into training dataset
    training_df = interaction_df.merge(
        agent_df, on="agent_id", how="left", suffixes=("", "_agent")
    )
    training_df = training_df.merge(
        product_df, on="product_id", how="left", suffixes=("", "_product")
    )

    # Create binary target: purchased or not
    training_df["purchased"] = (training_df["times_purchased"] > 0).astype(int)

    return training_df


def fetch_training_features() -> pd.DataFrame:
    """Fetch training features and save to parquet."""
    config = load_config()
    paths = config["paths"]
    base_path = Path(__file__).parents[2]

    # Try Feast first, fall back to direct Supabase query
    try:
        df = fetch_features_from_feast(config)
    except Exception as e:
        print(f"Feast fetch failed: {e}")
        print("Falling back to direct Supabase query...")
        df = fetch_features_from_supabase(config)

    # Save to parquet
    output_path = base_path / paths["training_features"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Training features saved to {output_path}")
    print(f"Shape: {df.shape}")

    return df


if __name__ == "__main__":
    fetch_training_features()
