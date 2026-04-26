"""Fetch training features using Feast's offline store.

Replaces direct SQL queries with Feast's get_historical_features() API,
providing point-in-time correct feature joins and train/serve consistency.

Flow:
1. Build entity DataFrame from interaction table (agent_id, product_id, timestamp)
2. Call Feast get_historical_features() for all 3 FeatureViews
3. Rename columns to strip Feast prefixes, handling cross-view conflicts
4. Create purchased label, cast object types, save to parquet
"""

import os
from pathlib import Path

import pandas as pd
import yaml
from feast import FeatureStore
from sqlalchemy import create_engine, text


def load_config() -> dict:
    config_path = Path(__file__).parents[2] / "configs" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_feast_store(config: dict) -> FeatureStore:
    """Initialize Feast FeatureStore from repo path in config."""
    feast_config = config.get("feast", {})
    repo_path = feast_config.get("repo_path", "./feast")
    abs_repo_path = Path(__file__).parents[2] / repo_path
    return FeatureStore(repo_path=str(abs_repo_path))


def get_entity_dataframe(config: dict) -> pd.DataFrame:
    """Build entity DataFrame from the interaction feature table.

    Queries fct_agent_product_interactions for (agent_id, product_id,
    event_timestamp) tuples. Feast uses this to perform point-in-time
    correct joins against all FeatureViews.
    """
    dbt_config = config.get("dbt", {})
    schema = dbt_config.get("schema", "dbt_vor")

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        host = os.getenv("SUPABASE_DB_HOST")
        user = os.getenv("SUPABASE_DB_USER")
        password = os.getenv("SUPABASE_DB_PASSWORD")
        if not host or not user or not password:
            raise RuntimeError(
                "Database connection not configured. Set DATABASE_URL or "
                "SUPABASE_DB_HOST, SUPABASE_DB_USER, SUPABASE_DB_PASSWORD."
            )
        db_url = f"postgresql://{user}:{password}@{host}:5432/postgres?sslmode=require"

    engine = create_engine(db_url)

    with engine.connect() as conn:
        df = pd.read_sql(
            text(
                f"""
                SELECT
                    agent_id,
                    product_id,
                    CURRENT_TIMESTAMP AS event_timestamp
                FROM {schema}.fct_agent_product_interactions
            """
            ),
            conn,
        )

    df["agent_id"] = df["agent_id"].astype(str)
    df["product_id"] = df["product_id"].astype(str)

    return df


def _all_feature_refs() -> list:
    """Build feature reference list from the 3 FeatureViews in priority order.

    Priority: interaction features first (they form the training rows),
    then agent features, then product features. This ordering controls
    which view "owns" a column when the same name appears in multiple views.
    """
    _AGENT_FEATURES = [
        "snapshot_date",
        "age",
        "age_group",
        "gender",
        "income_bracket",
        "household_size",
        "has_children",
        "location_region",
        "price_sensitivity",
        "brand_loyalty",
        "impulsivity",
        "tech_savviness",
        "preferred_categories",
        "declared_weekly_budget",
        "shopping_frequency",
        "declared_avg_cart_value",
        "pref_day_weekday",
        "pref_day_saturday",
        "pref_day_sunday",
        "pref_time_morning",
        "pref_time_afternoon",
        "pref_time_evening",
        "coupon_affinity",
        "deal_seeking_behavior",
        "remaining_budget",
        "spend_this_week",
        "spend_this_month",
        "days_since_last_purchase",
        "total_orders_lifetime",
        "total_orders_this_week",
        "cart_value",
        "cart_item_count",
        "has_active_cart",
        "products_viewed_count",
        "sessions_count_today",
        "sessions_count_this_week",
        "categories_purchased_count",
        "categories_purchased_this_week",
        "diversity_ratio",
        "pref_day_match",
        "pref_time_match",
        "active_coupons_count",
        "coupons_redeemed_this_week",
        "remaining_budget_pct",
        "spend_this_week_pct",
        "coupon_views",
        "coupon_clicks",
        "coupon_redeems",
        "total_coupons_assigned",
        "total_coupons_redeemed",
        "coupon_redemption_rate",
    ]

    _PRODUCT_FEATURES = [
        "product_category",
        "product_brand",
        "price",
        "margin_percent",
        "rating",
        "review_count",
        "in_stock",
        "order_count",
        "total_units_sold",
        "unique_buyers_count",
        "avg_selling_price",
        "avg_discount_given",
        "total_revenue",
        "available_quantity",
        "stores_available",
        "revenue_per_unit",
        "avg_quantity_per_buyer",
    ]

    _INTERACTION_FEATURES = [
        "last_purchase_date",
        "purchase_count",
        "total_spent",
        "avg_discount_received",
        "total_units_bought",
        "product_category",
        "category_purchase_count",
        "category_total_spent",
        "total_coupons_assigned",
        "total_coupons_redeemed",
        "coupon_redemption_rate",
    ]

    refs = []
    for f in _INTERACTION_FEATURES:
        refs.append(f"agent_product_interaction:{f}")
    for f in _AGENT_FEATURES:
        refs.append(f"agent_features:{f}")
    for f in _PRODUCT_FEATURES:
        refs.append(f"product_features:{f}")

    return refs


def _build_rename_map(df: pd.DataFrame) -> dict:
    """Build a column rename map handling Feast prefixes and conflicts.

    Feast returns columns like "agent_features__age". This strips the
    prefix. For names that appear in multiple FeatureViews (e.g.,
    coupon_redemption_rate in both agent and interaction), the FIRST
    occurrence keeps the base name and subsequent ones get suffixes.

    Suffix rules (matching historical merge order):
    - interaction view  -> base name (first priority, no suffix)
    - agent view        -> "_agent" suffix when conflicting
    - product view      -> "_product" suffix when conflicting
    """
    PREFIX_LENGTHS = {
        "agent_features__": 16,
        "agent_product_interaction__": 27,
        "product_features__": 18,
    }

    rename_map = {}
    seen: dict = {}

    for col in df.columns:
        # Strip known Feast prefix to get base column name
        base = col
        for prefix, length in PREFIX_LENGTHS.items():
            if col.startswith(prefix):
                base = col[length:]
                break

        # Drop all event_timestamp columns (not part of training data)
        if base == "event_timestamp":
            continue

        if base in seen:
            seen[base] += 1
            if col.startswith("agent_features__"):
                rename_map[col] = f"{base}_agent"
            elif col.startswith("product_features__"):
                rename_map[col] = f"{base}_product"
            else:
                rename_map[col] = f"{base}_dup"
        else:
            seen[base] = 1
            rename_map[col] = base

    return rename_map


def _rename_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply rename map and drop unneeded columns."""
    rename_map = _build_rename_map(df)
    df = df.rename(columns=rename_map)
    df = df[list(rename_map.values())]
    return df


def fetch_features_from_feast(config: dict) -> pd.DataFrame:
    """Fetch training features using Feast point-in-time joins."""
    store = get_feast_store(config)
    entity_df = get_entity_dataframe(config)
    print(f"Entity DataFrame: {len(entity_df)} rows")

    feature_refs = _all_feature_refs()
    print(f"Fetching {len(feature_refs)} features via Feast...")

    job = store.get_historical_features(
        entity_df=entity_df,
        features=feature_refs,
        full_feature_names=True,
    )
    df = job.to_df()

    df = _rename_and_clean(df)

    print(f"Fetched {df.shape[0]} rows x {df.shape[1]} columns")

    return df


def fetch_training_features() -> pd.DataFrame:
    """Main entry point: fetch features via Feast and save to parquet."""
    config = load_config()
    paths = config["paths"]
    base_path = Path(__file__).parents[2]

    df = fetch_features_from_feast(config)

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str)

    if "purchase_count" in df.columns:
        df["purchased"] = (df["purchase_count"] > 0).astype(int)

    output_path = base_path / paths["training_features"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Training features saved to {output_path}")
    print(f"Final shape: {df.shape}")

    return df


if __name__ == "__main__":
    fetch_training_features()
