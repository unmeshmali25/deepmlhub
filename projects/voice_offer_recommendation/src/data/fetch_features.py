"""Fetch training features from dbt feature tables in Supabase.

Uses direct PostgreSQL connection (SQLAlchemy + psycopg2) to query
dbt materialized tables in the dbt_vor schema. This is preferred over
the Supabase REST client because:
1. dbt tables live in dbt_vor schema (not public), which PostgREST can't access
2. Direct SQL is faster for batch feature extraction
3. Consistent with Feast offline store access pattern
"""

import os
from pathlib import Path

import pandas as pd
import yaml
from sqlalchemy import create_engine, text


def load_config() -> dict:
    config_path = Path(__file__).parents[2] / "configs" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_db_engine(config: dict):
    dbt_config = config.get("dbt", {})
    schema = dbt_config.get("schema", "dbt_vor")

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        host = os.getenv("SUPABASE_DB_HOST")
        user = os.getenv("SUPABASE_DB_USER")
        password = os.getenv("SUPABASE_DB_PASSWORD")
        db_url = f"postgresql://{user}:{password}@{host}:5432/postgres?sslmode=require"

    engine = create_engine(
        db_url, connect_args={"options": f"-c search_path={schema},public"}
    )
    return engine, schema


def fetch_features_from_db(config: dict) -> pd.DataFrame:
    engine, schema = get_db_engine(config)

    print(f"Fetching training data from schema: {schema}")

    with engine.connect() as conn:
        agent_df = pd.read_sql(text(f"SELECT * FROM {schema}.fct_agent_features"), conn)
        product_df = pd.read_sql(
            text(f"SELECT * FROM {schema}.fct_product_features"), conn
        )
        interaction_df = pd.read_sql(
            text(f"SELECT * FROM {schema}.fct_agent_product_interactions"), conn
        )

    for df in [agent_df, product_df, interaction_df]:
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str)

    print(f"Agent features: {len(agent_df)} rows")
    print(f"Product features: {len(product_df)} rows")
    print(f"Interactions: {len(interaction_df)} rows")

    training_df = interaction_df.merge(
        agent_df, on="agent_id", how="left", suffixes=("", "_agent")
    )
    training_df = training_df.merge(
        product_df, on="product_id", how="left", suffixes=("", "_product")
    )

    if "purchase_count" in training_df.columns:
        training_df["purchased"] = (training_df["purchase_count"] > 0).astype(int)

    return training_df


def fetch_training_features() -> pd.DataFrame:
    config = load_config()
    paths = config["paths"]
    base_path = Path(__file__).parents[2]

    df = fetch_features_from_db(config)

    output_path = base_path / paths["training_features"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Training features saved to {output_path}")
    print(f"Shape: {df.shape}")

    return df


if __name__ == "__main__":
    fetch_training_features()
