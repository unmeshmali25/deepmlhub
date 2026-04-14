"""Feast feature definitions for voice offer recommendation.

Uses Supabase PostgreSQL as the data source.
The online store is also Supabase PostgreSQL (zero additional cost).
"""

from datetime import timedelta

from feast import Entity, Feature, FeatureView, ValueType
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)
from feast.types import Float32, Int64

# Entities
agent = Entity(
    name="agent",
    value_type=ValueType.INT64,
    description="Customer/agent ID",
    join_key="agent_id",
)

product = Entity(
    name="product",
    value_type=ValueType.INT64,
    description="Product ID",
    join_key="product_id",
)

# Data Sources (pointing to Supabase)
# NOTE: These assume the Supabase tables exist.
# Adjust table/column names to match your actual schema.

agent_features_source = PostgreSQLSource(
    name="agent_features_source",
    query="""
        SELECT
            agent_id,
            total_orders,
            total_spend,
            favorite_category,
            event_timestamp
        FROM public.agent_stats
    """,
    timestamp_field="event_timestamp",
)

product_features_source = PostgreSQLSource(
    name="product_features_source",
    query="""
        SELECT
            product_id,
            order_count_7d,
            order_count_30d,
            avg_discount,
            event_timestamp
        FROM public.product_stats
    """,
    timestamp_field="event_timestamp",
)

agent_product_interaction_source = PostgreSQLSource(
    name="agent_product_interaction_source",
    query="""
        SELECT
            agent_id,
            product_id,
            times_viewed,
            times_added_to_cart,
            times_purchased,
            last_interaction_timestamp as event_timestamp
        FROM public.agent_product_interactions
    """,
    timestamp_field="event_timestamp",
)

# Feature Views
agent_features_view = FeatureView(
    name="agent_features",
    entities=[agent],
    ttl=timedelta(days=1),
    features=[
        Feature(name="total_orders", dtype=Int64),
        Feature(name="total_spend", dtype=Float32),
        Feature(name="favorite_category", dtype=Int64),
    ],
    online=True,
    source=agent_features_source,
)

product_features_view = FeatureView(
    name="product_features",
    entities=[product],
    ttl=timedelta(days=1),
    features=[
        Feature(name="order_count_7d", dtype=Int64),
        Feature(name="order_count_30d", dtype=Int64),
        Feature(name="avg_discount", dtype=Float32),
    ],
    online=True,
    source=product_features_source,
)

agent_product_interaction_view = FeatureView(
    name="agent_product_interaction",
    entities=[agent, product],
    ttl=timedelta(days=1),
    features=[
        Feature(name="times_viewed", dtype=Int64),
        Feature(name="times_added_to_cart", dtype=Int64),
        Feature(name="times_purchased", dtype=Int64),
    ],
    online=True,
    source=agent_product_interaction_source,
)
