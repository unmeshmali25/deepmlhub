"""Feast feature definitions for voice offer recommendation.

Points to dbt-generated feature tables in the dbt_vor schema.
Uses Supabase PostgreSQL as the data source.
"""

from datetime import timedelta

from feast import Entity, Feature, FeatureView, ValueType
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)
from feast.types import Float32, Int64, String

# Entities
agent = Entity(
    name="agent",
    value_type=ValueType.STRING,
    description="Customer/agent ID",
    join_key="agent_id",
)

product = Entity(
    name="product",
    value_type=ValueType.STRING,
    description="Product ID (UUID)",
    join_key="product_id",
)

# Data Sources — pointing to dbt feature tables
agent_features_source = PostgreSQLSource(
    name="agent_features_source",
    query="""
        SELECT
            agent_id,
            age,
            age_group,
            gender,
            income_bracket,
            household_size,
            has_children,
            location_region,
            price_sensitivity,
            brand_loyalty,
            impulsivity,
            tech_savviness,
            preferred_categories,
            weekly_budget,
            shopping_frequency,
            avg_cart_value,
            coupon_affinity,
            deal_seeking_behavior,
            remaining_budget_pct,
            spend_this_week_pct,
            days_since_last_purchase,
            total_orders_lifetime,
            total_orders_this_week,
            diversity_ratio,
            active_coupons_count,
            coupons_redeemed_this_week,
            coupon_views,
            coupon_clicks,
            coupon_redeems,
            coupon_redemption_rate,
            snapshot_date as event_timestamp
        FROM {{ var('dbt_schema', 'dbt_vor') }}.fct_agent_features
    """,
    timestamp_field="event_timestamp",
)

product_features_source = PostgreSQLSource(
    name="product_features_source",
    query="""
        SELECT
            product_id,
            product_category,
            product_brand,
            price,
            margin_percent,
            rating,
            review_count,
            in_stock,
            order_count,
            total_units_sold,
            unique_buyers_count,
            available_quantity,
            revenue_per_unit,
            avg_quantity_per_buyer,
            CURRENT_TIMESTAMP as event_timestamp
        FROM {{ var('dbt_schema', 'dbt_vor') }}.fct_product_features
    """,
    timestamp_field="event_timestamp",
)

agent_product_interaction_source = PostgreSQLSource(
    name="agent_product_interaction_source",
    query="""
        SELECT
            agent_id,
            product_id,
            purchase_count,
            last_purchase_date,
            total_spent,
            avg_discount_received,
            total_units_bought,
            category_purchase_count,
            category_total_spent,
            total_coupons_assigned,
            total_coupons_redeemed,
            coupon_redemption_rate,
            last_purchase_date as event_timestamp
        FROM {{ var('dbt_schema', 'dbt_vor') }}.fct_agent_product_interactions
    """,
    timestamp_field="event_timestamp",
)

# Feature Views
agent_features_view = FeatureView(
    name="agent_features",
    entities=[agent],
    ttl=timedelta(days=1),
    features=[
        Feature(name="age", dtype=Int64),
        Feature(name="price_sensitivity", dtype=Float32),
        Feature(name="brand_loyalty", dtype=Float32),
        Feature(name="impulsivity", dtype=Float32),
        Feature(name="tech_savviness", dtype=Float32),
        Feature(name="coupon_affinity", dtype=Float32),
        Feature(name="remaining_budget_pct", dtype=Float32),
        Feature(name="spend_this_week_pct", dtype=Float32),
        Feature(name="days_since_last_purchase", dtype=Int64),
        Feature(name="total_orders_lifetime", dtype=Int64),
        Feature(name="diversity_ratio", dtype=Float32),
        Feature(name="active_coupons_count", dtype=Int64),
        Feature(name="coupons_redeemed_this_week", dtype=Int64),
        Feature(name="coupon_redemption_rate", dtype=Float32),
    ],
    online=True,
    source=agent_features_source,
)

product_features_view = FeatureView(
    name="product_features",
    entities=[product],
    ttl=timedelta(days=1),
    features=[
        Feature(name="product_category", dtype=String),
        Feature(name="product_brand", dtype=String),
        Feature(name="price", dtype=Float32),
        Feature(name="margin_percent", dtype=Float32),
        Feature(name="rating", dtype=Float32),
        Feature(name="review_count", dtype=Int64),
        Feature(name="in_stock", dtype=Int64),
        Feature(name="total_units_sold", dtype=Int64),
        Feature(name="unique_buyers_count", dtype=Int64),
        Feature(name="available_quantity", dtype=Int64),
    ],
    online=True,
    source=product_features_source,
)

agent_product_interaction_view = FeatureView(
    name="agent_product_interaction",
    entities=[agent, product],
    ttl=timedelta(days=1),
    features=[
        Feature(name="purchase_count", dtype=Int64),
        Feature(name="total_spent", dtype=Float32),
        Feature(name="avg_discount_received", dtype=Float32),
        Feature(name="total_units_bought", dtype=Int64),
        Feature(name="category_purchase_count", dtype=Int64),
        Feature(name="coupon_redemption_rate", dtype=Float32),
    ],
    online=True,
    source=agent_product_interaction_source,
)
