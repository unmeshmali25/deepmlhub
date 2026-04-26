"""Feast feature definitions for voice offer recommendation.

Points to dbt-generated feature tables in the dbt_vor schema.
Uses Supabase PostgreSQL as the data source.

Column coverage: ALL 52 agent + 18 product + 13 interaction columns
from the dbt feature tables are defined here.

Compatible with Feast 0.40+ (Field API, join_keys).
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field, ValueType
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)
from feast.types import Float32, Int64, String

_SOURCE_SCHEMA = "dbt_vor"

agent = Entity(
    name="agent",
    join_keys=["agent_id"],
    value_type=ValueType.STRING,
    description="Customer/agent ID",
)

product = Entity(
    name="product",
    join_keys=["product_id"],
    value_type=ValueType.STRING,
    description="Product ID",
)

# ---------------------------------------------------------------------------
# Data Sources — direct SQL queries against dbt materialized tables
# ---------------------------------------------------------------------------

agent_features_source = PostgreSQLSource(
    name="agent_features_source",
    query=f"""
        SELECT
            agent_id,
            snapshot_date,
            snapshot_date AS event_timestamp,
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
            declared_weekly_budget,
            shopping_frequency,
            declared_avg_cart_value,
            pref_day_weekday,
            pref_day_saturday,
            pref_day_sunday,
            pref_time_morning,
            pref_time_afternoon,
            pref_time_evening,
            coupon_affinity,
            deal_seeking_behavior,
            remaining_budget,
            spend_this_week,
            spend_this_month,
            days_since_last_purchase,
            total_orders_lifetime,
            total_orders_this_week,
            cart_value,
            cart_item_count,
            has_active_cart,
            products_viewed_count,
            sessions_count_today,
            sessions_count_this_week,
            categories_purchased_count,
            categories_purchased_this_week,
            diversity_ratio,
            pref_day_match,
            pref_time_match,
            active_coupons_count,
            coupons_redeemed_this_week,
            remaining_budget_pct,
            spend_this_week_pct,
            coupon_views,
            coupon_clicks,
            coupon_redeems,
            total_coupons_assigned,
            total_coupons_redeemed,
            coupon_redemption_rate
        FROM {_SOURCE_SCHEMA}.fct_agent_features
    """,
    timestamp_field="event_timestamp",
)

product_features_source = PostgreSQLSource(
    name="product_features_source",
    query=f"""
        SELECT
            product_id,
            '2020-01-01 00:00:00'::timestamp AS event_timestamp,
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
            avg_selling_price,
            avg_discount_given,
            total_revenue,
            available_quantity,
            stores_available,
            revenue_per_unit,
            avg_quantity_per_buyer
        FROM {_SOURCE_SCHEMA}.fct_product_features
    """,
    timestamp_field="event_timestamp",
)

agent_product_interaction_source = PostgreSQLSource(
    name="agent_product_interaction_source",
    query=f"""
        SELECT
            agent_id,
            product_id,
            last_purchase_date,
            last_purchase_date AS event_timestamp,
            purchase_count,
            total_spent,
            avg_discount_received,
            total_units_bought,
            product_category,
            category_purchase_count,
            category_total_spent,
            total_coupons_assigned,
            total_coupons_redeemed,
            coupon_redemption_rate
        FROM {_SOURCE_SCHEMA}.fct_agent_product_interactions
    """,
    timestamp_field="event_timestamp",
)

# ---------------------------------------------------------------------------
# Feature Views
# ---------------------------------------------------------------------------

agent_features_view = FeatureView(
    name="agent_features",
    entities=[agent],
    ttl=timedelta(days=36500),
    schema=[
        Field(name="snapshot_date", dtype=String),
        Field(name="age", dtype=Int64),
        Field(name="age_group", dtype=String),
        Field(name="gender", dtype=String),
        Field(name="income_bracket", dtype=String),
        Field(name="household_size", dtype=Int64),
        Field(name="has_children", dtype=Int64),
        Field(name="location_region", dtype=String),
        Field(name="price_sensitivity", dtype=Float32),
        Field(name="brand_loyalty", dtype=Float32),
        Field(name="impulsivity", dtype=Float32),
        Field(name="tech_savviness", dtype=Float32),
        Field(name="preferred_categories", dtype=String),
        Field(name="declared_weekly_budget", dtype=Float32),
        Field(name="shopping_frequency", dtype=String),
        Field(name="declared_avg_cart_value", dtype=Float32),
        Field(name="pref_day_weekday", dtype=Int64),
        Field(name="pref_day_saturday", dtype=Int64),
        Field(name="pref_day_sunday", dtype=Int64),
        Field(name="pref_time_morning", dtype=Int64),
        Field(name="pref_time_afternoon", dtype=Int64),
        Field(name="pref_time_evening", dtype=Int64),
        Field(name="coupon_affinity", dtype=Float32),
        Field(name="deal_seeking_behavior", dtype=Float32),
        Field(name="remaining_budget", dtype=Float32),
        Field(name="spend_this_week", dtype=Float32),
        Field(name="spend_this_month", dtype=Float32),
        Field(name="days_since_last_purchase", dtype=Int64),
        Field(name="total_orders_lifetime", dtype=Int64),
        Field(name="total_orders_this_week", dtype=Int64),
        Field(name="cart_value", dtype=Float32),
        Field(name="cart_item_count", dtype=Int64),
        Field(name="has_active_cart", dtype=Int64),
        Field(name="products_viewed_count", dtype=Int64),
        Field(name="sessions_count_today", dtype=Int64),
        Field(name="sessions_count_this_week", dtype=Int64),
        Field(name="categories_purchased_count", dtype=Int64),
        Field(name="categories_purchased_this_week", dtype=Int64),
        Field(name="diversity_ratio", dtype=Float32),
        Field(name="pref_day_match", dtype=Float32),
        Field(name="pref_time_match", dtype=Float32),
        Field(name="active_coupons_count", dtype=Int64),
        Field(name="coupons_redeemed_this_week", dtype=Int64),
        Field(name="remaining_budget_pct", dtype=Float32),
        Field(name="spend_this_week_pct", dtype=Float32),
        Field(name="coupon_views", dtype=Int64),
        Field(name="coupon_clicks", dtype=Int64),
        Field(name="coupon_redeems", dtype=Int64),
        Field(name="total_coupons_assigned", dtype=Int64),
        Field(name="total_coupons_redeemed", dtype=Int64),
        Field(name="coupon_redemption_rate", dtype=Float32),
    ],
    online=False,
    source=agent_features_source,
)

product_features_view = FeatureView(
    name="product_features",
    entities=[product],
    ttl=timedelta(days=36500),
    schema=[
        Field(name="product_category", dtype=String),
        Field(name="product_brand", dtype=String),
        Field(name="price", dtype=Float32),
        Field(name="margin_percent", dtype=Float32),
        Field(name="rating", dtype=Float32),
        Field(name="review_count", dtype=Int64),
        Field(name="in_stock", dtype=Int64),
        Field(name="order_count", dtype=Int64),
        Field(name="total_units_sold", dtype=Int64),
        Field(name="unique_buyers_count", dtype=Int64),
        Field(name="avg_selling_price", dtype=Float32),
        Field(name="avg_discount_given", dtype=Float32),
        Field(name="total_revenue", dtype=Float32),
        Field(name="available_quantity", dtype=Int64),
        Field(name="stores_available", dtype=Int64),
        Field(name="revenue_per_unit", dtype=Float32),
        Field(name="avg_quantity_per_buyer", dtype=Float32),
    ],
    online=False,
    source=product_features_source,
)

agent_product_interaction_view = FeatureView(
    name="agent_product_interaction",
    entities=[agent, product],
    ttl=timedelta(days=36500),
    schema=[
        Field(name="last_purchase_date", dtype=String),
        Field(name="purchase_count", dtype=Int64),
        Field(name="total_spent", dtype=Float32),
        Field(name="avg_discount_received", dtype=Float32),
        Field(name="total_units_bought", dtype=Int64),
        Field(name="product_category", dtype=String),
        Field(name="category_purchase_count", dtype=Int64),
        Field(name="category_total_spent", dtype=Float32),
        Field(name="total_coupons_assigned", dtype=Int64),
        Field(name="total_coupons_redeemed", dtype=Int64),
        Field(name="coupon_redemption_rate", dtype=Float32),
    ],
    online=False,
    source=agent_product_interaction_source,
)
