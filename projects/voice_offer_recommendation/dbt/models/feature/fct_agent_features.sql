{{ config(materialized='table') }}

with agents as (
    select * from {{ ref('stg_agents') }}
),

snapshots as (
    select * from {{ ref('stg_agent_snapshots') }}
),

coupon_behavior as (
    select * from {{ ref('int_agent_coupon_behavior') }}
),

-- Use latest snapshot per agent for current feature values
latest_snapshot as (
    select distinct on (agent_id)
        agent_id,
        snapshot_date,
        remaining_budget,
        weekly_budget,
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
        spend_this_week_pct
    from snapshots
    order by agent_id, snapshot_date desc
),

combined as (
    select
        latest.agent_id,
        latest.snapshot_date,

        -- Static traits from agents
        agents.age,
        agents.age_group,
        agents.gender,
        agents.income_bracket,
        agents.household_size,
        agents.has_children,
        agents.location_region,
        agents.price_sensitivity,
        agents.brand_loyalty,
        agents.impulsivity,
        agents.tech_savviness,
        agents.preferred_categories,
        agents.weekly_budget as declared_weekly_budget,
        agents.shopping_frequency,
        agents.avg_cart_value as declared_avg_cart_value,
        agents.pref_day_weekday,
        agents.pref_day_saturday,
        agents.pref_day_sunday,
        agents.pref_time_morning,
        agents.pref_time_afternoon,
        agents.pref_time_evening,
        agents.coupon_affinity,
        agents.deal_seeking_behavior,

        -- Dynamic state from snapshots
        latest.remaining_budget,
        latest.spend_this_week,
        latest.spend_this_month,
        latest.days_since_last_purchase,
        latest.total_orders_lifetime,
        latest.total_orders_this_week,
        latest.cart_value,
        latest.cart_item_count,
        latest.has_active_cart,
        latest.products_viewed_count,
        latest.sessions_count_today,
        latest.sessions_count_this_week,
        latest.categories_purchased_count,
        latest.categories_purchased_this_week,
        latest.diversity_ratio,
        latest.pref_day_match,
        latest.pref_time_match,
        latest.active_coupons_count,
        latest.coupons_redeemed_this_week,
        latest.remaining_budget_pct,
        latest.spend_this_week_pct,

        -- Coupon behavior
        coalesce(coupon.coupon_views, 0) as coupon_views,
        coalesce(coupon.coupon_clicks, 0) as coupon_clicks,
        coalesce(coupon.coupon_redeems, 0) as coupon_redeems,
        coalesce(coupon.total_coupons_assigned, 0) as total_coupons_assigned,
        coalesce(coupon.total_coupons_redeemed, 0) as total_coupons_redeemed,
        coalesce(coupon.coupon_redemption_rate, 0) as coupon_redemption_rate
    from latest
    inner join agents on latest.agent_id = agents.agent_id
    left join coupon_behavior coupon on latest.agent_id = coupon.agent_id
)

select * from combined