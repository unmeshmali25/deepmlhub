{{ config(materialized='view') }}

with source as (
    select * from {{ source('public', 'agent_state_snapshots') }}
),

with_derived as (
    select
        id as snapshot_uuid,
        cast(agent_id as varchar) as agent_id,
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
        price_sensitivity,
        brand_loyalty,
        impulsivity,
        tech_savviness,
        coupon_affinity,
        pref_day_weekday,
        pref_day_saturday,
        pref_day_sunday,
        pref_time_morning,
        pref_time_afternoon,
        pref_time_evening,
        case
            when weekly_budget > 0
            then round(remaining_budget / weekly_budget, 4)
            else 0
        end as remaining_budget_pct,
        case
            when weekly_budget > 0
            then round(spend_this_week / weekly_budget, 4)
            else 0
        end as spend_this_week_pct
    from source
)

select * from with_derived