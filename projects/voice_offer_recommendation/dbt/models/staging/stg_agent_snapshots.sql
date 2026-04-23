{{ config(materialized='view') }}

with source as (
    select * from {{ source('public', 'agent_state_snapshots') }}
),

agents as (
    select id, agent_id from {{ source('public', 'agents') }}
),

with_derived as (
    select
        source.id as snapshot_uuid,
        agents.agent_id,
        source.snapshot_date,
        source.remaining_budget,
        source.weekly_budget,
        source.spend_this_week,
        source.spend_this_month,
        source.days_since_last_purchase,
        source.total_orders_lifetime,
        source.total_orders_this_week,
        source.cart_value,
        source.cart_item_count,
        source.has_active_cart,
        source.products_viewed_count,
        source.sessions_count_today,
        source.sessions_count_this_week,
        source.categories_purchased_count,
        source.categories_purchased_this_week,
        source.diversity_ratio,
        source.pref_day_match,
        source.pref_time_match,
        source.active_coupons_count,
        source.coupons_redeemed_this_week,
        source.price_sensitivity,
        source.brand_loyalty,
        source.impulsivity,
        source.tech_savviness,
        source.coupon_affinity,
        source.pref_day_weekday,
        source.pref_day_saturday,
        source.pref_day_sunday,
        source.pref_time_morning,
        source.pref_time_afternoon,
        source.pref_time_evening,
        case
            when source.weekly_budget > 0
            then round(source.remaining_budget / source.weekly_budget, 4)
            else 0
        end as remaining_budget_pct,
        case
            when source.weekly_budget > 0
            then round(source.spend_this_week / source.weekly_budget, 4)
            else 0
        end as spend_this_week_pct
    from source
    inner join agents on source.agent_id = agents.id
)

select * from with_derived