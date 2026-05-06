{{ config(materialized='view') }}

with source as (
    select * from {{ source('public', 'agents') }}
),

cleaned as (
    select
        id as agent_uuid,
        agent_id,
        user_id,
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
        pref_day_weekday,
        pref_day_saturday,
        pref_day_sunday,
        pref_time_morning,
        pref_time_afternoon,
        pref_time_evening,
        coupon_affinity,
        deal_seeking_behavior,
        is_active,
        generated_at,
        updated_at as src_updated_at
    from source
    where is_active = true
)

select * from cleaned