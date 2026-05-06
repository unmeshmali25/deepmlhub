{{ config(materialized='view') }}

with interactions as (
    select * from {{ source('public', 'coupon_interactions') }}
),

coupons as (
    select * from {{ source('public', 'coupons') }}
),

enriched as (
    select
        interactions.id as interaction_id,
        interactions.user_id,
        interactions.coupon_id,
        interactions.action as coupon_action,
        interactions.order_id,
        interactions.created_at as interaction_created_at,
        coupons.type as coupon_type,
        coupons.category_or_brand as coupon_category_or_brand,
        coupons.discount_type,
        coupons.discount_value,
        coupons.min_purchase_amount,
        coupons.max_discount
    from interactions
    left join coupons on interactions.coupon_id = coupons.id
)

select * from enriched