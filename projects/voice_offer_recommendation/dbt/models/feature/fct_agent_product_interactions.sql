{{ config(materialized='table') }}

with purchases as (
    select * from {{ ref('int_agent_product_purchases') }}
),

category_affinity as (
    select * from {{ ref('int_agent_category_affinity') }}
),

products as (
    select * from {{ ref('stg_products') }}
),

coupon_behavior as (
    select * from {{ ref('int_agent_coupon_behavior') }}
),

-- Enrich purchases with product category for category affinity join
purchases_with_category as (
    select
        purchases.agent_id,
        purchases.product_id,
        purchases.purchase_count,
        purchases.last_purchase_date,
        purchases.total_spent,
        purchases.avg_discount_received,
        purchases.total_quantity as total_units_bought,
        products.product_category
    from purchases
    inner join products on purchases.product_id = products.product_uuid
)

select
    pwc.agent_id,
    pwc.product_id,
    pwc.purchase_count,
    pwc.last_purchase_date,
    pwc.total_spent,
    pwc.avg_discount_received,
    pwc.total_units_bought,
    pwc.product_category,
    coalesce(cat.category_purchase_count, 0) as category_purchase_count,
    coalesce(cat.category_total_spent, 0) as category_total_spent,
    coalesce(cb.total_coupons_assigned, 0) as total_coupons_assigned,
    coalesce(cb.total_coupons_redeemed, 0) as total_coupons_redeemed,
    coalesce(cb.coupon_redemption_rate, 0) as coupon_redemption_rate
from purchases_with_category pwc
left join category_affinity cat
    on pwc.agent_id = cat.agent_id
    and pwc.product_category = cat.product_category
left join coupon_behavior cb on pwc.agent_id = cb.agent_id