{{ config(materialized='view') }}

with items as (
    select * from {{ source('public', 'order_items') }}
),

orders as (
    select * from {{ ref('stg_orders') }}
),

enriched as (
    select
        items.id as order_item_id,
        items.order_id,
        orders.user_id,
        orders.order_created_at,
        cast(items.product_id as varchar) as product_id,
        items.product_name,
        items.product_price,
        items.quantity,
        items.applied_coupon_id,
        items.discount_amount,
        items.line_total,
        items.created_at as item_created_at
    from items
    inner join orders on items.order_id = orders.order_id
)

select * from enriched