{{ config(materialized='table') }}

with order_items as (
    select * from {{ ref('stg_order_items') }}
),

agents as (
    select * from {{ ref('stg_agents') }}
),

products as (
    select * from {{ ref('stg_products') }}
),

category_agg as (
    select
        agents.agent_id,
        products.product_category,
        count(*) as category_purchase_count,
        sum(order_items.line_total) as category_total_spent,
        sum(order_items.quantity) as category_total_quantity
    from order_items
    inner join agents on order_items.user_id = agents.user_id
    inner join products on order_items.product_id = products.product_id
    group by agents.agent_id, products.product_category
)

select * from category_agg