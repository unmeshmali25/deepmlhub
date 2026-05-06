{{ config(materialized='table') }}

with order_items as (
    select * from {{ ref('stg_order_items') }}
),

agents as (
    select * from {{ ref('stg_agents') }}
),

purchases as (
    select
        agents.agent_id,
        order_items.product_id,
        count(*) as purchase_count,
        max(order_items.order_created_at) as last_purchase_date,
        sum(order_items.line_total) as total_spent,
        avg(order_items.discount_amount) as avg_discount_received,
        sum(order_items.quantity) as total_quantity
    from order_items
    inner join agents on order_items.user_id = agents.user_id
    group by agents.agent_id, order_items.product_id
)

select * from purchases