{{ config(materialized='view') }}

with source as (
    select * from {{ source('public', 'orders') }}
),

cleaned as (
    select
        id as order_id,
        user_id,
        store_id,
        subtotal,
        discount_total,
        final_total,
        status as order_status,
        created_at as order_created_at,
        updated_at as order_updated_at,
        shopping_session_id,
        is_simulated,
        item_count
    from source
    where status in ('completed', 'delivered', 'picked_up')
)

select * from cleaned