{{ config(materialized='table') }}

with order_items as (
    select * from {{ ref('stg_order_items') }}
),

products as (
    select * from {{ ref('stg_products') }}
),

inventory as (
    select * from {{ source('public', 'store_inventory') }}
),

sales_metrics as (
    select
        order_items.product_id,
        count(distinct order_items.order_id) as order_count,
        sum(order_items.quantity) as total_units_sold,
        count(distinct order_items.user_id) as unique_buyers_count,
        avg(order_items.product_price) as avg_selling_price,
        avg(order_items.discount_amount) as avg_discount_given,
        sum(order_items.line_total) as total_revenue
    from order_items
    group by order_items.product_id
),

availability as (
    select
        product_id,
        sum(quantity) as available_quantity,
        count(distinct store_id) as stores_available
    from inventory
    group by product_id
)

select
    products.product_uuid as product_id,
    products.product_category,
    products.product_brand,
    products.price,
    products.margin_percent,
    products.rating,
    products.review_count,
    products.in_stock,
    coalesce(sales.order_count, 0) as order_count,
    coalesce(sales.total_units_sold, 0) as total_units_sold,
    coalesce(sales.unique_buyers_count, 0) as unique_buyers_count,
    coalesce(sales.avg_selling_price, products.price) as avg_selling_price,
    coalesce(sales.avg_discount_given, 0) as avg_discount_given,
    coalesce(sales.total_revenue, 0) as total_revenue,
    coalesce(avail.available_quantity, 0) as available_quantity,
    coalesce(avail.stores_available, 0) as stores_available
from products
left join sales_metrics sales on products.product_uuid = sales.product_id
left join availability avail on products.product_uuid = avail.product_id