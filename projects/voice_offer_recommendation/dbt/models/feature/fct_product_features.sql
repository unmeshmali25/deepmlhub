{{ config(materialized='table') }}

with products as (
    select * from {{ ref('stg_products') }}
),

sales_metrics as (
    select * from {{ ref('int_product_sales_metrics') }}
)

select
    sales.product_id,
    sales.product_category,
    sales.product_brand,
    sales.price,
    sales.margin_percent,
    sales.rating,
    sales.review_count,
    sales.in_stock,
    sales.order_count,
    sales.total_units_sold,
    sales.unique_buyers_count,
    sales.avg_selling_price,
    sales.avg_discount_given,
    sales.total_revenue,
    sales.available_quantity,
    sales.stores_available,
    case
        when sales.total_units_sold > 0
        then round(sales.total_revenue / sales.total_units_sold, 2)
        else sales.price
    end as revenue_per_unit,
    case
        when sales.unique_buyers_count > 0
        then round(sales.total_units_sold::numeric / sales.unique_buyers_count, 2)
        else 0
    end as avg_quantity_per_buyer
from sales