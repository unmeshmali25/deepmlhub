{{ config(materialized='table') }}

with products as (
    select * from {{ ref('stg_products') }}
),

sales_metrics as (
    select * from {{ ref('int_product_sales_metrics') }}
)

select
    sm.product_id,
    sm.product_category,
    sm.product_brand,
    sm.price,
    sm.margin_percent,
    sm.rating,
    sm.review_count,
    sm.in_stock,
    sm.order_count,
    sm.total_units_sold,
    sm.unique_buyers_count,
    sm.avg_selling_price,
    sm.avg_discount_given,
    sm.total_revenue,
    sm.available_quantity,
    sm.stores_available,
    case
        when sm.total_units_sold > 0
        then round(sm.total_revenue / sm.total_units_sold, 2)
        else sm.price
    end as revenue_per_unit,
    case
        when sm.unique_buyers_count > 0
        then round(sm.total_units_sold::numeric / sm.unique_buyers_count, 2)
        else 0
    end as avg_quantity_per_buyer
from sales_metrics sm