{{ config(materialized='view') }}

with source as (
    select * from {{ source('public', 'products') }}
),

cleaned as (
    select
        cast(id as varchar) as product_id,
        id as product_uuid,
        name as product_name,
        description,
        image_url,
        price,
        rating,
        review_count,
        category as product_category,
        brand as product_brand,
        promo_text,
        in_stock,
        cost,
        margin_percent,
        created_at as product_created_at,
        updated_at as product_updated_at
    from source
)

select * from cleaned