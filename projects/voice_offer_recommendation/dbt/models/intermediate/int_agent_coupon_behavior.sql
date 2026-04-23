{{ config(materialized='table') }}

with coupon_interactions as (
    select * from {{ ref('stg_coupon_interactions') }}
),

agents as (
    select * from {{ ref('stg_agents') }}
),

user_coupons as (
    select * from {{ source('public', 'user_coupons') }}
),

interaction_agg as (
    select
        agents.agent_id,
        coupon_interactions.coupon_action,
        count(*) as action_count
    from coupon_interactions
    inner join agents on coupon_interactions.user_id = agents.user_id
    group by agents.agent_id, coupon_interactions.coupon_action
),

view_counts as (
    select agent_id, action_count as coupon_views
    from interaction_agg
    where coupon_action = 'added_to_cart'
),

click_counts as (
    select agent_id, 0 as coupon_clicks
    from (select distinct agent_id from agents) a
),

redeem_counts as (
    select agent_id, action_count as coupon_redeems
    from interaction_agg
    where coupon_action = 'redeemed'
),

category_coupon as (
    select
        agents.agent_id,
        coupon_interactions.coupon_category_or_brand,
        count(*) as category_coupon_count
    from coupon_interactions
    inner join agents on coupon_interactions.user_id = agents.user_id
    where coupon_interactions.coupon_category_or_brand is not null
    group by agents.agent_id, coupon_interactions.coupon_category_or_brand
),

redemption_agg as (
    select
        agents.agent_id,
        count(*) as total_coupons_assigned,
        count(case when user_coupons.status = 'used' then 1 end) as total_coupons_redeemed
    from user_coupons
    inner join agents on user_coupons.user_id = agents.user_id
    group by agents.agent_id
)

select
    agents.agent_id,
    coalesce(views.coupon_views, 0) as coupon_views,
    coalesce(clicks.coupon_clicks, 0) as coupon_clicks,
    coalesce(redeems.coupon_redeems, 0) as coupon_redeems,
    coalesce(redemptions.total_coupons_assigned, 0) as total_coupons_assigned,
    coalesce(redemptions.total_coupons_redeemed, 0) as total_coupons_redeemed,
    case
        when coalesce(redemptions.total_coupons_assigned, 0) > 0
        then round(
            coalesce(redemptions.total_coupons_redeemed, 0)::numeric /
            redemptions.total_coupons_assigned,
            4
        )
        else 0
    end as coupon_redemption_rate
from agents
left join view_counts views on agents.agent_id = views.agent_id
left join click_counts clicks on agents.agent_id = clicks.agent_id
left join redeem_counts redeems on agents.agent_id = redeems.agent_id
left join redemption_agg redemptions on agents.agent_id = redemptions.agent_id