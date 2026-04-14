# Supabase Database Schema Summary

**Generated**: 2026-04-14  
**Database**: Supabase PostgreSQL (project: xrzpyapwnygdcjmcmnxg)

---

## Overview

- **22 tables** (17 data tables + 5 views)
- **~2.1M total rows** across all tables
- **371 simulated agents** driving shopping behavior
- **72 products** across **10 stores**

---

## Core Entities

### `users` — 384 rows, 5 cols

| Column | Type | Nullable |
|--------|------|----------|
| id | uuid | NOT NULL |
| email | text | NOT NULL |
| full_name | text | NULL |
| created_at | timestamp | NULL |
| updated_at | timestamp | NULL |

### `agents` — 371 rows, 33 cols

| Column | Type | Nullable |
|--------|------|----------|
| id | uuid | NOT NULL |
| agent_id | varchar | NOT NULL |
| user_id | uuid | NULL |
| generation_model | varchar | NULL |
| generated_at | timestamp | NULL |
| age | integer | NULL |
| age_group | varchar | NULL |
| gender | varchar | NULL |
| income_bracket | varchar | NULL |
| household_size | integer | NULL |
| has_children | boolean | NULL |
| location_region | varchar | NULL |
| price_sensitivity | numeric | NULL |
| brand_loyalty | numeric | NULL |
| impulsivity | numeric | NULL |
| tech_savviness | numeric | NULL |
| preferred_categories | text | NULL |
| weekly_budget | numeric | NULL |
| shopping_frequency | varchar | NULL |
| avg_cart_value | numeric | NULL |
| pref_day_weekday | numeric | NULL |
| pref_day_saturday | numeric | NULL |
| pref_day_sunday | numeric | NULL |
| pref_time_morning | numeric | NULL |
| pref_time_afternoon | numeric | NULL |
| pref_time_evening | numeric | NULL |
| coupon_affinity | numeric | NULL |
| deal_seeking_behavior | varchar | NULL |
| backstory | text | NULL |
| sample_shopping_patterns | text | NULL |
| is_active | boolean | NULL |
| created_at | timestamp | NULL |
| updated_at | timestamp | NULL |

### `products` — 72 rows, 16 cols

| Column | Type | Nullable |
|--------|------|----------|
| id | uuid | NOT NULL |
| name | varchar | NOT NULL |
| description | text | NULL |
| image_url | text | NOT NULL |
| price | numeric | NOT NULL |
| rating | numeric | NULL |
| review_count | integer | NULL |
| category | varchar | NULL |
| brand | varchar | NULL |
| promo_text | varchar | NULL |
| in_stock | boolean | NULL |
| text_vector | tsvector | NULL |
| created_at | timestamp | NULL |
| updated_at | timestamp | NULL |
| cost | numeric | NULL |
| margin_percent | numeric | NULL |

### `stores` — 10 rows, 3 cols

| Column | Type | Nullable |
|--------|------|----------|
| id | uuid | NOT NULL |
| name | varchar | NOT NULL |
| created_at | timestamp | NULL |

### `coupons` — 257 rows, 14 cols

| Column | Type | Nullable |
|--------|------|----------|
| id | uuid | NOT NULL |
| type | varchar | NOT NULL |
| discount_details | text | NOT NULL |
| category_or_brand | varchar | NULL |
| expiration_date | timestamp | NOT NULL |
| terms | text | NULL |
| text_vector | tsvector | NULL |
| created_at | timestamp | NULL |
| updated_at | timestamp | NULL |
| discount_type | varchar | NULL |
| discount_value | numeric | NULL |
| min_purchase_amount | numeric | NULL |
| max_discount | numeric | NULL |
| is_active | boolean | NULL |

---

## Simulation / Interaction Data

### `shopping_sessions` — 162,751 rows, 8 cols

| Column | Type | Nullable |
|--------|------|----------|
| id | uuid | NOT NULL |
| user_id | uuid | NOT NULL |
| store_id | uuid | NULL |
| status | varchar | NOT NULL |
| started_at | timestamp | NOT NULL |
| last_seen_at | timestamp | NOT NULL |
| ended_at | timestamp | NULL |
| is_simulated | boolean | NULL |

### `shopping_session_events` — 1,418,550 rows, 7 cols

| Column | Type | Nullable |
|--------|------|----------|
| id | uuid | NOT NULL |
| session_id | uuid | NOT NULL |
| user_id | uuid | NOT NULL |
| event_type | varchar | NOT NULL |
| payload | jsonb | NOT NULL |
| created_at | timestamp | NOT NULL |
| is_simulated | boolean | NULL |

### `orders` — 92,550 rows, 12 cols

| Column | Type | Nullable |
|--------|------|----------|
| id | uuid | NOT NULL |
| user_id | uuid | NOT NULL |
| store_id | uuid | NOT NULL |
| subtotal | numeric | NOT NULL |
| discount_total | numeric | NOT NULL |
| final_total | numeric | NOT NULL |
| status | varchar | NOT NULL |
| created_at | timestamp | NULL |
| updated_at | timestamp | NULL |
| shopping_session_id | uuid | NULL |
| is_simulated | boolean | NULL |
| item_count | integer | NOT NULL |

### `order_items` — 150,509 rows, 10 cols

| Column | Type | Nullable |
|--------|------|----------|
| id | uuid | NOT NULL |
| order_id | uuid | NOT NULL |
| product_id | uuid | NOT NULL |
| product_name | varchar | NOT NULL |
| product_price | numeric | NOT NULL |
| quantity | integer | NOT NULL |
| applied_coupon_id | uuid | NULL |
| discount_amount | numeric | NOT NULL |
| line_total | numeric | NOT NULL |
| created_at | timestamp | NULL |

### `coupon_interactions` — 101,912 rows, 6 cols

| Column | Type | Nullable |
|--------|------|----------|
| id | uuid | NOT NULL |
| user_id | uuid | NOT NULL |
| coupon_id | uuid | NOT NULL |
| action | varchar | NOT NULL |
| order_id | uuid | NULL |
| created_at | timestamp | NULL |

### `user_coupons` — 96,211 rows, 10 cols

| Column | Type | Nullable |
|--------|------|----------|
| id | integer | NOT NULL |
| user_id | uuid | NOT NULL |
| coupon_id | uuid | NOT NULL |
| assigned_at | timestamp | NULL |
| eligible_until | timestamp | NULL |
| status | varchar | NULL |
| offer_cycle_id | uuid | NULL |
| is_simulation | boolean | NULL |
| order_id | uuid | NULL |
| redeemed_at | timestamptz | NULL |

### `llm_decisions` — 721 rows, 21 cols

| Column | Type | Nullable |
|--------|------|----------|
| id | integer | NOT NULL |
| agent_id | varchar | NOT NULL |
| decision_type | varchar | NOT NULL |
| llm_tier | varchar | NOT NULL |
| simulated_timestamp | timestamp | NOT NULL |
| context_hash | varchar | NOT NULL |
| decision_context | jsonb | NOT NULL |
| prompt_text | text | NOT NULL |
| raw_llm_response | text | NOT NULL |
| decision | boolean | NOT NULL |
| confidence | double precision | NULL |
| reasoning | text | NULL |
| urgency | double precision | NULL |
| cache_hit | boolean | NULL |
| latency_ms | integer | NULL |
| tokens_input | integer | NULL |
| tokens_output | integer | NULL |
| model_name | varchar | NULL |
| created_at | timestamp | NULL |
| simulation_id | varchar | NULL |
| llm_provider | varchar | NULL |

---

## Feature / Analytics Tables

### `agent_state_snapshots` — 107,961 rows, 36 cols

| Column | Type | Nullable |
|--------|------|----------|
| id | uuid | NOT NULL |
| agent_id | uuid | NOT NULL |
| snapshot_date | date | NOT NULL |
| remaining_budget | numeric | NOT NULL |
| weekly_budget | numeric | NOT NULL |
| spend_this_week | numeric | NOT NULL |
| spend_this_month | numeric | NOT NULL |
| days_since_last_purchase | integer | NULL |
| total_orders_lifetime | integer | NOT NULL |
| total_orders_this_week | integer | NOT NULL |
| cart_value | numeric | NOT NULL |
| cart_item_count | integer | NOT NULL |
| has_active_cart | boolean | NOT NULL |
| products_viewed_count | integer | NOT NULL |
| sessions_count_today | integer | NOT NULL |
| sessions_count_this_week | integer | NOT NULL |
| categories_purchased_count | integer | NOT NULL |
| categories_purchased_this_week | integer | NOT NULL |
| diversity_ratio | numeric | NULL |
| pref_day_match | boolean | NULL |
| pref_time_match | boolean | NULL |
| active_coupons_count | integer | NOT NULL |
| coupons_redeemed_this_week | integer | NOT NULL |
| price_sensitivity | numeric | NULL |
| brand_loyalty | numeric | NULL |
| impulsivity | numeric | NULL |
| tech_savviness | numeric | NULL |
| coupon_affinity | numeric | NULL |
| pref_day_weekday | numeric | NULL |
| pref_day_saturday | numeric | NULL |
| pref_day_sunday | numeric | NULL |
| pref_time_morning | numeric | NULL |
| pref_time_afternoon | numeric | NULL |
| pref_time_evening | numeric | NULL |
| created_at | timestamp | NULL |
| updated_at | timestamp | NULL |

### `agent_daily_rewards` — 107,961 rows, 17 cols

| Column | Type | Nullable |
|--------|------|----------|
| id | uuid | NOT NULL |
| agent_id | uuid | NOT NULL |
| reward_date | date | NOT NULL |
| reward_weights_id | integer | NULL |
| profit_reward | numeric | NULL |
| satisfaction_reward | numeric | NULL |
| frequency_reward | numeric | NULL |
| diversity_reward | numeric | NULL |
| total_revenue | numeric | NULL |
| total_cost | numeric | NULL |
| total_profit | numeric | NULL |
| avg_discount_rate | numeric | NULL |
| days_since_last_purchase | integer | NULL |
| unique_categories_count | integer | NULL |
| total_reward | numeric | NULL |
| created_at | timestamp | NULL |
| updated_at | timestamp | NULL |

---

## Reference / Config Tables

### `offer_cycles` — 2 rows, 9 cols

| Column | Type | Nullable |
|--------|------|----------|
| id | uuid | NOT NULL |
| cycle_number | integer | NOT NULL |
| started_at | timestamp | NOT NULL |
| ends_at | timestamp | NOT NULL |
| simulated_start_date | date | NULL |
| simulated_end_date | date | NULL |
| is_simulation | boolean | NOT NULL |
| created_at | timestamp | NULL |
| holdout_percentage | numeric | NULL |

### `reward_weights` — 1 row, 13 cols

| Column | Type | Nullable |
|--------|------|----------|
| id | integer | NOT NULL |
| name | varchar | NOT NULL |
| profit_weight | numeric | NOT NULL |
| satisfaction_weight | numeric | NOT NULL |
| frequency_weight | numeric | NOT NULL |
| diversity_weight | numeric | NOT NULL |
| profit_margin_threshold | numeric | NULL |
| satisfaction_discount_threshold | numeric | NULL |
| frequency_target_days | integer | NULL |
| diversity_target_ratio | numeric | NULL |
| is_active | boolean | NULL |
| created_at | timestamp | NULL |
| updated_at | timestamp | NULL |

### `user_offer_cycles` — 371 rows, 9 cols

| Column | Type | Nullable |
|--------|------|----------|
| id | uuid | NOT NULL |
| user_id | uuid | NOT NULL |
| current_cycle_id | uuid | NULL |
| last_refresh_at | timestamp | NULL |
| next_refresh_at | timestamp | NULL |
| is_simulation | boolean | NOT NULL |
| created_at | timestamp | NULL |
| updated_at | timestamp | NULL |
| is_holdout | boolean | NULL |

### `store_inventory` — 720 rows, 6 cols

| Column | Type | Nullable |
|--------|------|----------|
| id | uuid | NOT NULL |
| store_id | uuid | NOT NULL |
| product_id | uuid | NOT NULL |
| quantity | integer | NOT NULL |
| created_at | timestamp | NULL |
| updated_at | timestamp | NULL |

### `user_preferences` — 8 rows, 4 cols

| Column | Type | Nullable |
|--------|------|----------|
| user_id | uuid | NOT NULL |
| selected_store_id | uuid | NULL |
| created_at | timestamp | NULL |
| updated_at | timestamp | NULL |

---

## Empty Tables (0 rows)

| Table | Cols | Purpose |
|-------|------|---------|
| `cart_items` | 7 | Cart line items |
| `cart_coupons` | 4 | Cart-level coupon associations |
| `coupon_usage` | 6 | View/redeem tracking |
| `user_attributes` | 5 | Generic KV attributes |
| `simulation_state` | 9 | Simulation clock state |

---

## Views

### `v_llm_cache_metrics` — 3 rows, 8 cols

| Column | Type |
|--------|------|
| llm_tier | varchar |
| decision_type | varchar |
| total_decisions | bigint |
| cache_hits | bigint |
| cache_misses | bigint |
| cache_hit_rate_pct | numeric |
| avg_latency_ms_no_cache | numeric |
| avg_latency_ms_cache | numeric |

### `v_llm_decision_summary` — 12 rows, 10 cols

| Column | Type |
|--------|------|
| hour | timestamp |
| llm_tier | varchar |
| decision_type | varchar |
| total_decisions | bigint |
| positive_decisions | bigint |
| negative_decisions | bigint |
| avg_confidence | double precision |
| avg_latency_ms | numeric |
| avg_tokens_input | numeric |
| avg_tokens_output | numeric |

### `v_offer_cycle_uplift` — 2 rows, 16 cols

| Column | Type |
|--------|------|
| offer_cycle_id | uuid |
| cycle_number | integer |
| simulated_start_date | date |
| simulated_end_date | date |
| is_holdout | boolean |
| user_count | bigint |
| total_orders | numeric |
| avg_orders_per_user | numeric |
| total_revenue | numeric |
| avg_revenue_per_user | numeric |
| avg_order_value | numeric |
| total_discounts | numeric |
| avg_discounts_per_user | numeric |
| total_coupons_sent | numeric |
| total_coupons_redeemed | numeric |
| avg_redemption_rate | numeric |

### `v_uplift_summary` — 1 row, 11 cols

| Column | Type |
|--------|------|
| avg_orders_treatment | numeric |
| avg_orders_holdout | numeric |
| orders_uplift_pct | numeric |
| avg_revenue_treatment | numeric |
| avg_revenue_holdout | numeric |
| revenue_uplift_pct | numeric |
| avg_order_value_treatment | numeric |
| avg_order_value_holdout | numeric |
| aov_uplift_pct | numeric |
| avg_redemption_rate_treatment | numeric |
| avg_redemption_rate_holdout | numeric |

### `v_margin_uplift` — 0 rows, 9 cols

| Column | Type |
|--------|------|
| treatment_orders | bigint |
| holdout_orders | bigint |
| treatment_avg_margin | numeric |
| holdout_avg_margin | numeric |
| margin_uplift_pct | numeric |
| treatment_total_margin | numeric |
| holdout_total_margin | numeric |
| treatment_avg_discount | numeric |
| holdout_avg_discount | numeric |

---

## Row Count Summary

| Table | Rows |
|-------|-----:|
| shopping_session_events | 1,418,550 |
| shopping_sessions | 162,751 |
| order_items | 150,509 |
| agent_state_snapshots | 107,961 |
| agent_daily_rewards | 107,961 |
| coupon_interactions | 101,912 |
| user_coupons | 96,211 |
| orders | 92,550 |
| users | 384 |
| agents | 371 |
| user_offer_cycles | 371 |
| llm_decisions | 721 |
| coupons | 257 |
| products | 72 |
| store_inventory | 720 |
| stores | 10 |
| user_preferences | 8 |
| offer_cycles | 2 |
| reward_weights | 1 |
| cart_items | 0 |
| cart_coupons | 0 |
| coupon_usage | 0 |
| user_attributes | 0 |
| simulation_state | 0 |