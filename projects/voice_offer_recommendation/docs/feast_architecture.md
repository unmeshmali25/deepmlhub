# Feast Feature Store - Simple Flow

## Two Independent Paths from Source

```
┌─────────────────────────────────────────────────────────────────┐
│                     SUPABASE (Source of Truth)                   │
│            agents, products, orders, shopping_sessions          │
└──────────────────┬──────────────────────────────┬───────────────┘
                   │                              │
                   │                              │
        ┌──────────▼──────────┐        ┌─────────▼──────────┐
        │  MATERIALIZATION    │        │  TRAINING QUERY    │
        │      JOB            │        │  (get_historical_) │
        │                     │        │                    │
        │  - Scheduled hourly │        │  - Runs on demand  │
        │  - Executes SQL     │        │  - Point-in-time   │
        │    against source   │        │  - For model       │
        │  - Computes latest  │        │    training        │
        │    feature values   │        │                    │
        └──────────┬──────────┘        └─────────┬──────────┘
                   │                              │
                   ▼                              ▼
        ┌──────────────────┐          ┌──────────────────┐
        │   ONLINE STORE   │          │  OFFLINE STORE   │
        │    (Redis)       │          │ (PostgreSQL/BQ)  │
        │                  │          │                  │
        │  - <10ms lookup  │          │  - Historical    │
        │  - For serving   │          │    data          │
        │  - Always fresh  │          │  - May be stale  │
        └──────────┬───────┘          └──────────┬───────┘
                   │                              │
                   │                              │
                   ▼                              ▼
        ┌──────────────────┐          ┌──────────────────┐
        │     SERVING      │          │     TRAINING     │
        │   (Real-time)    │          │   (Batch)        │
        │                  │          │                  │
        │ get_online_      │          │ get_historical_  │
        │ features()       │          │ features()       │
        └──────────────────┘          └──────────────────┘
```

## Key Insight

**Materialization reads from SOURCE, not offline store.**

The materialization job runs Feature Definition SQL directly against your Supabase:
```sql
-- This runs every hour against Supabase
SELECT product_id, COUNT(*) as order_count_7d
FROM orders
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY product_id
```

Then pushes results to Redis for fast serving.

## Independence

| Component | Updated By | Used For | Staleness |
|-----------|-----------|----------|-----------|
| **Online Store** | Materialization job (hourly) | Real-time serving | 1 hour max |
| **Offline Store** | Training queries (on-demand) | Model training | Could be months |

**Critical**: Running online inference for 6 months without training does NOT populate offline store. When you want to train later, you must explicitly query historical features.

## When to Use Feast

**Use Feast if:**
- Need <10ms serving latency
- Complex aggregations (expensive to compute on-the-fly)
- Multiple models sharing features
- High serving traffic (protect source DB)

**Skip Feast if:**
- 100-500ms latency acceptable
- Simple queries (direct SQL from Supabase works fine)
- Low traffic
- Small scale (your current case: 86K orders, 371 agents)

## Alternative: Direct Supabase Queries

```python
# No Feast needed - query source directly
def get_recommendations(agent_id):
    result = supabase.rpc('get_agent_product_scores', {
        'agent_id': agent_id
    }).execute()
    return result.data

# Pros: Always fresh data, simpler infrastructure
# Cons: 100-500ms latency, load on production DB
```
