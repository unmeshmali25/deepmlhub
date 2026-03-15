# ML Pipeline Architecture: Feast + DVC + MLflow

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                    FEAST STACK                                      │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐              │
│  │   Online Store  │     │  Feature Registry│     │  Offline Store  │              │
│  │   (Redis)       │     │  (Metadata)      │     │  (Snowflake)    │              │
│  │                 │     │                  │     │                 │              │
│  │  Latest values  │     │  Feature defs    │     │  Historical     │              │
│  │  for serving    │     │  Entity types    │     │  feature tables │              │
│  └────────┬────────┘     └─────────────────┘     └────────┬────────┘              │
│           ↑                                               │                         │
│           │                                               │ get_historical_features()│
│           │                                               ↓                         │
│  ┌────────▼────────┐                          ┌─────────────────────┐              │
│  │  Materialization │                          │   Training Data      │              │
│  │     Jobs         │                          │   (DataFrame or      │              │
│  │  (hourly/daily)  │                          │    Parquet files)    │              │
│  └──────────────────┘                          └──────────┬──────────┘              │
│                                                           │                         │
│                                                           ↓                         │
└─────────────────────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            DVC PIPELINE (Horizontal)                                │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐  │
│  │  Stage  │ ───→ │  Stage  │ ───→ │  Stage  │ ───→ │  Stage  │ ───→ │  Stage  │  │
│  │   1     │      │   2     │      │   3     │      │   4     │      │   5     │  │
│  │         │      │         │      │         │      │         │      │         │  │
│  │Prepare  │      │ Feature │      │  Train  │      │ Evaluate│      │  Model  │  │
│  │ Entities│      │  Eng    │      │  Model  │      │         │      │  Save   │  │
│  └────┬────┘      └────┬────┘      └────┬────┘      └────┬────┘      └────┬────┘  │
│       │                │                │                │                │       │
│       ▼                ▼                ▼                ▼                ▼       │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │                         MLFLOW TRACKING                                      │  │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐│  │
│  │  │  • Log parameters (lr, batch_size)    • Log metrics (accuracy, loss)   ││  │
│  │  │  • Log artifacts (confusion matrix)   • Log model (pickled artifact)   ││  │
│  │  └─────────────────────────────────────────────────────────────────────────┘│  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│       │                │                │                │                │       │
│       ▼                ▼                ▼                ▼                ▼       │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │                      DVC REMOTE STORAGE (S3/GCS)                             │  │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐│  │
│  │  │                                                                         ││  │
│  │  │   files/md5/abc...   models/model.pkl   metrics/metrics.json            ││  │
│  │  │   files/md5/def...   data/features.parquet                             ││  │
│  │  │   files/md5/123...   config/params.yaml                                ││  │
│  │  │                                                                         ││  │
│  │  │   [Content-addressed by hash - same file = same location]              ││  │
│  │  │                                                                         ││  │
│  │  └─────────────────────────────────────────────────────────────────────────┘│  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Legend

- **═══** Horizontal flow: DVC pipeline stages
- **↑** Vertical flow: Feast data feeding from source
- **┌─┐** Boxes: Stages/stores/components
- **[MLFLOW]** Integrated tracking within training stage

## Phase Separation

| Phase | DVC | Feast | Purpose |
|-------|-----|-------|---------|
| **Training** | ✅ Yes | Offline Store | Track experiments, get historical features |
| **Evaluation** | ✅ Yes | Offline Store | Reproduce results, compare models |
| **Deployment** | ⚠️ Optional | N/A | Track which model version goes to prod |
| **Inference** | ❌ No | Online Store | Just serve - no tracking needed |

## Data Flow

1. **Feast Offline Store** provides historical features for training
2. **DVC Pipeline** orchestrates the training workflow
3. **MLflow** tracks metrics and parameters during training
4. **DVC Remote** stores versioned artifacts (models, data, configs)
5. **Feast Online Store** serves features for real-time inference

## Key Components

### Feast Stack
- **Online Store (Redis)**: Latest feature values for real-time serving
- **Feature Registry**: Metadata about features, entities, and data sources
- **Offline Store (Snowflake/BigQuery)**: Historical feature data for training
- **Materialization Jobs**: Sync offline data to online store periodically

### DVC Pipeline Stages
1. **Prepare Entities**: Create entity dataframe for Feast query
2. **Feature Engineering**: Transform and select features
3. **Train Model**: Fit model using prepared features
4. **Evaluate**: Validate model performance
5. **Model Save**: Persist trained model artifact

### MLflow Integration
Tracks throughout pipeline:
- Parameters (learning rate, batch size, etc.)
- Metrics (accuracy, loss, etc.)
- Artifacts (confusion matrices, model files)

### DVC Remote Storage
- S3, GCS, Azure, or local storage
- Content-addressed by MD5 hash
- Stores: models, datasets, configurations
- Enables reproducibility and collaboration
