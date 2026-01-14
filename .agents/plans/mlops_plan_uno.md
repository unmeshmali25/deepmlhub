# MLOps Infrastructure Plan

## Overview

Complete MLOps infrastructure for a solo developer managing 10 ML projects on GCP, optimized for minimal cost (~$5-15/month baseline).

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              GCP PROJECT                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │  Cloud Run   │    │     GCS      │    │    GKE       │               │
│  │   MLflow     │    │   Bucket     │    │   Standard   │               │
│  │              │    │              │    │              │               │
│  │ - Tracking   │    │ - DVC Remote │    │ - Training   │               │
│  │ - Registry   │    │ - MLflow DB  │    │ - Inference  │               │
│  │ - Serving    │    │ - Artifacts  │    │ - Dask       │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│         │                   │                   │                        │
│         └───────────────────┼───────────────────┘                        │
│                             │                                            │
│  ┌──────────────┐    ┌──────────────┐                                   │
│  │  Artifact    │    │  Supabase    │                                   │
│  │  Registry    │    │  (External)  │                                   │
│  │              │    │              │                                   │
│  │ Docker       │    │ Source       │                                   │
│  │ Images       │    │ Data         │                                   │
│  └──────────────┘    └──────────────┘                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌──────────────┐        ┌──────────────┐        ┌──────────────┐
│    Local     │        │   Colab /    │        │   GitHub     │
│    Dev       │        │ Lambda Labs  │        │   Actions    │
│              │        │              │        │              │
│ - Docker     │        │ - Notebooks  │        │ - CI/CD      │
│ - DVC CLI    │        │ - Testing    │        │ - Build      │
│ - Testing    │        │ - Training   │        │ - Deploy     │
└──────────────┘        └──────────────┘        └──────────────┘
```

---

## Key Technology Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| MLflow Hosting | Cloud Run | Serverless, scales to zero, ~$0-5/month |
| MLflow Backend | SQLite on GCS | Sufficient for solo dev, saves $7-15/month vs Cloud SQL |
| MLflow Auth | IAM Protected | Requires GCP credentials to access |
| DVC Remote | GCS Bucket | Native GCP integration, ~$0.02/GB/month |
| Kubernetes | GKE Standard | Free tier control plane, Spot VM support |
| Data Processing | Dask | Simpler than Spark for GB-scale data |
| Distributed Training | PyTorch DDP | Native PyTorch, well-supported |
| IaC | Terraform | Reproducible, version controlled |
| CI/CD | GitHub Actions | Free for public repos, integrated |
| Inference | FastAPI | Async support, automatic OpenAPI docs |

---

## Directory Structure

```
infra-claude/
├── .github/
│   └── workflows/
│       ├── ci.yaml                      # Lint, test, type-check
│       ├── build-push.yaml              # Build and push Docker images
│       ├── deploy-infra.yaml            # Terraform apply
│       └── train.yaml                   # Trigger training jobs
│
├── infrastructure/
│   ├── terraform/
│   │   ├── environments/
│   │   │   ├── dev/
│   │   │   │   ├── main.tf
│   │   │   │   ├── variables.tf
│   │   │   │   ├── terraform.tfvars
│   │   │   │   └── backend.tf
│   │   │   └── prod/
│   │   │       └── (same structure)
│   │   ├── modules/
│   │   │   ├── mlflow/
│   │   │   │   ├── main.tf              # Cloud Run + GCS
│   │   │   │   ├── variables.tf
│   │   │   │   └── outputs.tf
│   │   │   ├── gcs/
│   │   │   │   ├── main.tf              # DVC remote + MLflow artifacts
│   │   │   │   └── variables.tf
│   │   │   ├── gke/
│   │   │   │   ├── main.tf              # GKE Standard cluster
│   │   │   │   └── variables.tf
│   │   │   └── networking/
│   │   │       └── main.tf              # VPC, firewall rules
│   │   └── shared/
│   │       └── state-bucket.tf          # Terraform state bucket
│   │
│   └── kubernetes/
│       ├── base/
│       │   ├── namespace.yaml
│       │   ├── configmaps/
│       │   │   ├── training-config.yaml
│       │   │   └── inference-config.yaml
│       │   ├── secrets/
│       │   │   └── mlflow-credentials.yaml
│       │   └── rbac/
│       │       └── training-sa.yaml
│       ├── training/
│       │   ├── pytorch-ddp-job.yaml     # Template for DDP training
│       │   └── dask-cluster.yaml        # Optional Dask on K8s
│       ├── inference/
│       │   ├── batch-job.yaml
│       │   └── api-deployment.yaml
│       └── kustomization.yaml
│
├── docker/
│   ├── training/
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── inference/
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── mlflow/
│       ├── Dockerfile
│       ├── entrypoint.sh
│       └── requirements.txt
│
├── shared/
│   ├── src/
│   │   └── mlops_common/
│   │       ├── __init__.py
│   │       ├── config.py                # Configuration utilities
│   │       ├── mlflow_utils.py          # MLflow connection helpers
│   │       ├── dvc_utils.py             # DVC pipeline helpers
│   │       ├── data_loader.py           # Supabase/GCS data loading
│   │       └── metrics.py               # Common evaluation metrics
│   ├── pyproject.toml                   # Shared package definition
│   └── tests/
│       └── test_mlops_common.py
│
├── projects/
│   ├── synth_tabular_classification/    # Week 1 test project
│   │   ├── src/
│   │   │   ├── __init__.py
│   │   │   ├── data/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── generate.py          # Synthetic data generation
│   │   │   │   ├── download.py          # Supabase -> Parquet
│   │   │   │   └── preprocess.py        # Feature engineering with Dask
│   │   │   ├── model/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── train.py             # Training script
│   │   │   │   ├── train_distributed.py # PyTorch DDP training
│   │   │   │   ├── evaluate.py          # Evaluation script
│   │   │   │   └── architecture.py      # Model definition
│   │   │   └── inference/
│   │   │       ├── __init__.py
│   │   │       ├── predict.py           # Prediction utilities
│   │   │       └── server.py            # FastAPI server
│   │   ├── data/
│   │   │   ├── raw/                     # DVC tracked
│   │   │   ├── processed/               # DVC tracked
│   │   │   └── .gitkeep
│   │   ├── models/
│   │   │   └── .gitkeep                 # DVC tracked models
│   │   ├── notebooks/
│   │   │   ├── 01_exploration.ipynb
│   │   │   ├── 02_training.ipynb
│   │   │   └── colab_setup.py           # Colab connection helper
│   │   ├── tests/
│   │   │   ├── __init__.py
│   │   │   ├── test_data.py
│   │   │   ├── test_model.py
│   │   │   └── test_inference.py
│   │   ├── configs/
│   │   │   └── experiment.yaml          # Experiment configurations
│   │   ├── metrics/                     # Git tracked (not DVC)
│   │   ├── plots/                       # Git tracked (not DVC)
│   │   ├── dvc.yaml                     # DVC pipeline definition
│   │   ├── dvc.lock                     # DVC lock file (auto-generated)
│   │   ├── params.yaml                  # Pipeline parameters
│   │   ├── requirements.txt             # Project dependencies
│   │   └── README.md
│   │
│   ├── pytorch_00_tensor_gpu_basics/    # Existing
│   ├── pytorch_01_neural_networks/      # Existing
│   ├── pytorch_02_training_optimization/ # Existing
│   └── pytorch_03_computer_vision/      # Existing
│
├── scripts/
│   ├── setup_local.sh                   # Local dev environment setup
│   ├── setup_gcp.sh                     # GCP project initialization
│   └── sync_dvc.sh                      # DVC sync helper
│
├── .dvc/
│   ├── config                           # Global DVC config (GCS remote)
│   └── .gitignore
│
├── .gitignore
├── pyproject.toml                       # Root project config
├── requirements-dev.txt                 # Development dependencies
└── README.md
```

---

## Component Details

### 1. MLflow on Cloud Run

**Architecture**: Cloud Run + SQLite on GCS (mounted via gcsfuse)

**Dockerfile** (`docker/mlflow/Dockerfile`):
```dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    curl gnupg lsb-release \
    && export GCSFUSE_REPO=gcsfuse-$(lsb_release -c -s) \
    && echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
    && apt-get update && apt-get install -y gcsfuse \
    && apt-get clean

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /mnt/mlflow-data
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["./entrypoint.sh"]
```

**Entrypoint** (`docker/mlflow/entrypoint.sh`):
```bash
#!/bin/bash
set -e
gcsfuse --implicit-dirs ${MLFLOW_BUCKET} /mnt/mlflow-data
exec mlflow server \
    --backend-store-uri sqlite:////mnt/mlflow-data/mlflow.db \
    --default-artifact-root gs://${MLFLOW_BUCKET}/artifacts \
    --host 0.0.0.0 \
    --port 8080
```

**Terraform** (`infrastructure/terraform/modules/mlflow/main.tf`):
```hcl
resource "google_storage_bucket" "mlflow" {
  name          = "${var.project_id}-mlflow"
  location      = var.region
  force_destroy = false
  uniform_bucket_level_access = true
}

resource "google_service_account" "mlflow" {
  account_id   = "mlflow-server"
  display_name = "MLflow Server Service Account"
}

resource "google_storage_bucket_iam_member" "mlflow_admin" {
  bucket = google_storage_bucket.mlflow.name
  role   = "roles/storage.admin"
  member = "serviceAccount:${google_service_account.mlflow.email}"
}

resource "google_cloud_run_v2_service" "mlflow" {
  name     = "mlflow-server"
  location = var.region

  template {
    service_account = google_service_account.mlflow.email

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/mlflow/mlflow-server:latest"

      ports {
        container_port = 8080
      }

      env {
        name  = "MLFLOW_BUCKET"
        value = google_storage_bucket.mlflow.name
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
      }
    }

    scaling {
      min_instance_count = 0  # Scale to zero
      max_instance_count = 2
    }
  }
}

# IAM-protected access (not public)
resource "google_cloud_run_v2_service_iam_member" "invoker" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.mlflow.name
  role     = "roles/run.invoker"
  member   = "user:${var.user_email}"  # Your Google account
}
```

---

### 2. DVC Configuration

**Global Config** (`.dvc/config`):
```ini
[core]
    remote = gcs
    autostage = true

[remote "gcs"]
    url = gs://deepmlhub-dvc-storage
```

**Per-Project Pipeline** (`projects/synth_tabular_classification/dvc.yaml`):
```yaml
stages:
  generate_data:
    cmd: python -m src.data.generate
    params:
      - data.n_samples
      - data.n_features
      - data.n_classes
      - data.random_seed
    outs:
      - data/raw/synthetic_data.parquet

  preprocess:
    cmd: python -m src.data.preprocess
    deps:
      - data/raw/synthetic_data.parquet
      - src/data/preprocess.py
    params:
      - preprocessing.test_size
      - preprocessing.scaling
    outs:
      - data/processed/train.parquet
      - data/processed/test.parquet
      - data/processed/feature_info.json

  train:
    cmd: python -m src.model.train
    deps:
      - data/processed/train.parquet
      - data/processed/feature_info.json
      - src/model/train.py
      - src/model/architecture.py
    params:
      - model.hidden_sizes
      - model.dropout
      - training.epochs
      - training.batch_size
      - training.learning_rate
    outs:
      - models/model.pt
    metrics:
      - metrics/train_metrics.json:
          cache: false
    plots:
      - plots/loss_curve.csv:
          x: epoch
          y: loss

  evaluate:
    cmd: python -m src.model.evaluate
    deps:
      - data/processed/test.parquet
      - models/model.pt
      - src/model/evaluate.py
    metrics:
      - metrics/eval_metrics.json:
          cache: false
    plots:
      - plots/confusion_matrix.png:
          cache: false
```

**Parameters** (`projects/synth_tabular_classification/params.yaml`):
```yaml
data:
  n_samples: 10000
  n_features: 20
  n_classes: 3
  random_seed: 42

preprocessing:
  test_size: 0.2
  scaling: standard

model:
  hidden_sizes: [64, 32]
  dropout: 0.2

training:
  epochs: 50
  batch_size: 128
  learning_rate: 0.001
  early_stopping_patience: 5
```

**What Gets Versioned**:

| Artifact Type | DVC Tracked | Git Tracked | Location |
|--------------|-------------|-------------|----------|
| Raw data | Yes | No | `data/raw/*.parquet` |
| Processed data | Yes | No | `data/processed/*.parquet` |
| Model weights | Yes | No | `models/*.pt` |
| Metrics | No | Yes | `metrics/*.json` |
| Plots | No | Yes | `plots/` |
| Parameters | No | Yes | `params.yaml` |
| Pipeline | No | Yes | `dvc.yaml` |

---

### 3. GKE Cluster

**Terraform** (`infrastructure/terraform/modules/gke/main.tf`):
```hcl
resource "google_container_cluster" "primary" {
  name     = "deepmlhub-cluster"
  location = var.zone  # Zonal for free tier

  remove_default_node_pool = true
  initial_node_count       = 1

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  cluster_autoscaling {
    enabled = true
    resource_limits {
      resource_type = "cpu"
      minimum       = 0
      maximum       = 32
    }
    resource_limits {
      resource_type = "memory"
      minimum       = 0
      maximum       = 128
    }
  }
}

# CPU node pool (Spot VMs)
resource "google_container_node_pool" "cpu_pool" {
  name       = "cpu-pool"
  cluster    = google_container_cluster.primary.name

  autoscaling {
    min_node_count = 0
    max_node_count = 3
  }

  node_config {
    preemptible  = true
    machine_type = "e2-standard-4"
    labels = { workload-type = "cpu" }
  }
}

# GPU node pool (Spot VMs, scale to zero)
resource "google_container_node_pool" "gpu_pool" {
  name       = "gpu-pool"
  cluster    = google_container_cluster.primary.name

  autoscaling {
    min_node_count = 0
    max_node_count = 2
  }

  node_config {
    preemptible  = true
    machine_type = "n1-standard-8"

    guest_accelerator {
      type  = "nvidia-tesla-t4"
      count = 1
    }

    labels = { workload-type = "gpu" }

    taint {
      key    = "nvidia.com/gpu"
      value  = "present"
      effect = "NO_SCHEDULE"
    }
  }
}
```

---

### 4. PyTorch DDP Training Job

**Kubernetes Manifest** (`infrastructure/kubernetes/training/pytorch-ddp-job.yaml`):
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: synth-tabular-training
  namespace: ml-training
spec:
  parallelism: 2
  completions: 2
  backoffLimit: 3
  template:
    spec:
      restartPolicy: OnFailure
      serviceAccountName: training-sa

      containers:
      - name: pytorch
        image: ${REGION}-docker.pkg.dev/${PROJECT_ID}/ml/training:latest

        command: ["torchrun"]
        args:
          - "--nproc_per_node=1"
          - "--nnodes=2"
          - "--node_rank=$(JOB_COMPLETION_INDEX)"
          - "--master_addr=$(MASTER_ADDR)"
          - "--master_port=29500"
          - "-m"
          - "src.model.train_distributed"

        env:
        - name: MLFLOW_TRACKING_URI
          valueFrom:
            configMapKeyRef:
              name: training-config
              key: mlflow_tracking_uri

        resources:
          requests:
            nvidia.com/gpu: "1"
          limits:
            nvidia.com/gpu: "1"

      tolerations:
      - key: "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"

      nodeSelector:
        workload-type: gpu
```

---

### 5. FastAPI Inference Server

**Server Code** (`projects/synth_tabular_classification/src/inference/server.py`):
```python
"""FastAPI inference server."""
import json
from pathlib import Path
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..model.architecture import TabularClassifier

app = FastAPI(title="Synthetic Tabular Classification API")

model = None
feature_info = None


class PredictionRequest(BaseModel):
    features: List[List[float]]


class PredictionResponse(BaseModel):
    predictions: List[int]
    probabilities: List[List[float]]


@app.on_event("startup")
async def load_model():
    global model, feature_info

    with open("/models/feature_info.json") as f:
        feature_info = json.load(f)

    model = TabularClassifier(
        n_features=feature_info["n_features"],
        n_classes=feature_info["n_classes"],
    )
    model.load_state_dict(torch.load("/models/model.pt", map_location="cpu"))
    model.eval()


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/ready")
async def ready():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = torch.tensor(request.features, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

    return PredictionResponse(
        predictions=preds.tolist(),
        probabilities=probs.tolist(),
    )
```

**Kubernetes Deployment** (`infrastructure/kubernetes/inference/api-deployment.yaml`):
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: synth-tabular-api
  namespace: ml-inference
spec:
  replicas: 1
  selector:
    matchLabels:
      app: synth-tabular-api
  template:
    spec:
      containers:
      - name: fastapi
        image: ${REGION}-docker.pkg.dev/${PROJECT_ID}/ml/inference:latest
        ports:
        - containerPort: 8000

        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"

        livenessProbe:
          httpGet:
            path: /health
            port: 8000

        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
---
apiVersion: v1
kind: Service
metadata:
  name: synth-tabular-api
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: synth-tabular-api
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: synth-tabular-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: synth-tabular-api
  minReplicas: 1
  maxReplicas: 5
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

### 6. GitHub Actions CI/CD

**CI Pipeline** (`.github/workflows/ci.yaml`):
```yaml
name: CI

on:
  push:
    branches: [main, infra-claude]
  pull_request:
    branches: [main]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install -r requirements-dev.txt
          pip install -e shared/

      - name: Lint with ruff
        run: ruff check .

      - name: Type check with mypy
        run: mypy shared/src projects/synth_tabular_classification/src

      - name: Run tests
        run: pytest --cov=shared --cov=projects -v
```

**Build & Push** (`.github/workflows/build-push.yaml`):
```yaml
name: Build and Push Docker Images

on:
  push:
    branches: [main]
    paths:
      - 'docker/**'
      - 'shared/**'
      - 'projects/**/src/**'

env:
  PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
  REGION: us-central1

jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        image: [training, inference, mlflow]
    steps:
      - uses: actions/checkout@v4

      - uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}

      - uses: google-github-actions/setup-gcloud@v2

      - name: Configure Docker
        run: gcloud auth configure-docker ${{ env.REGION }}-docker.pkg.dev

      - name: Build and push
        run: |
          docker build -t ${{ env.REGION }}-docker.pkg.dev/${{ env.PROJECT_ID }}/ml/${{ matrix.image }}:${{ github.sha }} \
            -t ${{ env.REGION }}-docker.pkg.dev/${{ env.PROJECT_ID }}/ml/${{ matrix.image }}:latest \
            -f docker/${{ matrix.image }}/Dockerfile .
          docker push ${{ env.REGION }}-docker.pkg.dev/${{ env.PROJECT_ID }}/ml/${{ matrix.image }} --all-tags
```

---

### 7. Colab/Lambda Labs Workflow

**Setup Helper** (`projects/synth_tabular_classification/notebooks/colab_setup.py`):
```python
"""Setup helper for Google Colab notebooks."""
import os


def setup_environment(mlflow_tracking_uri: str):
    """
    Configure Colab for MLflow and DVC.

    Usage:
        from google.colab import auth
        auth.authenticate_user()

        setup_environment("https://mlflow-server-xxxxx.run.app")
    """
    import mlflow

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri

    print(f"MLflow tracking URI: {mlflow_tracking_uri}")
    print("Run 'dvc pull data/' to download data")


def pull_data():
    """Pull data from DVC remote."""
    os.system("dvc pull data/")


def push_results():
    """Push results to DVC and Git."""
    os.system("dvc push")
    os.system("git add dvc.lock metrics/ plots/")
    os.system('git commit -m "Update from Colab"')
    os.system("git push")
```

**Workflow**:
1. Install: `pip install mlflow dvc[gs] google-cloud-storage`
2. Authenticate: `from google.colab import auth; auth.authenticate_user()`
3. Clone repo and cd into project
4. Pull data: `dvc pull data/`
5. Run training (logs to MLflow)
6. Push: `dvc push && git commit && git push`

---

## GCP Project Setup

```bash
# 1. Create new GCP project
gcloud projects create deepmlhub-YOUR_ID --name="DeepMLHub"

# 2. Set as default
gcloud config set project deepmlhub-YOUR_ID

# 3. Link billing account
gcloud billing projects link deepmlhub-YOUR_ID --billing-account=BILLING_ACCOUNT_ID

# 4. Enable required APIs
gcloud services enable \
  run.googleapis.com \
  storage.googleapis.com \
  container.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  iam.googleapis.com \
  secretmanager.googleapis.com

# 5. Create Terraform state bucket
gsutil mb -l us-central1 gs://deepmlhub-YOUR_ID-tfstate

# 6. Create service account for GitHub Actions
gcloud iam service-accounts create github-actions \
  --display-name="GitHub Actions"

gcloud projects add-iam-policy-binding deepmlhub-YOUR_ID \
  --member="serviceAccount:github-actions@deepmlhub-YOUR_ID.iam.gserviceaccount.com" \
  --role="roles/editor"
```

---

## Cost Estimation

| Service | Monthly Cost |
|---------|-------------|
| MLflow on Cloud Run | $0-5 (scales to zero) |
| GCS Storage (10GB) | ~$0.20 |
| GKE Control Plane | $0 (free tier) |
| GKE Nodes | $0 (scale to zero when idle) |
| GPU Training (10 hrs/month, Spot T4) | ~$5-10 |
| Artifact Registry | ~$0.10/GB |
| **Total Baseline** | **~$5-15/month** |

---

## Security Considerations

1. **MLflow**: IAM-protected (requires `roles/run.invoker`)
2. **GCS**: Service accounts with minimal permissions
3. **Supabase**: Credentials in GitHub Secrets / K8s Secrets
4. **GKE**: Workload Identity for pod-level IAM
5. **Secrets**: Never commit credentials to Git

---

## Git + DVC Workflow

```bash
# Development workflow
git checkout -b feature/new-experiment

# Run pipeline
cd projects/synth_tabular_classification
dvc repro

# Track changes
git add dvc.lock metrics/ plots/
git commit -m "Experiment: new hyperparameters"

# Push to remotes
dvc push
git push origin feature/new-experiment

# Create PR for review
```
