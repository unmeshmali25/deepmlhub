# MLOps Human Tasks

> **You (the human) must complete these tasks.** They require account signups, CLI installations, manual configurations, and decisions that cannot be automated.

---

## How This Works

1. **You** complete tasks in this file (signups, installs, configs)
2. **AI** completes tasks in `.agents/todos/mlops_todos_uno.md` (code, configs)
3. Tasks are numbered to show dependencies (e.g., H1.1 must finish before AI can do A1.1)

---

## Phase 0: Prerequisites (Before AI Can Start)

### H0.1 Install Required Tools on MacBook

**Task**: Install Python, pip, and virtual environment tools.

```bash
# Check Python version (need 3.10+)
python3 --version

# If not installed, install via Homebrew
brew install python@3.10

# Verify pip
pip3 --version
```

**Verification**: Run `python3 --version` and confirm 3.10 or higher.

- [ ] Done

---

### H0.2 Install Docker Desktop

**Task**: Install Docker Desktop for Mac.

1. Go to: https://www.docker.com/products/docker-desktop/
2. Download Docker Desktop for Mac (Apple Silicon or Intel)
3. Install and start Docker Desktop
4. Open Docker Desktop and complete setup wizard

**Verification**:
```bash
docker --version
docker run hello-world
```

- [ ] Done

---

### H0.3 Create Project Virtual Environment

**Task**: Create a Python virtual environment for the project.

```bash
cd /Users/unmeshmali/Downloads/Unmesh/deepmlhub

# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Verify
which python
# Should show: /Users/.../deepmlhub/.venv/bin/python
```

**Add to your shell profile** (~/.zshrc or ~/.bashrc):
```bash
# Auto-activate deepmlhub venv
alias deepmlhub="cd /Users/unmeshmali/Downloads/Unmesh/deepmlhub && source .venv/bin/activate"
```

- [ ] Done

---

### H0.4 Install DVC and MLflow

**Task**: Install DVC and MLflow in your virtual environment.

```bash
# Activate venv first
source .venv/bin/activate

# Install tools
pip install dvc[gs] mlflow

# Verify
dvc version
mlflow --version
```

- [ ] Done

---

## Phase 1: GCP Account Setup

### H1.1 Create Google Cloud Account

**Task**: Sign up for Google Cloud Platform.

1. Go to: https://cloud.google.com/
2. Click "Get started for free" or "Start free"
3. Sign in with your Google account
4. Enter billing information (you get $300 free credit for 90 days)
5. Accept terms and conditions

**Important**: You won't be charged during the free trial. Set up billing alerts later.

- [ ] Done

---

### H1.2 Install Google Cloud CLI (gcloud)

**Task**: Install the gcloud CLI on your Mac.

**Option A: Homebrew (Recommended)**
```bash
brew install --cask google-cloud-sdk
```

**Option B: Direct Download**
1. Go to: https://cloud.google.com/sdk/docs/install
2. Download the macOS package
3. Extract and run: `./google-cloud-sdk/install.sh`

**After installation**:
```bash
# Initialize gcloud
gcloud init

# This will:
# 1. Open browser for authentication
# 2. Ask you to select/create a project
# 3. Set default region (choose: us-central1)
```

**Verification**:
```bash
gcloud --version
gcloud auth list
# Should show your Google account as active
```

- [ ] Done

---

### H1.3 Create GCP Project

**Task**: Create a new GCP project for this MLOps setup.

```bash
# Create project (replace YOUR_UNIQUE_ID with something like 'deepmlhub-unmesh')
gcloud projects create deepmlhub-YOUR_UNIQUE_ID --name="DeepMLHub"

# Set as default project
gcloud config set project deepmlhub-YOUR_UNIQUE_ID

# Verify
gcloud config get-value project
```

**Write down your project ID**: `deepmlhub-__________________`

- [ ] Done

---

### H1.4 Link Billing Account

**Task**: Link your billing account to the new project.

```bash
# List your billing accounts
gcloud billing accounts list

# Link billing to project (replace BILLING_ACCOUNT_ID)
gcloud billing projects link deepmlhub-YOUR_UNIQUE_ID \
  --billing-account=BILLING_ACCOUNT_ID
```

**Alternative**: Do this in the Console
1. Go to: https://console.cloud.google.com/billing
2. Select your project
3. Link to billing account

- [ ] Done

---

### H1.5 Enable Required GCP APIs

**Task**: Enable the APIs needed for MLOps infrastructure.

```bash
gcloud services enable \
  run.googleapis.com \
  storage.googleapis.com \
  container.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  iam.googleapis.com \
  secretmanager.googleapis.com \
  cloudresourcemanager.googleapis.com
```

**Verification**:
```bash
gcloud services list --enabled
# Should show all the above services
```

- [ ] Done

---

### H1.6 Set Up Billing Alerts (Recommended)

**Task**: Set up budget alerts so you don't get surprise bills.

1. Go to: https://console.cloud.google.com/billing/budgets
2. Click "Create Budget"
3. Set budget amount: $20/month (or your preference)
4. Set alerts at: 50%, 90%, 100%
5. Add your email for notifications

- [ ] Done

---

## Phase 2: Terraform Setup

### H2.1 Install Terraform

**Task**: Install Terraform CLI.

```bash
# Using Homebrew
brew tap hashicorp/tap
brew install hashicorp/tap/terraform

# Verify
terraform --version
```

- [ ] Done

---

### H2.2 Create Terraform State Bucket

**Task**: Create a GCS bucket to store Terraform state.

```bash
# Create bucket (must be globally unique)
gsutil mb -l us-central1 gs://deepmlhub-YOUR_UNIQUE_ID-tfstate

# Enable versioning (protects state history)
gsutil versioning set on gs://deepmlhub-YOUR_UNIQUE_ID-tfstate

# Verify
gsutil ls
```

**Write down your state bucket**: `gs://deepmlhub-__________________-tfstate`

- [ ] Done

---

### H2.3 Create Service Account for Terraform

**Task**: Create a service account that Terraform will use.

```bash
# Create service account
gcloud iam service-accounts create terraform \
  --display-name="Terraform Service Account"

# Grant necessary roles
PROJECT_ID=$(gcloud config get-value project)

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:terraform@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/editor"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:terraform@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountAdmin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:terraform@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

# Download key (store securely!)
gcloud iam service-accounts keys create ~/.config/gcloud/terraform-key.json \
  --iam-account=terraform@${PROJECT_ID}.iam.gserviceaccount.com

# Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/terraform-key.json
```

**Add to your shell profile** (~/.zshrc):
```bash
export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/terraform-key.json
```

- [ ] Done

---

## Phase 3: GitHub Setup

### H3.1 Create GitHub Repository (If Not Exists)

**Task**: Ensure your repo is on GitHub.

If not already on GitHub:
```bash
cd /Users/unmeshmali/Downloads/Unmesh/deepmlhub
git remote add origin https://github.com/YOUR_USERNAME/deepmlhub.git
git push -u origin main
```

- [ ] Done (already exists: https://github.com/unmeshmali25/deepmlhub.git)

---

### H3.2 Create GitHub Service Account for CI/CD

**Task**: Create a GCP service account for GitHub Actions.

```bash
PROJECT_ID=$(gcloud config get-value project)

# Create service account
gcloud iam service-accounts create github-actions \
  --display-name="GitHub Actions"

# Grant roles
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/editor"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/container.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/run.admin"

# Download key
gcloud iam service-accounts keys create ~/github-actions-key.json \
  --iam-account=github-actions@${PROJECT_ID}.iam.gserviceaccount.com
```

- [ ] Done

---

### H3.3 Add GitHub Repository Secrets

**Task**: Add secrets to your GitHub repository.

1. Go to: https://github.com/unmeshmali25/deepmlhub/settings/secrets/actions
2. Click "New repository secret"
3. Add these secrets:

| Secret Name | Value |
|-------------|-------|
| `GCP_PROJECT_ID` | Your project ID (e.g., `deepmlhub-unmesh`) |
| `GCP_SA_KEY` | Contents of `~/github-actions-key.json` (entire JSON) |
| `GCP_REGION` | `us-central1` |

**To get the JSON contents**:
```bash
cat ~/github-actions-key.json
# Copy the entire output
```

- [ ] Done

---

## Phase 4: DVC Remote Setup

### H4.1 Create GCS Bucket for DVC

**Task**: Create a bucket for DVC data storage.

```bash
PROJECT_ID=$(gcloud config get-value project)

# Create bucket
gsutil mb -l us-central1 gs://${PROJECT_ID}-dvc-storage

# Verify
gsutil ls
```

**Write down your DVC bucket**: `gs://deepmlhub-__________________-dvc-storage`

- [ ] Done

---

### H4.2 Configure DVC Remote

**Task**: Configure DVC to use the GCS bucket.

```bash
cd /Users/unmeshmali/Downloads/Unmesh/deepmlhub

# Initialize DVC (if not done)
dvc init

# Add GCS remote
dvc remote add -d gcs gs://YOUR_PROJECT_ID-dvc-storage

# Configure GCS credentials
dvc remote modify gcs credentialpath ~/.config/gcloud/terraform-key.json

# Verify
dvc remote list
cat .dvc/config
```

- [ ] Done

---

## Phase 5: Apply Terraform Infrastructure

### H5.1 Initialize Terraform

**Task**: Initialize Terraform with the backend.

```bash
cd infrastructure/terraform/environments/dev

# Initialize (downloads providers, configures backend)
terraform init
```

If you see errors about the backend bucket, ensure H2.2 is complete.

- [ ] Done

---

### H5.2 Review Terraform Plan

**Task**: Review what Terraform will create.

```bash
cd infrastructure/terraform/environments/dev

# See what will be created
terraform plan

# Review the output carefully:
# - GCS buckets
# - Service accounts
# - Cloud Run service (MLflow)
# - GKE cluster (if included)
# - Artifact Registry
```

**Important**: Review the plan before applying. Ask questions if unsure.

- [ ] Done (reviewed and approved)

---

### H5.3 Apply Terraform

**Task**: Create the infrastructure.

```bash
cd infrastructure/terraform/environments/dev

# Apply (type 'yes' when prompted)
terraform apply
```

This will create:
- GCS bucket for MLflow
- Cloud Run service for MLflow
- Artifact Registry for Docker images
- Service accounts with proper IAM

**Save the outputs**:
```bash
terraform output
# Write down the MLflow URL and other outputs
```

**MLflow URL**: `https://mlflow-server-____________________.run.app`

- [ ] Done

---

### H5.4 Verify MLflow Deployment

**Task**: Verify MLflow is running on Cloud Run.

```bash
# Get the MLflow URL
MLFLOW_URL=$(terraform output -raw mlflow_url)

# Test health (may need authentication)
curl $MLFLOW_URL

# Or authenticate first
gcloud auth print-identity-token | xargs -I {} curl -H "Authorization: Bearer {}" $MLFLOW_URL
```

**Alternative**: Check in Cloud Console
1. Go to: https://console.cloud.google.com/run
2. Find `mlflow-server` service
3. Click the URL to open MLflow UI

- [ ] Done

---

## Phase 6: GKE Cluster Setup (Optional - For Distributed Training)

### H6.1 Apply GKE Terraform

**Task**: Create the GKE cluster (only when ready for K8s).

```bash
cd infrastructure/terraform/environments/dev

# Apply with GKE module enabled
terraform apply -target=module.gke
```

**Warning**: GKE clusters cost money even when idle (~$70/month for control plane on Autopilot, free on Standard). The Terraform is configured for Standard with scale-to-zero nodes.

- [ ] Done (or skipped for now)

---

### H6.2 Get GKE Credentials

**Task**: Configure kubectl to connect to your cluster.

```bash
PROJECT_ID=$(gcloud config get-value project)

gcloud container clusters get-credentials deepmlhub-cluster \
  --zone us-central1-a \
  --project $PROJECT_ID

# Verify
kubectl get nodes
kubectl get namespaces
```

- [ ] Done

---

### H6.3 Install kubectl (If Not Installed)

**Task**: Install kubectl CLI.

```bash
# Using Homebrew
brew install kubectl

# Or via gcloud
gcloud components install kubectl

# Verify
kubectl version --client
```

- [ ] Done

---

## Phase 7: Manual Verifications

### H7.1 Test Full Pipeline Locally

**Task**: Run the ML pipeline locally and verify it works.

```bash
cd projects/synth_tabular_classification

# Activate venv
source ../../.venv/bin/activate

# Run pipeline
dvc repro

# Check outputs
ls data/raw/
ls data/processed/
ls models/
cat metrics/metrics.json

# Start MLflow UI locally
mlflow ui --backend-store-uri file://$(pwd)/mlruns

# Open http://localhost:5000 and verify experiments
```

- [ ] Done

---

### H7.2 Test DVC Push to GCS

**Task**: Push data to GCS and verify.

```bash
cd projects/synth_tabular_classification

# Push to remote
dvc push

# Verify in GCS
gsutil ls gs://YOUR_PROJECT_ID-dvc-storage/
```

- [ ] Done

---

### H7.3 Test MLflow Connection to Cloud Run

**Task**: Verify local training can log to Cloud Run MLflow.

```bash
cd projects/synth_tabular_classification

# Set environment variable to Cloud Run MLflow
export MLFLOW_TRACKING_URI=https://mlflow-server-XXXX.run.app

# Authenticate
gcloud auth print-identity-token > /tmp/token
export MLFLOW_TRACKING_TOKEN=$(cat /tmp/token)

# Run training (should log to Cloud Run MLflow)
python -m src.model.train

# Check Cloud Run MLflow UI for new run
```

- [ ] Done

---

### H7.4 Test Docker Build and Push

**Task**: Build and push Docker image to Artifact Registry.

```bash
PROJECT_ID=$(gcloud config get-value project)
REGION=us-central1

# Configure Docker for Artifact Registry
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Build image
cd projects/synth_tabular_classification
docker build -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/ml/inference:test .

# Push to Artifact Registry
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/ml/inference:test

# Verify in Console
# Go to: https://console.cloud.google.com/artifacts
```

- [ ] Done

---

## Phase 8: Ongoing Human Tasks

### H8.1 Monitor Costs

**Task**: Check GCP costs weekly.

1. Go to: https://console.cloud.google.com/billing
2. Review cost breakdown
3. Shut down unused resources

**Cost control tips**:
- Scale GKE nodes to zero when not training
- Delete old Docker images from Artifact Registry
- Use Spot VMs for training

- [ ] Set up weekly reminder

---

### H8.2 Rotate Service Account Keys

**Task**: Rotate keys every 90 days for security.

```bash
PROJECT_ID=$(gcloud config get-value project)

# List existing keys
gcloud iam service-accounts keys list \
  --iam-account=github-actions@${PROJECT_ID}.iam.gserviceaccount.com

# Create new key
gcloud iam service-accounts keys create ~/github-actions-key-new.json \
  --iam-account=github-actions@${PROJECT_ID}.iam.gserviceaccount.com

# Update GitHub secret with new key

# Delete old key (after updating GitHub)
gcloud iam service-accounts keys delete OLD_KEY_ID \
  --iam-account=github-actions@${PROJECT_ID}.iam.gserviceaccount.com
```

- [ ] Set up quarterly reminder

---

## Quick Reference

### Your Project Values (Fill In)

| Item | Value |
|------|-------|
| GCP Project ID | `deepmlhub-________________` |
| Terraform State Bucket | `gs://deepmlhub-________________-tfstate` |
| DVC Storage Bucket | `gs://deepmlhub-________________-dvc-storage` |
| MLflow URL | `https://mlflow-server-________________.run.app` |
| GKE Cluster | `deepmlhub-cluster` |
| Region | `us-central1` |

### Important File Locations

| File | Purpose |
|------|---------|
| `~/.config/gcloud/terraform-key.json` | Terraform service account key |
| `~/github-actions-key.json` | GitHub Actions service account key |
| `.dvc/config` | DVC remote configuration |
| `infrastructure/terraform/environments/dev/terraform.tfvars` | Terraform variables |

---

## Status Tracker

| Phase | Task | Status |
|-------|------|--------|
| 0 | H0.1 Install Python | ⬜ |
| 0 | H0.2 Install Docker | ⬜ |
| 0 | H0.3 Create venv | ⬜ |
| 0 | H0.4 Install DVC/MLflow | ⬜ |
| 1 | H1.1 GCP Account | ⬜ |
| 1 | H1.2 Install gcloud | ⬜ |
| 1 | H1.3 Create Project | ⬜ |
| 1 | H1.4 Link Billing | ⬜ |
| 1 | H1.5 Enable APIs | ⬜ |
| 1 | H1.6 Billing Alerts | ⬜ |
| 2 | H2.1 Install Terraform | ⬜ |
| 2 | H2.2 State Bucket | ⬜ |
| 2 | H2.3 Terraform SA | ⬜ |
| 3 | H3.1 GitHub Repo | ⬜ |
| 3 | H3.2 GitHub SA | ⬜ |
| 3 | H3.3 GitHub Secrets | ⬜ |
| 4 | H4.1 DVC Bucket | ⬜ |
| 4 | H4.2 Configure DVC | ⬜ |
| 5 | H5.1 Terraform Init | ⬜ |
| 5 | H5.2 Terraform Plan | ⬜ |
| 5 | H5.3 Terraform Apply | ⬜ |
| 5 | H5.4 Verify MLflow | ⬜ |
| 6 | H6.1 GKE Terraform | ⬜ |
| 6 | H6.2 GKE Credentials | ⬜ |
| 6 | H6.3 Install kubectl | ⬜ |
| 7 | H7.1 Test Pipeline | ⬜ |
| 7 | H7.2 Test DVC Push | ⬜ |
| 7 | H7.3 Test MLflow Cloud | ⬜ |
| 7 | H7.4 Test Docker Push | ⬜ |

Legend: ⬜ Not Started | 🔄 In Progress | ✅ Complete | ⏭️ Skipped
