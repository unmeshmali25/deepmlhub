# Sprint 3: GitHub Setup

**Status**: ✅ Complete  
**Started**: 2026-02-04  
**Completed**: 2026-02-05  
**Sprint Goal**: Configure GitHub repository with Actions workflows for CI/CD

---

## Summary

Set up GitHub repository and create GitHub Actions workflows for continuous integration and deployment. This includes CI workflow for linting and testing, build workflow for Docker images, and training trigger workflow.

**Current Status**: 6/6 tasks complete (100%)  
**Completed**: All tasks ✅  
**Time Taken**: ~30 minutes

---

## Prerequisites

### [HUMAN] 3.1: Create GitHub Repository ✅

**Status**: ✅ Complete (2026-02-04)  
**Assigned To**: Human

**Definition of Done**:
- [x] Repository exists on GitHub
- [x] Local repo connected to remote
- [x] Can push/pull without errors

**URL**: https://github.com/unmeshmali25/deepmlhub

**Verification**: Repository exists and is accessible

---

### [HUMAN] 3.2: Create GitHub Service Account for CI/CD ✅

**Status**: ✅ Complete (2026-02-05)  
**Assigned To**: Human  
**Time Taken**: ~15 minutes

**Definition of Done**:
- [x] Service account created: `github-actions@deepmlhub-voiceoffers.iam.gserviceaccount.com`
- [x] IAM roles granted:
  - roles/artifactregistry.writer
  - roles/storage.objectAdmin
  - roles/cloudbuild.builds.editor
  - roles/run.developer
- [x] JSON key downloaded to `~/github-actions-key.json`

**Verification**:
```bash
gcloud iam service-accounts list --project=deepmlhub-voiceoffers
# Shows: github-actions@deepmlhub-voiceoffers.iam.gserviceaccount.com
```

**Blocking**: AI 3.1-3.3 (now unblocked)

---

### [HUMAN] 3.3: Add GitHub Repository Secrets ✅

**Status**: ✅ Complete (2026-02-05)  
**Assigned To**: Human  
**Time Taken**: ~5 minutes

**Definition of Done**:
- [x] `GCP_PROJECT_ID` secret added: `deepmlhub-voiceoffers`
- [x] `GCP_SA_KEY` secret added (full JSON key content)
- [x] `GCP_REGION` secret added: `us-central1`
- [x] `ARTIFACT_REGISTRY` secret added: `us-central1-docker.pkg.dev/deepmlhub-voiceoffers/ml-images`
- [x] `MLFLOW_TRACKING_URI` secret added: Cloud Run MLflow URL

**URL**: https://github.com/unmeshmali25/deepmlhub/settings/secrets/actions

**Verification**: All 5 secrets visible in GitHub UI

**Blocking**: AI 3.1-3.3 (now unblocked)

---

## AI Tasks

### [AI] 3.1: Create CI Workflow ✅

**Status**: ✅ Complete (2026-02-05)  
**Priority**: High  
**Time Taken**: ~10 minutes

**Definition of Done**:
- [x] `.github/workflows/ci.yaml` created
- [x] Runs on push/PR to main
- [x] Lints with ruff
- [x] Type checks with mypy
- [x] Runs all tests with pytest
- [x] Checks DVC pipeline validity

**File**: `.github/workflows/ci.yaml`

**Features**:
- Runs on Python 3.10
- Installs dependencies from project requirements.txt
- Performs ruff linting and format checking
- Type checks with mypy (ignoring missing imports)
- Runs pytest with coverage reporting
- Validates DVC pipeline with `dvc dag`

---

### [AI] 3.2: Create Build and Push Workflow ✅

**Status**: ✅ Complete (2026-02-05)  
**Priority**: Medium  
**Time Taken**: ~10 minutes

**Definition of Done**:
- [x] `.github/workflows/build-push.yaml` created
- [x] Builds Docker images on relevant changes
- [x] Pushes to Artifact Registry
- [x] Uses GCP service account key from secrets

**File**: `.github/workflows/build-push.yaml`

**Features**:
- Triggers on push to main when docker/** or project files change
- Authenticates to GCP using service account JSON key
- Configures Docker for Artifact Registry
- Builds training, inference, and MLflow images (if Dockerfiles exist)
- Tags images with both SHA and 'latest'
- Gracefully skips missing Dockerfiles

---

### [AI] 3.3: Create Training Trigger Workflow ✅

**Status**: ✅ Complete (2026-02-05)  
**Priority**: Medium  
**Time Taken**: ~10 minutes

**Definition of Done**:
- [x] `.github/workflows/train.yaml` created
- [x] Manual trigger with workflow_dispatch
- [x] Configurable experiment name and parameters
- [x] Runs DVC pipeline
- [x] Logs to Cloud Run MLflow
- [x] Pushes results to DVC remote
- [x] Commits changes back to repo

**File**: `.github/workflows/train.yaml`

**Features**:
- Manual trigger with configurable inputs:
  - Project name (default: synth_tabular_classification)
  - MLflow experiment name
  - DVC pull/push toggles
- Authenticates to GCP for DVC remote access
- Pulls data from DVC remote (optional)
- Runs full DVC pipeline with MLflow tracking
- Pushes results to DVC remote (optional)
- Commits dvc.lock, metrics, and models back to Git

---

## Next Sprint

**Sprint 4**: DVC Remote Setup
- [HUMAN] 4.1: Create GCS Bucket for DVC
- [HUMAN] 4.2: Configure DVC Remote
