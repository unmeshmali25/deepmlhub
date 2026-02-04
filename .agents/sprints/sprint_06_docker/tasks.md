# Sprint 6: Docker Images

**Status**: ⬜ Not Started  
**Sprint Goal**: Create Dockerfiles for training and inference

---

## Summary

Create Docker images for the ML pipeline. Training image for running the full pipeline, inference image for serving predictions via API.

**Current Status**: 0/3 tasks complete  
**Prerequisites**: Sprint 5 (Terraform infrastructure applied)

---

## Prerequisites

⚠️ **BLOCKING**: Sprint 5 must be complete before starting these tasks.

---

## Tasks

### [AI] 6.1: Create Training Dockerfile 🚫

**Status**: 🚫 Blocked (waiting on Sprint 5)  
**Priority**: High

**Definition of Done**:
- [ ] Dockerfile created for training
- [ ] Base image is python:3.10-slim
- [ ] All dependencies installed
- [ ] Source code copied
- [ ] Default command runs training

**File**: `docker/training/Dockerfile`

**Requirements**:
- Python 3.10
- Install requirements.txt
- Copy src/ and configs/
- Set PYTHONPATH
- Default CMD runs training

---

### [AI] 6.2: Create Inference Dockerfile 🚫

**Status**: 🚫 Blocked  
**Priority**: High

**Definition of Done**:
- [ ] Dockerfile created for inference
- [ ] Base image is python:3.10-slim
- [ ] Model files copied
- [ ] Health check configured
- [ ] Exposes port 8000
- [ ] Runs uvicorn server

**File**: `docker/inference/Dockerfile`

**Requirements**:
- Python 3.10
- Install requirements.txt
- Copy src/, configs/, and models/
- Health check endpoint
- Expose port 8000
- Run uvicorn

---

### [AI] 6.3: Create .dockerignore 🚫

**Status**: 🚫 Blocked  
**Priority**: Medium

**Definition of Done**:
- [ ] .dockerignore created
- [ ] Excludes unnecessary files
- [ ] Keeps build small

**File**: `.dockerignore`

**Should Exclude**:
- .git/
- __pycache__/
- .venv/
- data/ (use DVC to pull)
- mlruns/
- .dvc/cache/
- Terraform state files
- Secrets (*.key, *.pem)
- Agent files (.agents/)

---

## Verification

After Sprint 5 is complete:

```bash
PROJECT_ID=$(gcloud config get-value project)
REGION=us-central1

# Configure Docker for Artifact Registry
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Build training image
docker build -f docker/training/Dockerfile \
  -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/ml/training:latest \
  .

# Build inference image
docker build -f docker/inference/Dockerfile \
  -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/ml/inference:latest \
  .

# Test locally
docker run -p 8000:8000 ${REGION}-docker.pkg.dev/${PROJECT_ID}/ml/inference:latest

# Push to Artifact Registry (after Sprint 3.3 is complete)
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/ml/training:latest
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/ml/inference:latest
```

---

## Next Sprint

**Sprint 7**: Kubernetes Manifests (Optional)
- [HUMAN] 6.1-6.3: GKE setup
- [AI] 7.1-7.2: K8s manifests
