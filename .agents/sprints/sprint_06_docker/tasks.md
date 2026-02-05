# Sprint 6: Docker Images

**Status**: 🔄 Ready to Start  
**Sprint Goal**: Create Dockerfiles for training and inference  
**Started**: 2026-02-05

---

## Summary

Create Docker images for the ML pipeline. Training image for running the full pipeline, inference image for serving predictions via API.

**Current Status**: 0/3 tasks complete  
**Prerequisites**: Sprint 5 (Terraform infrastructure applied) ✅ Complete

---

## Prerequisites

✅ **Sprint 5 Complete**:
- Artifact Registry created: `us-central1-docker.pkg.dev/deepmlhub-voiceoffers/ml-images`
- Terraform infrastructure applied
- All GCP resources ready

---

## Tasks

### [AI] 6.1: Create Training Dockerfile ⬜

**Status**: ⬜ Not Started  
**Priority**: High  
**Assigned To**: AI

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
- MLflow integration configured
- DVC support for data

---

### [AI] 6.2: Create Inference Dockerfile ⬜

**Status**: ⬜ Not Started  
**Priority**: High  
**Assigned To**: AI

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
- Optimized for Cloud Run

---

### [AI] 6.3: Create .dockerignore ⬜

**Status**: ⬜ Not Started  
**Priority**: Medium  
**Assigned To**: AI

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
- Test files and cache

---

## Verification

```bash
PROJECT_ID=deepmlhub-voiceoffers
REGION=us-central1

# Configure Docker for Artifact Registry
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Build training image
docker build -f docker/training/Dockerfile \
  -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/ml-images/training:latest \
  .

# Build inference image
docker build -f docker/inference/Dockerfile \
  -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/ml-images/inference:latest \
  .

# Test inference locally
docker run -p 8000:8000 ${REGION}-docker.pkg.dev/${PROJECT_ID}/ml-images/inference:latest

# Push to Artifact Registry
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/ml-images/training:latest
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/ml-images/inference:latest
```

---

## Next Sprint

**Sprint 7**: Kubernetes Manifests (Optional)
- [HUMAN] 7.1-7.3: GKE setup prerequisites
- [AI] 7.1: Create Namespace and ConfigMaps
- [AI] 7.2: Create Inference Deployment

---

## Quick Links

- [Active Sprint](../active_sprint.md)
- [Master Backlog](../backlog.md)
- [Previous Sprint: Terraform Apply](../archive/sprint_05_terraform_apply.md)
- [Next Sprint: Kubernetes](../sprint_07_kubernetes/tasks.md)
- [Architecture Plan](../plans/mlops_plan_uno.md)
- [Artifact Registry](https://console.cloud.google.com/artifacts/docker/deepmlhub-voiceoffers/us-central1/ml-images)
