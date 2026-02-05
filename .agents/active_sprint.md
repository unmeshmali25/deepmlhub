# Active Sprint: Docker Images

**Sprint**: sprint_06_docker  
**Last Updated**: 2026-02-05  
**Status**: ⬜ Not Started

---

## Sprint Goal

Create Docker images for ML training and inference components. Build optimized, secure containers ready for deployment to GCP Artifact Registry.

---

## What's Happening Now

### ⬜ Ready to Start

| Task | Assignee | Status | Priority |
|------|----------|--------|----------|
| [AI] 6.1: Create Training Dockerfile | AI | ⬜ Not Started | High |
| [AI] 6.2: Create Inference Dockerfile | AI | ⬜ Not Started | High |
| [AI] 6.3: Create .dockerignore | AI | ⬜ Not Started | Medium |

---

## Sprint Metrics

**Tasks**: 0/3 Complete (0%)  
**Blockers**: None  
**ETA**: ~2-3 hours

---

## Prerequisites

✅ **All Complete**:
- Sprint 5 (Terraform Infrastructure) - Complete
- Artifact Registry created: `us-central1-docker.pkg.dev/deepmlhub-voiceoffers/ml-images`
- Docker installed locally

---

## Sprint Scope

### 1. Training Dockerfile
Create a containerized training environment:
- Python 3.10 base image
- All ML dependencies
- Training script entrypoint
- MLflow integration
- DVC support

### 2. Inference Dockerfile  
Create a containerized inference API:
- FastAPI application
- Model serving endpoint
- Health checks
- Optimized for Cloud Run

### 3. Docker Configuration
- `.dockerignore` for efficient builds
- Multi-stage builds (if needed)
- Security best practices

---

## Definition of Done

- [ ] All Dockerfiles build successfully locally
- [ ] Images are optimized (small size, fast builds)
- [ ] Security best practices followed (non-root user, minimal base image)
- [ ] .dockerignore excludes unnecessary files
- [ ] Documentation on how to build and run containers

---

## Next Sprint

**Sprint 7**: Kubernetes Manifests (Optional)
- Deploy containers to GKE
- Create Kubernetes manifests
- Configure services and ingress

---

## Quick Links

- [Sprint Tasks](sprints/sprint_06_docker/tasks.md)
- [Master Backlog](backlog.md)
- [Previous Sprint: Terraform Apply](sprints/archive/sprint_05_terraform_apply.md)
- [Next Sprint: Kubernetes](sprints/sprint_07_kubernetes/tasks.md)
- [Architecture Plan](plans/mlops_plan_uno.md)
- [Artifact Registry](https://console.cloud.google.com/artifacts/docker/deepmlhub-voiceoffers/us-central1/ml-images)
