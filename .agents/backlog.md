# Master Backlog

**Last Updated**: 2026-02-10  
**Total Phases**: 9  
**Status**: Phase 0-6 Complete, Phase 7 Optional

---

## Summary

| Phase | Name | Status | Sprint | Completion |
|-------|------|--------|--------|------------|
| 0 | Local ML Pipeline | ✅ Complete | sprint_00_local_ml | 2026-01-30 |
| 1 | DVC Pipeline | ✅ Complete | sprint_01_dvc_pipeline | 2026-02-01 |
| 2 | GCP & Terraform | ✅ Complete | sprint_02_terraform_infra | 2026-02-04 |
| 3 | GitHub Setup | ✅ Complete | sprint_03_github_setup | 2026-02-05 |
| 4 | DVC Remote | ✅ Complete | sprint_04_dvc_remote | 2026-02-05 |
| 5 | Terraform Apply | ✅ Complete | sprint_05_terraform_apply | 2026-02-05 |
| 6 | Docker Images | ✅ Complete | sprint_06_docker | 2026-02-10 |
| 7 | Kubernetes | ⬜ Optional | sprint_07_kubernetes | - |
| 8 | Verification | ⬜ Not Started | sprint_08_verification | - |

---

## Phase 0: Local ML Pipeline Setup ✅

**Status**: ✅ Complete  
**Sprint**: sprint_00_local_ml  
**Completion Date**: 2026-01-30

### [HUMAN] Prerequisites
- [x] [HUMAN] 0.1: Install Required Tools on MacBook
- [x] [HUMAN] 0.2: Install Docker Desktop
- [x] [HUMAN] 0.3: Create Project Virtual Environment
- [x] [HUMAN] 0.4: Install DVC and MLflow

### [AI] Implementation Tasks
- [x] [AI] 0.1: Create Project Directory Structure
- [x] [AI] 0.2: Create requirements.txt
- [x] [AI] 0.3: Create Configuration File
- [x] [AI] 0.4: Create Data Generation Script
- [x] [AI] 0.5: Create Data Preprocessing Script
- [x] [AI] 0.6: Create Model Training Script (with MLflow)
- [x] [AI] 0.7: Create Model Evaluation Script
- [x] [AI] 0.8: Create Prediction Script
- [x] [AI] 0.9: Create FastAPI Inference Server
- [x] [AI] 0.10: Create Unit Tests
- [x] [AI] 0.11: Create .gitignore for Project

---

## Phase 1: DVC Pipeline Setup ✅

**Status**: ✅ Complete  
**Sprint**: sprint_01_dvc_pipeline  
**Completion Date**: 2026-02-01

### [AI] Implementation Tasks
- [x] [AI] 1.1: Create DVC Pipeline File
- [x] [AI] 1.2: Create params.yaml for DVC
- [x] [AI] 1.3: Initialize DVC in Project

---

## Phase 2: GCP & Terraform Infrastructure 🔄

**Status**: 🔄 In Progress  
**Sprint**: sprint_02_terraform_infra  
**Started**: 2026-02-04

### [HUMAN] Prerequisites (ALL COMPLETE ✅)
- [x] [HUMAN] 2.0.1: Create Google Cloud Account
- [x] [HUMAN] 2.0.2: Install Google Cloud CLI (gcloud)
- [x] [HUMAN] 2.0.3: Create GCP Project
- [x] [HUMAN] 2.0.4: Link Billing Account
- [x] [HUMAN] 2.0.5: Enable Required GCP APIs
- [x] [HUMAN] 2.0.6: Set Up Billing Alerts
- [x] [HUMAN] 2.0.7: Install Terraform
- [x] [HUMAN] 2.0.8: Create Terraform State Bucket
- [x] [HUMAN] 2.0.9: Create Service Account for Terraform

### [AI] Implementation Tasks
- [x] [AI] 2.1: Create Terraform Directory Structure
- [x] [AI] 2.2: Create GCS Module
- [x] [AI] 2.3: Create Artifact Registry Module
- [x] [AI] 2.4: Create MLflow Cloud Run Module
- [⏭️] [AI] 2.5: Create GKE Module (Optional) - Deferred to Phase 7
- [x] [AI] 2.6: Create Dev Environment Config

---

## Phase 3: GitHub Setup ✅

**Status**: ✅ Complete  
**Sprint**: sprint_03_github_setup  
**Started**: 2026-02-04  
**Completed**: 2026-02-05

### [HUMAN] Prerequisites
- [x] [HUMAN] 3.1: Create GitHub Repository (If Not Exists)
- [x] [HUMAN] 3.2: Create GitHub Service Account for CI/CD
- [x] [HUMAN] 3.3: Add GitHub Repository Secrets

### [AI] Implementation Tasks
- [x] [AI] 3.1: Create CI Workflow
- [x] [AI] 3.2: Create Build and Push Workflow
- [x] [AI] 3.3: Create Training Trigger Workflow

---

## Phase 4: DVC Remote Setup ✅

**Status**: ✅ Complete  
**Sprint**: sprint_04_dvc_remote  
**Started**: 2026-02-05  
**Completed**: 2026-02-05

### [HUMAN] Prerequisites
- [x] [HUMAN] 4.1: Create GCS Bucket for DVC
- [x] [HUMAN] 4.2: Configure DVC Remote

### [AI] Implementation Tasks
- [x] [AI] 4.3: Initialize DVC in synth_tabular_classification
- [x] [AI] 4.4: Configure DVC remote for project-level setup
- [x] [AI] 4.5: Test project-level DVC pipeline

---

## Phase 5: Apply Terraform Infrastructure ✅

**Status**: ✅ Complete  
**Sprint**: sprint_05_terraform_apply  
**Started**: 2026-02-05  
**Completed**: 2026-02-05

### [HUMAN] Prerequisites
- [x] [HUMAN] 5.1: Initialize Terraform
- [x] [HUMAN] 5.2: Review Terraform Plan
- [x] [HUMAN] 5.3: Apply Terraform
- [x] [HUMAN] 5.4: Verify MLflow Deployment

---

## Phase 6: Docker Images ✅

**Status**: ✅ Complete  
**Sprint**: sprint_06_docker  
**Started**: 2026-02-05  
**Completed**: 2026-02-10  
**Time Taken**: ~1 hour

### Summary
Created production-ready Docker images with security best practices. Images auto-build via GitHub Actions on push to main.

### [AI] Implementation Tasks
- [x] [AI] 6.1: Create Training Dockerfile ✅
- [x] [AI] 6.2: Create Inference Dockerfile ✅
- [x] [AI] 6.3: Create .dockerignore ✅

### Deliverables
- `docker/training/Dockerfile` - Python 3.10-slim, DVC + MLflow support
- `docker/inference/Dockerfile` - FastAPI server, health checks, port 8000
- `.dockerignore` - Excludes 200+ MB of unnecessary files

### Key Features
- **Security**: Non-root user (mluser), minimal base images
- **Optimization**: Layer caching, multi-stage ready
- **CI/CD**: GitHub Actions auto-build and push to Artifact Registry

---

## Phase 7: Kubernetes Manifests (Optional) ⬜

**Status**: ⬜ Not Started  
**Sprint**: sprint_07_kubernetes

### [HUMAN] Prerequisites
- [ ] [HUMAN] 6.1: Apply GKE Terraform
- [ ] [HUMAN] 6.2: Get GKE Credentials
- [ ] [HUMAN] 6.3: Install kubectl (If Not Installed)

### [AI] Implementation Tasks
- [ ] [AI] 7.1: Create Namespace and ConfigMaps
- [ ] [AI] 7.2: Create Inference Deployment

---

## Phase 8: Manual Verifications ⬜

**Status**: ⬜ Not Started  
**Sprint**: sprint_08_verification

### [HUMAN] Prerequisites
- [ ] [HUMAN] 7.1: Test Full Pipeline Locally
- [ ] [HUMAN] 7.2: Test DVC Push to GCS
- [ ] [HUMAN] 7.3: Test MLflow Connection to Cloud Run
- [ ] [HUMAN] 7.4: Test Docker Build and Push

---

## Phase 9: Ongoing Human Tasks ⬜

**Status**: ⬜ Not Started  
**Sprint**: N/A - Ongoing

### [HUMAN] Maintenance Tasks
- [ ] [HUMAN] 8.1: Monitor Costs (weekly)
- [ ] [HUMAN] 8.2: Rotate Service Account Keys (quarterly)

---

## Recent Activity

### 2026-02-10 - Sprint 6 Complete
**Completed**: Docker Images sprint
- Created training and inference Dockerfiles
- Added .dockerignore for optimized builds
- GitHub Actions ready for auto-build
- All images follow security best practices

### 2026-02-05 - Sprint 5 Complete
**Completed**: Terraform infrastructure applied
- All GCP resources provisioned
- Artifact Registry ready for Docker images
- MLflow deployed to Cloud Run

---

## Status Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Complete |
| 🔄 | In Progress |
| ⬜ | Not Started |
| ⏭️ | Skipped |
| 🚫 | Blocked |

---

## Quick Links

### Active & Recent
- [Active Sprint](active_sprint.md)
- [Sprint 6: Docker (Archive)](sprints/archive/sprint_06_docker.md)
- [Sprint 7: Kubernetes](sprints/sprint_07_kubernetes/tasks.md)

### All Sprints
- [Sprint 00: Local ML](sprints/sprint_00_local_ml/tasks.md)
- [Sprint 01: DVC Pipeline](sprints/sprint_01_dvc_pipeline/tasks.md)
- [Sprint 02: Terraform Infrastructure](sprints/sprint_02_terraform_infra/tasks.md)
- [Sprint 03: GitHub Setup](sprints/sprint_03_github_setup/tasks.md)
- [Sprint 04: DVC Remote](sprints/sprint_04_dvc_remote/tasks.md)
- [Sprint 05: Terraform Apply](sprints/archive/sprint_05_terraform_apply.md)

### Architecture
- [Architecture Plan](plans/mlops_plan_uno.md)
- [Artifact Registry](https://console.cloud.google.com/artifacts/docker/deepmlhub-voiceoffers/us-central1/ml-images)
