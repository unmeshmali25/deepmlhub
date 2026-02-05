# Master Backlog

**Last Updated**: 2026-02-05  
**Total Phases**: 9  
**Status**: Phase 0-3 Complete, Phase 4 Ready

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

## Phase 4: DVC Remote Setup ⬜

**Status**: ⬜ Not Started  
**Sprint**: sprint_04_dvc_remote

### [HUMAN] Prerequisites
- [ ] [HUMAN] 4.1: Create GCS Bucket for DVC
- [ ] [HUMAN] 4.2: Configure DVC Remote

---

## Phase 5: Apply Terraform Infrastructure ⬜

**Status**: ⬜ Not Started  
**Sprint**: sprint_05_terraform_apply

### [HUMAN] Prerequisites
- [ ] [HUMAN] 5.1: Initialize Terraform
- [ ] [HUMAN] 5.2: Review Terraform Plan
- [ ] [HUMAN] 5.3: Apply Terraform
- [ ] [HUMAN] 5.4: Verify MLflow Deployment

---

## Phase 6: Docker Images ⬜

**Status**: ⬜ Not Started  
**Sprint**: sprint_06_docker

### [AI] Implementation Tasks
- [ ] [AI] 6.1: Create Training Dockerfile
- [ ] [AI] 6.2: Create Inference Dockerfile
- [ ] [AI] 6.3: Create .dockerignore

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

- [Active Sprint](active_sprint.md)
- [Sprint 00: Local ML](sprints/sprint_00_local_ml/tasks.md)
- [Sprint 01: DVC Pipeline](sprints/sprint_01_dvc_pipeline/tasks.md)
- [Sprint 02: Terraform Infrastructure](sprints/sprint_02_terraform_infra/tasks.md)
- [Architecture Plan](plans/mlops_plan_uno.md)
