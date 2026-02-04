# Sprint 3: GitHub Setup

**Status**: ⬜ Not Started  
**Sprint Goal**: Configure GitHub repository with Actions workflows for CI/CD

---

## Summary

Set up GitHub repository and create GitHub Actions workflows for continuous integration and deployment. This includes CI workflow for linting and testing, build workflow for Docker images, and training trigger workflow.

**Current Status**: 0/6 tasks complete

---

## Prerequisites

### [HUMAN] 3.1: Create GitHub Repository (If Not Exists) ⬜

**Status**: ⬜ Not Started  
**Assigned To**: Human

**Definition of Done**:
- [ ] Repository exists on GitHub
- [ ] Local repo connected to remote
- [ ] Can push/pull without errors

**Steps** (if not already done):
```bash
cd /Users/unmeshmali/Downloads/Unmesh/deepmlhub
git remote add origin https://github.com/YOUR_USERNAME/deepmlhub.git
git branch -M main
git push -u origin main
```

**Note**: Already exists at https://github.com/unmeshmali25/deepmlhub.git

**Blocking**: AI 3.1-3.3

---

### [HUMAN] 3.2: Create GitHub Service Account for CI/CD ⬜

**Status**: ⬜ Not Started  
**Assigned To**: Human

**Definition of Done**:
- [ ] Service account created
- [ ] IAM roles granted
- [ ] JSON key downloaded

**Roles**:
- roles/editor
- roles/iam.serviceAccountUser
- roles/storage.admin
- roles/artifactregistry.admin
- roles/container.admin
- roles/run.admin

**Blocking**: AI 3.1-3.3

---

### [HUMAN] 3.3: Add GitHub Repository Secrets ⬜

**Status**: ⬜ Not Started  
**Assigned To**: Human

**Definition of Done**:
- [ ] `GCP_PROJECT_ID` secret added
- [ ] `GCP_SA_KEY` secret added (full JSON)
- [ ] `GCP_REGION` secret added

**URL**: https://github.com/unmeshmali25/deepmlhub/settings/secrets/actions

**Blocking**: AI 3.1-3.3

---

## AI Tasks

### [AI] 3.1: Create CI Workflow 🚫

**Status**: 🚫 Blocked (waiting on Human 3.1-3.3)  
**Priority**: High

**Definition of Done**:
- [ ] `.github/workflows/ci.yaml` created
- [ ] Runs on push/PR to main
- [ ] Lints with ruff
- [ ] Type checks with mypy
- [ ] Runs all tests with pytest
- [ ] Checks DVC pipeline validity

**File**: `.github/workflows/ci.yaml`

---

### [AI] 3.2: Create Build and Push Workflow 🚫

**Status**: 🚫 Blocked  
**Priority**: Medium

**Definition of Done**:
- [ ] `.github/workflows/build-push.yaml` created
- [ ] Builds Docker images on relevant changes
- [ ] Pushes to Artifact Registry
- [ ] Uses GCP service account key from secrets

**File**: `.github/workflows/build-push.yaml`

---

### [AI] 3.3: Create Training Trigger Workflow 🚫

**Status**: 🚫 Blocked  
**Priority**: Medium

**Definition of Done**:
- [ ] `.github/workflows/train.yaml` created
- [ ] Manual trigger with workflow_dispatch
- [ ] Configurable experiment name and parameters
- [ ] Runs DVC pipeline
- [ ] Logs to Cloud Run MLflow
- [ ] Pushes results to DVC remote
- [ ] Commits changes back to repo

**File**: `.github/workflows/train.yaml`

---

## Next Sprint

**Sprint 4**: DVC Remote Setup
- [HUMAN] 4.1: Create GCS Bucket for DVC
- [HUMAN] 4.2: Configure DVC Remote
