# Sprint 2: GCP & Terraform Infrastructure

**Status**: 🔄 In Progress  
**Started**: 2026-02-04  
**Sprint Goal**: Create reusable Terraform infrastructure modules for GCP resources (GCS, MLflow on Cloud Run, Artifact Registry)

---

## Summary

Set up GCP infrastructure using Terraform. All human prerequisites are complete. AI now proceeding with Terraform module creation.

**Current Status**: 0/6 AI tasks complete  
**Blockers**: None (all prerequisites complete)

---

## Prerequisites Status

✅ **All [HUMAN] Prerequisites Complete** (2026-02-04)

| Task | Status | Completed |
|------|--------|-----------|
| [HUMAN] 2.0.1: Create Google Cloud Account | ✅ | 2026-02-04 |
| [HUMAN] 2.0.2: Install Google Cloud CLI (gcloud) | ✅ | 2026-02-04 |
| [HUMAN] 2.0.3: Create GCP Project | ✅ | 2026-02-04 |
| [HUMAN] 2.0.4: Link Billing Account | ✅ | 2026-02-04 |
| [HUMAN] 2.0.5: Enable Required GCP APIs | ✅ | 2026-02-04 |
| [HUMAN] 2.0.6: Set Up Billing Alerts | ✅ | 2026-02-04 |
| [HUMAN] 2.0.7: Install Terraform | ✅ | 2026-02-04 |
| [HUMAN] 2.0.8: Create Terraform State Bucket | ✅ | 2026-02-04 |
| [HUMAN] 2.0.9: Create Service Account for Terraform | ✅ | 2026-02-04 |

**All AI tasks are now unblocked.**

---

## Tasks

### [AI] 2.1: Create Terraform Directory Structure ✅

**Status**: ✅ Complete (2026-02-04)  
**Estimated Time**: 10 minutes  
**Actual Time**: 2 minutes  
**Priority**: High

**Definition of Done**:
- [x] Directory structure created per plan
- [x] All directories have .gitkeep if empty
- [x] Structure matches plan specification

**Created Directories**:
```
infrastructure/terraform/
├── environments/
│   └── dev/
├── modules/
│   ├── artifact-registry/
│   ├── gcs/
│   ├── gke/
│   └── mlflow/
```

**Directories to Create**:
```bash
infrastructure/terraform/environments/dev
infrastructure/terraform/modules/gcs
infrastructure/terraform/modules/mlflow
infrastructure/terraform/modules/artifact-registry
infrastructure/terraform/modules/gke
```

**Blocking**: AI 2.2, AI 2.3, AI 2.4, AI 2.6

---

### [AI] 2.2: Create GCS Module ✅

**Status**: ✅ Complete (2026-02-04)  
**Estimated Time**: 30 minutes  
**Actual Time**: 5 minutes  
**Priority**: High  
**Blocked By**: AI 2.1

**Definition of Done**:
- [x] `main.tf` with DVC and MLflow bucket resources
- [x] Variables defined in `variables.tf`
- [x] Outputs defined for bucket URLs
- [x] Lifecycle rules configured
- [x] Versioning enabled

**Files Created**:
- `infrastructure/terraform/modules/gcs/main.tf` - Bucket resources
- `infrastructure/terraform/modules/gcs/variables.tf` - Input variables
- `infrastructure/terraform/modules/gcs/outputs.tf` - Bucket URLs/names

**Resources Created**:
- `${project_id}-dvc-storage` bucket (with versioning & lifecycle)
- `${project_id}-mlflow-artifacts` bucket (with versioning & lifecycle)

**Blocking**: AI 2.6

---

### [AI] 2.3: Create Artifact Registry Module ✅

**Status**: ✅ Complete (2026-02-04)  
**Estimated Time**: 20 minutes  
**Actual Time**: 3 minutes  
**Priority**: Medium  
**Blocked By**: AI 2.1

**Definition of Done**:
- [x] Docker repository resource created
- [x] Variables and outputs defined
- [x] Repository URL output available

**Files Created**:
- `infrastructure/terraform/modules/artifact-registry/main.tf` - Docker repository
- `infrastructure/terraform/modules/artifact-registry/variables.tf` - Input variables
- `infrastructure/terraform/modules/artifact-registry/outputs.tf` - Repository URL

**Resources Created**:
- Docker repository: `${var.repository_id}` (default: "ml-images")

**Blocking**: None

---

### [AI] 2.4: Create MLflow Cloud Run Module ✅

**Status**: ✅ Complete (2026-02-04)  
**Estimated Time**: 45 minutes  
**Actual Time**: 5 minutes  
**Priority**: High  
**Blocked By**: AI 2.1

**Definition of Done**:
- [x] Cloud Run service resource
- [x] Service account with GCS access
- [x] IAM bindings for invoker
- [x] Environment variables configured
- [x] Scaling configured (0-2 instances)

**Files Created**:
- `infrastructure/terraform/modules/mlflow/main.tf` - Cloud Run service, SA, IAM
- `infrastructure/terraform/modules/mlflow/variables.tf` - Input variables
- `infrastructure/terraform/modules/mlflow/outputs.tf` - Service URL, SA email

**Resources Created**:
- Cloud Run service: `mlflow-server` (scale 0-2, SQLite backend)
- Service account: `mlflow-server@${project}.iam.gserviceaccount.com`
- GCS access: `roles/storage.objectAdmin` on artifacts bucket

**Features**:
- MLflow server on Cloud Run
- SQLite backend (for dev)
- GCS artifact storage
- Autoscaling 0-2 instances

**Blocking**: AI 2.6

---

### [AI] 2.5: Create GKE Module (Optional) ⏭️

**Status**: ⏭️ Skipped (User Decision)  
**Priority**: Low  
**Note**: Deferred to Phase 7 when Kubernetes is needed

---

### [AI] 2.6: Create Dev Environment Config ✅

**Status**: ✅ Complete (2026-02-04)  
**Estimated Time**: 30 minutes  
**Actual Time**: 8 minutes  
**Priority**: High  
**Blocked By**: AI 2.1, AI 2.2, AI 2.3, AI 2.4

**Definition of Done**:
- [x] `main.tf` with all module calls
- [x] `variables.tf` with project-specific vars
- [x] `terraform.tfvars.example` template
- [x] `backend.tf.example` template
- [x] Outputs defined for all important values

**Files Created**:
- `infrastructure/terraform/environments/dev/main.tf` - Orchestrates all modules
- `infrastructure/terraform/environments/dev/variables.tf` - 10+ configurable variables
- `infrastructure/terraform/environments/dev/outputs.tf` - All important URLs and names
- `infrastructure/terraform/environments/dev/terraform.tfvars.example` - Configuration template
- `infrastructure/terraform/environments/dev/backend.tf.example` - Remote state template
- `infrastructure/terraform/environments/dev/.gitignore` - Prevents committing sensitive files

**Features**:
- Calls GCS, Artifact Registry, and MLflow modules
- Configurable region, scaling, lifecycle rules
- Remote state configuration for GCS backend
- Comprehensive outputs with usage instructions

**Blocking**: Phase 2 completion

---

## Verification

**Prerequisites**: All AI 2.x tasks complete

```bash
cd infrastructure/terraform/environments/dev

# Copy example files
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values

cp backend.tf.example backend.tf
# Edit backend.tf with your state bucket

# Initialize
terraform init

# Plan
terraform plan

# Apply
terraform apply

# Verify outputs
terraform output

echo "Phase 2 complete!"
```

---

## Next Sprint

**Sprint 3**: GitHub Setup
- [HUMAN] 3.1: Create GitHub Repository
- [HUMAN] 3.2: Create GitHub Service Account
- [HUMAN] 3.3: Add GitHub Repository Secrets
- [AI] 3.1-3.3: GitHub Actions workflows
