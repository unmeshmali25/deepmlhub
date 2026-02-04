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

### [AI] 2.1: Create Terraform Directory Structure 🔄

**Status**: 🔄 In Progress (2026-02-04)  
**Estimated Time**: 10 minutes  
**Priority**: High

**Definition of Done**:
- [ ] Directory structure created per plan
- [ ] All directories have .gitkeep if empty
- [ ] Structure matches plan specification

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

### [AI] 2.2: Create GCS Module ⬜

**Status**: ⬜ Not Started  
**Estimated Time**: 30 minutes  
**Priority**: High  
**Blocked By**: AI 2.1

**Definition of Done**:
- [ ] `main.tf` with DVC and MLflow bucket resources
- [ ] Variables defined in `variables.tf`
- [ ] Outputs defined for bucket URLs
- [ ] Lifecycle rules configured
- [ ] Versioning enabled

**File**: `infrastructure/terraform/modules/gcs/main.tf`

**Resources**:
- DVC storage bucket with versioning and lifecycle rules
- MLflow artifacts bucket with versioning

**Blocking**: AI 2.6

---

### [AI] 2.3: Create Artifact Registry Module ⬜

**Status**: ⬜ Not Started  
**Estimated Time**: 20 minutes  
**Priority**: Medium  
**Blocked By**: AI 2.1

**Definition of Done**:
- [ ] Docker repository resource created
- [ ] Variables and outputs defined
- [ ] Repository URL output available

**File**: `infrastructure/terraform/modules/artifact-registry/main.tf`

**Blocking**: None (can run in parallel with 2.2, 2.4)

---

### [AI] 2.4: Create MLflow Cloud Run Module ⬜

**Status**: ⬜ Not Started  
**Estimated Time**: 45 minutes  
**Priority**: High  
**Blocked By**: AI 2.1

**Definition of Done**:
- [ ] Cloud Run service resource
- [ ] Service account with GCS access
- [ ] IAM bindings for invoker
- [ ] Environment variables configured
- [ ] Scaling configured (0-2 instances)

**File**: `infrastructure/terraform/modules/mlflow/main.tf`

**Features**:
- MLflow server on Cloud Run
- SQLite backend (for dev)
- GCS artifact storage
- Autoscaling 0-2 instances

**Blocking**: AI 2.6

---

### [AI] 2.5: Create GKE Module (Optional) ⬜

**Status**: ⬜ Not Started (Optional)  
**Estimated Time**: 60 minutes  
**Priority**: Low  
**Blocked By**: AI 2.1

**Definition of Done**:
- [ ] GKE Standard cluster resource
- [ ] CPU node pool with Spot VMs
- [ ] Workload Identity enabled
- [ ] Autoscaling configured

**File**: `infrastructure/terraform/modules/gke/main.tf`

**Note**: Optional for this sprint, can be deferred to Phase 7

**Blocking**: None

---

### [AI] 2.6: Create Dev Environment Config ⬜

**Status**: ⬜ Not Started  
**Estimated Time**: 30 minutes  
**Priority**: High  
**Blocked By**: AI 2.1, AI 2.2, AI 2.3, AI 2.4

**Definition of Done**:
- [ ] `main.tf` with all module calls
- [ ] `variables.tf` with project-specific vars
- [ ] `terraform.tfvars.example` template
- [ ] `backend.tf.example` template
- [ ] Outputs defined for all important values

**Files**:
- `infrastructure/terraform/environments/dev/main.tf`
- `infrastructure/terraform/environments/dev/variables.tf`
- `infrastructure/terraform/environments/dev/terraform.tfvars.example`
- `infrastructure/terraform/environments/dev/backend.tf.example`
- `infrastructure/terraform/environments/dev/outputs.tf`

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
