# Sprint 2: GCP & Terraform Infrastructure

**Status**: ✅ COMPLETE  
**Started**: 2026-02-04  
**Completed**: 2026-02-04  
**Duration**: 23 minutes  
**Sprint Goal**: Create reusable Terraform infrastructure modules for GCP resources (GCS, MLflow on Cloud Run, Artifact Registry)

---

## Summary

Successfully set up GCP infrastructure using Terraform. All resources created and verified active in GCP Console.

**Final Status**: 5/5 tasks complete, 1 skipped (GKE)  
**Verification**: User confirmed all resources exist and are working  
**Blockers**: None

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

---

## Completed Tasks

### [AI] 2.1: Create Terraform Directory Structure ✅
**Duration**: 2 minutes (estimated 10 min)

Created:
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

---

### [AI] 2.2: Create GCS Module ✅
**Duration**: 5 minutes (estimated 30 min)

**Resources Created**:
- `deepmlhub-voiceoffers-dvc-storage` bucket
- `deepmlhub-voiceoffers-mlflow-artifacts` bucket

**Features**:
- Versioning enabled
- Lifecycle rules (30-day cleanup)
- Uniform bucket-level access

---

### [AI] 2.3: Create Artifact Registry Module ✅
**Duration**: 3 minutes (estimated 20 min)

**Resources Created**:
- Docker repository: `ml-images`

**Location**: `us-central1-docker.pkg.dev`

---

### [AI] 2.4: Create MLflow Cloud Run Module ✅
**Duration**: 5 minutes (estimated 45 min)

**Resources Created**:
- Cloud Run service: `mlflow-server`
- Service account: `mlflow-server@deepmlhub-voiceoffers.iam.gserviceaccount.com`
- IAM: `roles/storage.objectAdmin` on artifacts bucket
- IAM: `roles/run.invoker` for public access

**Features**:
- SQLite backend (dev)
- GCS artifact storage
- Autoscaling: 0-2 instances (scales to zero)
- Publicly accessible

---

### [AI] 2.5: Create GKE Module ⏭️
**Status**: SKIPPED (deferred to Phase 7)

**Reason**: User decided to skip optional GKE module

---

### [AI] 2.6: Create Dev Environment Config ✅
**Duration**: 8 minutes (estimated 30 min)

**Files Created**:
- `main.tf` - Orchestrates all modules
- `variables.tf` - 10+ configurable variables
- `outputs.tf` - Comprehensive outputs with URLs
- `terraform.tfvars.example` - Configuration template
- `backend.tf.example` - Remote state template
- `.gitignore` - Protects sensitive files

---

## Verification Results

**Date**: 2026-02-04  
**Verified By**: User via GCP Console

✅ All resources confirmed active:
- 2 GCS buckets visible in Storage Browser
- 1 Artifact Registry repository listed
- 1 Cloud Run service deployed and running
- Terraform state stored in GCS backend

**MLflow URL**: Available via `terraform output mlflow_tracking_url`

---

## Files Created

```
infrastructure/terraform/
├── environments/
│   └── dev/
│       ├── .gitignore
│       ├── backend.tf.example
│       ├── main.tf
│       ├── outputs.tf
│       ├── terraform.tfvars.example
│       └── variables.tf
└── modules/
    ├── artifact-registry/
    │   ├── main.tf
    │   ├── outputs.tf
    │   └── variables.tf
    ├── gcs/
    │   ├── main.tf
    │   ├── outputs.tf
    │   └── variables.tf
    └── mlflow/
        ├── main.tf
        ├── outputs.tf
        └── variables.tf
```

---

## Lessons Learned

1. **IAM Permissions**: Initially missed `roles/run.admin` for Cloud Run IAM policy - quickly resolved
2. **Module Pattern**: Modular Terraform structure worked well for reusability
3. **Autoscaling**: 0-2 instance config saves costs while maintaining availability
4. **State Management**: GCS backend configuration is clean and simple

---

## Next Sprint

**Sprint 3**: GitHub Setup
- Setup GitHub repository
- Configure CI/CD with GitHub Actions
- Create workflows for build, test, deploy
