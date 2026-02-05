# Active Sprint: Apply Terraform Infrastructure

**Sprint**: sprint_05_terraform_apply  
**Last Updated**: 2026-02-05  
**Status**: 🔄 In Progress

---

## Sprint Goal

Apply the Terraform configuration to create actual GCP infrastructure including GCS buckets, MLflow on Cloud Run, and Artifact Registry.

---

## What's Happening Now

### ⬜ Prerequisites (Ready to Start)

| Task | Assignee | Status | Dependencies |
|------|----------|--------|--------------|
| [HUMAN] 5.1: Initialize Terraform | Human | ⬜ Not Started | Terraform modules from Sprint 2 |
| [HUMAN] 5.2: Review Terraform Plan | Human | ⬜ Not Started | HUMAN 5.1 |
| [HUMAN] 5.3: Apply Terraform | Human | ⬜ Not Started | HUMAN 5.2 |
| [HUMAN] 5.4: Verify MLflow Deployment | Human | ⬜ Not Started | HUMAN 5.3 |

---

## Sprint Metrics

**Tasks**: 0/4 Complete (0%)  
**Blockers**: None  
**ETA**: ~45 minutes

---

## Current State

### ✅ Pre-Configured (from Sprint 2)
- Terraform modules created (GCS, Artifact Registry, MLflow)
- Dev environment configuration exists
- `terraform.tfvars` already configured with project ID
- `backend.tf` already configured with state bucket
- `.terraform` directory exists (previously initialized)

### 📍 Location
```
infrastructure/terraform/environments/dev/
├── backend.tf          ✅ Configured
├── terraform.tfvars    ✅ Configured  
├── main.tf            ✅ Modules defined
├── variables.tf       ✅ Variables defined
├── outputs.tf         ✅ Outputs defined
└── .terraform/        ✅ Initialized
```

---

## Prerequisites Detail

### [HUMAN] 5.1: Initialize Terraform

**Status**: ⬜ Not Started  
**Time**: ~5 minutes

**Tasks**:
1. Navigate to terraform directory
2. Run `terraform init` (or verify existing initialization)
3. Confirm backend is configured correctly

**Commands**:
```bash
cd infrastructure/terraform/environments/dev

# Initialize (if not already done)
terraform init

# Verify backend
terraform workspace list
```

**Verification**: No errors, backend configured with GCS bucket

---

### [HUMAN] 5.2: Review Terraform Plan

**Status**: ⬜ Not Started  
**Time**: ~10 minutes

**Tasks**:
1. Generate terraform plan
2. Review all resources to be created
3. Understand costs and implications

**Commands**:
```bash
cd infrastructure/terraform/environments/dev

# See what will be created
terraform plan
```

**Expected Resources**:
- GCS bucket for MLflow artifacts
- Cloud Run service for MLflow server
- Artifact Registry repository
- Service accounts with IAM bindings

**Verification**: Plan shows expected resources, no errors

---

### [HUMAN] 5.3: Apply Terraform

**Status**: ⬜ Not Started  
**Time**: ~15 minutes

**Tasks**:
1. Execute terraform apply
2. Confirm with 'yes'
3. Wait for resources to be created
4. Save outputs

**Commands**:
```bash
cd infrastructure/terraform/environments/dev

# Apply infrastructure
terraform apply

# Type 'yes' when prompted

# Save outputs
terraform output > deployment_outputs.txt
```

**Expected Outputs**:
- `mlflow_url`: URL for MLflow Cloud Run service
- `artifact_registry_repository`: Docker repository URL
- `gcs_bucket_name`: MLflow artifacts bucket

**Verification**: All resources created without errors

---

### [HUMAN] 5.4: Verify MLflow Deployment

**Status**: ⬜ Not Started  
**Time**: ~15 minutes

**Tasks**:
1. Get MLflow URL from terraform output
2. Test MLflow Cloud Run service
3. Verify UI is accessible

**Commands**:
```bash
cd infrastructure/terraform/environments/dev

# Get MLflow URL
MLFLOW_URL=$(terraform output -raw mlflow_url)
echo $MLFLOW_URL

# Test with authentication
gcloud auth print-identity-token | \
  xargs -I {} curl -H "Authorization: Bearer {}" $MLFLOW_URL
```

**Alternative - Cloud Console**:
1. Go to https://console.cloud.google.com/run
2. Find `mlflow-server` service
3. Click URL to open UI

**Verification**: MLflow UI loads successfully

---

## Resources to be Created

### GCS Buckets
- `deepmlhub-voiceoffers-mlflow-artifacts` - MLflow experiment artifacts

### Cloud Run
- `mlflow-server` - MLflow tracking server

### Artifact Registry
- `ml-images` - Docker images repository

### IAM
- Service accounts for Cloud Run and GCS access

---

## Cost Considerations

**Estimated Monthly Costs**:
- Cloud Run: $0 (scales to zero when idle)
- GCS: ~$0.02/GB/month
- Artifact Registry: ~$0.10/GB/month

**Total**: <$1/month for light usage

---

## Next Sprint

**Sprint 6**: Docker Images
- [AI] 6.1: Create Training Dockerfile
- [AI] 6.2: Create Inference Dockerfile
- [AI] 6.3: Create .dockerignore

---

## Quick Links

- [Sprint Tasks](sprints/sprint_05_terraform_apply/tasks.md)
- [Master Backlog](backlog.md)
- [Previous Sprint: DVC Remote](sprints/sprint_04_dvc_remote/tasks.md)
- [Next Sprint: Docker Images](sprints/sprint_06_docker/tasks.md)
- [Architecture Plan](plans/mlops_plan_uno.md)
- [Terraform Directory](../../infrastructure/terraform/environments/dev/)
