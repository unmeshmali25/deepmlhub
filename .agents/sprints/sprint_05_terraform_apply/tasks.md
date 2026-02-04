# Sprint 5: Apply Terraform Infrastructure

**Status**: ⬜ Not Started  
**Sprint Goal**: Create GCP infrastructure using Terraform

---

## Summary

Apply the Terraform configuration created in Sprint 2 to create actual GCP resources including GCS buckets, MLflow on Cloud Run, and Artifact Registry.

**Current Status**: 0/4 tasks complete

---

## Prerequisites

- Sprint 2 (Terraform modules) must be complete
- Sprint 3 (GitHub setup) should be in progress or complete

---

## Tasks

### [HUMAN] 5.1: Initialize Terraform ⬜

**Status**: ⬜ Not Started  
**Assigned To**: Human

**Definition of Done**:
- [ ] Terraform initialized successfully
- [ ] Backend configured with state bucket
- [ ] Providers downloaded

**Commands**:
```bash
cd infrastructure/terraform/environments/dev

# Copy example files
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your project ID and email

cp backend.tf.example backend.tf
# Edit backend.tf with your state bucket

# Initialize
terraform init
```

**Note**: If you see errors about the backend bucket, ensure Sprint 2.0.8 is complete.

---

### [HUMAN] 5.2: Review Terraform Plan ⬜

**Status**: ⬜ Not Started  
**Assigned To**: Human

**Definition of Done**:
- [ ] Terraform plan reviewed
- [ ] Understand what will be created
- [ ] No surprises in the plan

**Commands**:
```bash
cd infrastructure/terraform/environments/dev

# See what will be created
terraform plan

# Review the output carefully:
# - GCS buckets
# - Service accounts
# - Cloud Run service (MLflow)
# - GKE cluster (if included)
# - Artifact Registry
```

**Important**: Review the plan before applying. Ask questions if unsure.

---

### [HUMAN] 5.3: Apply Terraform ⬜

**Status**: ⬜ Not Started  
**Assigned To**: Human

**Definition of Done**:
- [ ] Terraform apply executed
- [ ] All resources created successfully
- [ ] Outputs captured

**Commands**:
```bash
cd infrastructure/terraform/environments/dev

# Apply (type 'yes' when prompted)
terraform apply
```

**This will create**:
- GCS bucket for MLflow
- Cloud Run service for MLflow
- Artifact Registry for Docker images
- Service accounts with proper IAM

**Save the outputs**:
```bash
terraform output
# Write down the MLflow URL and other outputs
```

**MLflow URL**: `https://mlflow-server-____________________.run.app`

---

### [HUMAN] 5.4: Verify MLflow Deployment ⬜

**Status**: ⬜ Not Started  
**Assigned To**: Human

**Definition of Done**:
- [ ] MLflow Cloud Run service is running
- [ ] MLflow UI is accessible
- [ ] Can authenticate and view UI

**Commands**:
```bash
# Get the MLflow URL
MLFLOW_URL=$(terraform output -raw mlflow_url)

# Test health (may need authentication)
curl $MLFLOW_URL

# Or authenticate first
gcloud auth print-identity-token | xargs -I {} curl -H "Authorization: Bearer {}" $MLFLOW_URL
```

**Alternative**: Check in Cloud Console
1. Go to: https://console.cloud.google.com/run
2. Find `mlflow-server` service
3. Click the URL to open MLflow UI

---

## Verification

After all tasks complete:

```bash
# Verify GCS buckets
gsutil ls

# Verify Cloud Run
gcloud run services list

# Verify Artifact Registry
gcloud artifacts repositories list

# Get MLflow URL
terraform output mlflow_url
```

---

## Next Sprint

**Sprint 6**: Docker Images
- [AI] 6.1: Create Training Dockerfile
- [AI] 6.2: Create Inference Dockerfile
- [AI] 6.3: Create .dockerignore
