# Sprint 4: DVC Remote Setup

**Status**: ⬜ Not Started  
**Sprint Goal**: Configure DVC to use GCS for remote data storage

---

## Summary

Set up a GCS bucket as the DVC remote storage backend. This allows data and model artifacts to be stored in the cloud and shared across team members.

**Current Status**: 0/2 tasks complete

---

## Prerequisites

None - Can be done in parallel with Sprint 3

---

## Tasks

### [HUMAN] 4.1: Create GCS Bucket for DVC ⬜

**Status**: ⬜ Not Started  
**Assigned To**: Human

**Definition of Done**:
- [ ] GCS bucket created for DVC data storage
- [ ] Bucket name: `${PROJECT_ID}-dvc-storage`
- [ ] Location: us-central1

**Commands**:
```bash
PROJECT_ID=$(gcloud config get-value project)

# Create bucket
gsutil mb -l us-central1 gs://${PROJECT_ID}-dvc-storage

# Verify
gsutil ls
```

**Write down your DVC bucket**: `gs://deepmlhub-__________________-dvc-storage`

---

### [HUMAN] 4.2: Configure DVC Remote ⬜

**Status**: ⬜ Not Started  
**Assigned To**: Human

**Definition of Done**:
- [ ] DVC remote added pointing to GCS bucket
- [ ] Credentials configured
- [ ] DVC config updated

**Commands**:
```bash
cd /Users/unmeshmali/Downloads/Unmesh/deepmlhub

# Add GCS remote
dvc remote add -d gcs gs://YOUR_PROJECT_ID-dvc-storage

# Configure GCS credentials
dvc remote modify gcs credentialpath ~/.config/gcloud/terraform-key.json

# Verify
dvc remote list
cat .dvc/config
```

---

## Verification

```bash
cd projects/synth_tabular_classification

# Push to remote
dvc push

# Verify in GCS
gsutil ls gs://YOUR_PROJECT_ID-dvc-storage/
```

---

## Next Sprint

**Sprint 5**: Apply Terraform Infrastructure
- [HUMAN] 5.1: Initialize Terraform
- [HUMAN] 5.2: Review Terraform Plan
- [HUMAN] 5.3: Apply Terraform
- [HUMAN] 5.4: Verify MLflow Deployment
