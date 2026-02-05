# Sprint 4: DVC Remote Setup

**Status**: ✅ Complete  
**Sprint Goal**: Configure DVC to use GCS for remote data storage  
**Completion Date**: 2026-02-05

---

## Summary

Successfully set up a GCS bucket as the DVC remote storage backend for the synth_tabular_classification project. Data and model artifacts can now be stored in the cloud and shared across team members.

**Current Status**: 5/5 tasks complete (100%)

---

## Completed Tasks

### [HUMAN] 4.1: Create GCS Bucket for DVC ✅

**Status**: ✅ Complete  
**Completed**: 2026-02-05  
**Assigned To**: Human

**Definition of Done**:
- [x] GCS bucket created for DVC data storage
- [x] Bucket name: `deepmlhub-voiceoffers-dvc`
- [x] Location: us-central1
- [x] Uniform bucket-level access enabled

**Commands Used**:
```bash
gcloud storage buckets create gs://deepmlhub-voiceoffers-dvc \
  --project=deepmlhub-voiceoffers \
  --location=us-central1 \
  --uniform-bucket-level-access
```

**Verification**:
```bash
gcloud storage buckets list --project=deepmlhub-voiceoffers
```

---

### [HUMAN] 4.2: Configure DVC Remote ✅

**Status**: ✅ Complete  
**Completed**: 2026-02-05  
**Assigned To**: Human

**Definition of Done**:
- [x] DVC remote added pointing to GCS bucket
- [x] Credentials configured
- [x] DVC config updated

**Commands Used**:
```bash
cd projects/synth_tabular_classification
dvc remote add -d gcs gs://deepmlhub-voiceoffers-dvc
dvc remote modify gcs credentialpath ~/.config/gcloud/application_default_credentials.json
```

**Verification**:
```bash
dvc remote list
cat .dvc/config
```

---

### [AI] 4.3: Initialize DVC in synth_tabular_classification ✅

**Status**: ✅ Complete  
**Completed**: 2026-02-05  
**Assigned To**: AI

**Definition of Done**:
- [x] DVC initialized in project directory
- [x] Uses --subdir flag (project is within main git repo)
- [x] .dvc directory created with proper structure

**Commands Used**:
```bash
cd projects/synth_tabular_classification
dvc init --subdir
```

**Files Created**:
- `.dvc/config`
- `.dvc/.gitignore`
- `.dvcignore`

---

### [AI] 4.4: Configure DVC Remote for Project-Level Setup ✅

**Status**: ✅ Complete  
**Completed**: 2026-02-05  
**Assigned To**: AI

**Definition of Done**:
- [x] GCS remote added as default
- [x] Credentials path configured
- [x] .gitignore fixed to allow .dvc files

**Configuration**:
```ini
[core]
    remote = gcs
['remote "gcs"']
    url = gs://deepmlhub-voiceoffers-dvc
    credentialpath = /Users/unmeshmali/.config/gcloud/application_default_credentials.json
```

**Gitignore Fix**:
Changed from ignoring entire `/data/raw/` directory to only ignoring data files:
```gitignore
# Data (DVC tracked) - ignore actual data files but keep .dvc metadata
/data/raw/*.csv
/data/raw/*.json
/data/raw/*.parquet
/data/processed/*.csv
/data/processed/*.json
/data/processed/*.parquet
```

---

### [AI] 4.5: Test Project-Level DVC Pipeline ✅

**Status**: ✅ Complete  
**Completed**: 2026-02-05  
**Assigned To**: AI

**Definition of Done**:
- [x] Data file tracked with DVC
- [x] Data pushed to GCS remote
- [x] Verified files exist in GCS bucket
- [x] Authentication working correctly

**Test Commands**:
```bash
# Track data
dvc add data/raw/synthetic_data.csv

# Push to remote
dvc push

# Verify in GCS
gcloud storage ls gs://deepmlhub-voiceoffers-dvc/
```

**Results**:
- Successfully tracked: `data/raw/synthetic_data.csv`
- Pushed to: `gs://deepmlhub-voiceoffers-dvc/files/`
- File size: ~500KB

---

## Issues Encountered & Resolutions

### Issue 1: .dvc Files Git-Ignored
**Problem**: `.dvc` metadata files were being ignored by git
**Root Cause**: `.gitignore` was ignoring entire `/data/raw/` directory
**Solution**: Modified `.gitignore` to only ignore actual data files (`.csv`, `.json`, `.parquet`)

### Issue 2: Authentication Error (401)
**Problem**: `dvc push` failed with "Invalid Credentials, 401"
**Root Cause**: Application default credentials not set up
**Solution**: Ran `gcloud auth application-default login`

---

## Verification Commands

```bash
cd projects/synth_tabular_classification

# Check DVC status
dvc status

# List tracked files
dvc list

# Verify remote
dvc remote list

# Check GCS bucket
gcloud storage ls gs://deepmlhub-voiceoffers-dvc/ -r
```

---

## Files Changed

1. `projects/synth_tabular_classification/.dvc/config` - DVC configuration
2. `projects/synth_tabular_classification/.dvc/.gitignore` - DVC ignore rules
3. `projects/synth_tabular_classification/.dvcignore` - DVC ignore patterns
4. `projects/synth_tabular_classification/.gitignore` - Fixed to allow .dvc files
5. `projects/synth_tabular_classification/data/raw/synthetic_data.csv.dvc` - Tracked data metadata

---

## Lessons Learned

1. **Project-level DVC**: Each ML project needs independent DVC initialization
2. **Gitignore Strategy**: Must be careful to track `.dvc` files while ignoring actual data
3. **Authentication**: GCS authentication requires explicit login for application-default credentials
4. **Subdirectory Setup**: When DVC project is within another git repo, use `--subdir` flag

---

## Next Sprint

**Sprint 5**: Apply Terraform Infrastructure

### Prerequisites
- [HUMAN] 5.1: Initialize Terraform
- [HUMAN] 5.2: Review Terraform Plan
- [HUMAN] 5.3: Apply Terraform
- [HUMAN] 5.4: Verify MLflow Deployment

---

## Quick Links

- [Active Sprint](../active_sprint.md)
- [Master Backlog](../backlog.md)
- [Previous Sprint: GitHub Setup](../sprints/sprint_03_github_setup/tasks.md)
- [Next Sprint: Terraform Apply](../sprints/sprint_05_terraform_apply/tasks.md)
