# Sprint 8: Manual Verifications

**Status**: ⬜ Not Started  
**Sprint Goal**: Validate all components work end-to-end

---

## Summary

Comprehensive manual testing of the entire MLOps pipeline. Verify that local pipeline, DVC remote, MLflow Cloud Run, and Docker images all work correctly.

**Current Status**: 0/4 tasks complete

---

## Tasks

### [HUMAN] 7.1: Test Full Pipeline Locally ⬜

**Status**: ⬜ Not Started  
**Assigned To**: Human

**Definition of Done**:
- [ ] DVC pipeline runs successfully
- [ ] All outputs generated
- [ ] Metrics look reasonable

**Commands**:
```bash
cd projects/synth_tabular_classification

# Activate venv
source ../../.venv/bin/activate

# Run pipeline
dvc repro

# Check outputs
ls data/raw/
ls data/processed/
ls models/
cat metrics/metrics.json

# Start MLflow UI locally
mlflow ui --backend-store-uri file://$(pwd)/mlruns

# Open http://localhost:5000 and verify experiments
```

---

### [HUMAN] 7.2: Test DVC Push to GCS ⬜

**Status**: ⬜ Not Started  
**Assigned To**: Human

**Definition of Done**:
- [ ] DVC push succeeds
- [ ] Data visible in GCS bucket

**Commands**:
```bash
cd projects/synth_tabular_classification

# Push to remote
dvc push

# Verify in GCS
gsutil ls gs://YOUR_PROJECT_ID-dvc-storage/
```

---

### [HUMAN] 7.3: Test MLflow Connection to Cloud Run ⬜

**Status**: ⬜ Not Started  
**Assigned To**: Human

**Definition of Done**:
- [ ] Local training logs to Cloud Run MLflow
- [ ] Experiments visible in Cloud Run UI

**Commands**:
```bash
cd projects/synth_tabular_classification

# Set environment variable to Cloud Run MLflow
export MLFLOW_TRACKING_URI=https://mlflow-server-XXXX.run.app

# Authenticate
gcloud auth print-identity-token > /tmp/token
export MLFLOW_TRACKING_TOKEN=$(cat /tmp/token)

# Run training (should log to Cloud Run MLflow)
python -m src.model.train

# Check Cloud Run MLflow UI for new run
```

---

### [HUMAN] 7.4: Test Docker Build and Push ⬜

**Status**: ⬜ Not Started  
**Assigned To**: Human

**Definition of Done**:
- [ ] Docker image builds successfully
- [ ] Image pushes to Artifact Registry
- [ ] Image visible in console

**Commands**:
```bash
PROJECT_ID=$(gcloud config get-value project)
REGION=us-central1

# Configure Docker for Artifact Registry
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Build image
cd projects/synth_tabular_classification
docker build -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/ml/inference:test .

# Push to Artifact Registry
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/ml/inference:test

# Verify in Console
# Go to: https://console.cloud.google.com/artifacts
```

---

## Success Criteria

All verifications passing means:
1. ✅ Local ML pipeline works
2. ✅ DVC remote storage works
3. ✅ MLflow Cloud Run works
4. ✅ Docker images build and push

The MLOps infrastructure is fully operational!

---

## Next Phase

**Phase 9**: Ongoing Maintenance
- [HUMAN] 8.1: Monitor Costs (weekly)
- [HUMAN] 8.2: Rotate Service Account Keys (quarterly)
