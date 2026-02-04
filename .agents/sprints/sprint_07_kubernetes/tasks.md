# Sprint 7: Kubernetes Manifests (Optional)

**Status**: ⬜ Not Started  
**Sprint Goal**: Create Kubernetes manifests for GKE deployment

---

## Summary

Create Kubernetes manifests for deploying the inference API to GKE. This sprint is optional and can be deferred if Cloud Run is sufficient.

**Current Status**: 0/5 tasks complete  
**Prerequisites**: Sprint 6 (Docker images created)

---

## Prerequisites

### [HUMAN] 6.1: Apply GKE Terraform ⬜

**Status**: ⬜ Not Started (Optional)  
**Assigned To**: Human

**Definition of Done**:
- [ ] GKE Terraform module applied
- [ ] GKE cluster created
- [ ] Node pools configured

**Commands**:
```bash
cd infrastructure/terraform/environments/dev

# Apply with GKE module enabled
terraform apply -target=module.gke
```

**Warning**: GKE clusters cost money even when idle (~$70/month for control plane on Autopilot, free on Standard). The Terraform is configured for Standard with scale-to-zero nodes.

**Blocking**: AI 7.1-7.2

---

### [HUMAN] 6.2: Get GKE Credentials ⬜

**Status**: ⬜ Not Started  
**Assigned To**: Human

**Definition of Done**:
- [ ] kubectl configured for cluster
- [ ] Can list nodes and namespaces

**Commands**:
```bash
PROJECT_ID=$(gcloud config get-value project)

gcloud container clusters get-credentials deepmlhub-cluster \
  --zone us-central1-a \
  --project $PROJECT_ID

# Verify
kubectl get nodes
kubectl get namespaces
```

**Blocking**: AI 7.1-7.2

---

### [HUMAN] 6.3: Install kubectl (If Not Installed) ⬜

**Status**: ⬜ Not Started  
**Assigned To**: Human

**Definition of Done**:
- [ ] kubectl CLI installed
- [ ] Version compatible with cluster

**Commands**:
```bash
# Using Homebrew
brew install kubectl

# Or via gcloud
gcloud components install kubectl

# Verify
kubectl version --client
```

**Blocking**: AI 7.1-7.2

---

## AI Tasks

### [AI] 7.1: Create Namespace and ConfigMaps 🚫

**Status**: 🚫 Blocked (waiting on Human 6.1-6.3)  
**Priority**: High

**Definition of Done**:
- [ ] Namespace manifest created
- [ ] ConfigMaps for configuration
- [ ] Kustomization base structure

**Files**:
- `infrastructure/kubernetes/base/namespace.yaml`
- `infrastructure/kubernetes/base/configmap.yaml`
- `infrastructure/kubernetes/base/kustomization.yaml`

---

### [AI] 7.2: Create Inference Deployment 🚫

**Status**: 🚫 Blocked  
**Priority**: High

**Definition of Done**:
- [ ] Deployment manifest created
- [ ] Service manifest created
- [ ] Health checks configured
- [ ] Resource limits set
- [ ] Uses Artifact Registry image

**Files**:
- `infrastructure/kubernetes/inference/deployment.yaml`
- `infrastructure/kubernetes/inference/service.yaml`
- `infrastructure/kubernetes/inference/kustomization.yaml`

**Features**:
- Deployment with 1 replica
- LoadBalancer service
- Liveness and readiness probes
- Resource requests/limits
- Environment variables from ConfigMap

---

## Verification

After all prerequisites complete:

```bash
# Apply base resources
kubectl apply -f infrastructure/kubernetes/base/

# Apply inference resources
kubectl apply -f infrastructure/kubernetes/inference/

# Check status
kubectl get pods -n ml-inference
kubectl get svc -n ml-inference

# Get external IP
kubectl get svc synth-tabular-api -n ml-inference

# Test
curl http://EXTERNAL_IP/health
```

---

## Next Sprint

**Sprint 8**: Manual Verifications
- [HUMAN] 7.1: Test Full Pipeline Locally
- [HUMAN] 7.2: Test DVC Push to GCS
- [HUMAN] 7.3: Test MLflow Connection to Cloud Run
- [HUMAN] 7.4: Test Docker Build and Push
