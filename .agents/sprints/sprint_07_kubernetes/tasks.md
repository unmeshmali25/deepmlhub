# Sprint 7: Kubernetes Manifests

**Status**: 🔄 AI Tasks Complete, Waiting for GKE  
**Sprint Goal**: Create Kubernetes manifests for GKE deployment  
**Started**: 2026-02-11  
**AI Tasks Completed**: 2026-02-11

---

## Summary

Created complete Kubernetes manifests for deploying the inference API to GKE with security best practices, autoscaling, and multi-environment support.

**Current Status**: 4/4 AI tasks complete (100%)  
**Prerequisites**: Sprint 6 (Docker images created) ✅ Complete

### Sprint Metrics
- **AI Tasks**: 4/4 Complete (100%)
- **Human Prerequisites**: 0/3 (Blocking deployment)
- **Total Files Created**: 11 manifests + documentation
- **Time Taken**: ~1.5 hours

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

### [AI] 7.1: Create Namespace and ConfigMaps ✅

**Status**: ✅ Complete (2026-02-11)  
**Priority**: High  
**Time Taken**: ~20 minutes

**Definition of Done**:
- [x] Namespace manifest created
- [x] ConfigMaps for configuration
- [x] Kustomization base structure

**Files Delivered**:
- `infrastructure/kubernetes/base/namespace.yaml` - ml-inference namespace
- `infrastructure/kubernetes/base/configmap.yaml` - Environment configuration
- `infrastructure/kubernetes/base/kustomization.yaml` - Base kustomization

**Features**:
- Namespace with proper labels
- ConfigMap with MLflow URI, model paths, server config
- Ready for multi-environment overlays

---

### [AI] 7.2: Create Inference Deployment ✅

**Status**: ✅ Complete (2026-02-11)  
**Priority**: High  
**Time Taken**: ~30 minutes

**Definition of Done**:
- [x] Deployment manifest created
- [x] Service manifest created
- [x] Health checks configured
- [x] Resource limits set
- [x] Uses Artifact Registry image
- [x] ServiceAccount created
- [x] HPA configured

**Files Delivered**:
- `infrastructure/kubernetes/inference/deployment.yaml` - Deployment with 2 replicas
- `infrastructure/kubernetes/inference/service.yaml` - LoadBalancer service
- `infrastructure/kubernetes/inference/serviceaccount.yaml` - Workload identity
- `infrastructure/kubernetes/inference/hpa.yaml` - Horizontal Pod Autoscaler

**Features**:
- Rolling update strategy (maxSurge: 1, maxUnavailable: 0)
- Liveness probe (/health, 30s interval)
- Readiness probe (/ready, 10s interval)
- Resource requests: 250m CPU, 256Mi memory
- Resource limits: 500m CPU, 512Mi memory
- Security context: non-root user (1000), drop all capabilities
---

### [AI] 7.3: Create Kustomize Overlays ✅

**Status**: ✅ Complete (2026-02-11)  
**Priority**: Medium  
**Time Taken**: ~30 minutes

**Definition of Done**:
- [x] Dev overlay created (1 replica, ClusterIP)
- [x] Production overlay created (3 replicas, higher resources)
- [x] Kustomization files with patches

**Files Delivered**:
- `infrastructure/kubernetes/overlays/dev/kustomization.yaml` - Dev environment
- `infrastructure/kubernetes/overlays/prod/kustomization.yaml` - Production environment

**Dev Features**:
- 1 replica
- ClusterIP service (no external LB)
- HPA min: 1, max: 2
- Debug logging

**Production Features**:
- 3 replicas
- Higher resource requests (512Mi, 500m)
- Higher resource limits (1Gi, 1000m)

---

### [AI] 7.4: Create Deployment Guide ✅

**Status**: ✅ Complete (2026-02-11)  
**Priority**: Medium  
**Time Taken**: ~15 minutes

**Definition of Done**:
- [x] Deployment commands documented
- [x] Architecture diagram described
- [x] Security features documented
- [x] Troubleshooting section
- [x] Test commands provided

**File Delivered**:
- `infrastructure/kubernetes/README.md`

**Sections**:
- Quick start deployment
- Environment-specific deployment
- Architecture overview
- Security features
- Scaling configuration
- Monitoring and health checks
- Troubleshooting guide
- Cleanup instructions

---

## Verification

**Status**: ⬜ Waiting for GKE cluster provisioning

### Prerequisites Checklist
- [ ] [HUMAN] 6.1: Apply GKE Terraform (creates cluster)
- [ ] [HUMAN] 6.2: Get GKE Credentials (kubectl configured)
- [ ] [HUMAN] 6.3: Install kubectl (if not installed)

### Deployment Commands (Ready to Use)

Once GKE cluster is ready:

```bash
# 1. Verify cluster connection
kubectl get nodes

# 2. Apply base resources
kubectl apply -k infrastructure/kubernetes/base/

# 3. Apply inference resources
kubectl apply -k infrastructure/kubernetes/inference/

# 4. Check deployment status
kubectl get pods -n ml-inference -w
kubectl get deployments -n ml-inference
kubectl get svc -n ml-inference

# 5. Get external IP (wait for LoadBalancer)
kubectl get svc inference-api -n ml-inference

# 6. Test the API
EXTERNAL_IP=$(kubectl get svc inference-api -n ml-inference -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
curl http://$EXTERNAL_IP/health
curl http://$EXTERNAL_IP/ready

# 7. Test prediction
curl -X POST http://$EXTERNAL_IP/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1.0, 0.1, -0.1, 0.2]]}'
```

### Validation Checklist
- [ ] Namespace `ml-inference` created
- [ ] ConfigMap `inference-config` created
- [ ] Deployment `inference-api` running (2 pods)
- [ ] Service `inference-api` has external IP
- [ ] Health endpoint returns 200
- [ ] Prediction endpoint returns results
- [ ] HPA is active and scaling

---

## Next Sprint

**Sprint 8**: Manual Verifications
- [HUMAN] 7.1: Test Full Pipeline Locally
- [HUMAN] 7.2: Test DVC Push to GCS
- [HUMAN] 7.3: Test MLflow Connection to Cloud Run
- [HUMAN] 7.4: Test Docker Build and Push
