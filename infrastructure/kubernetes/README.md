# GKE Deployment Guide

Deploy the ML inference API to Google Kubernetes Engine (GKE).

## Prerequisites

1. GKE cluster provisioned via Terraform
2. kubectl configured for the cluster
3. Docker images pushed to Artifact Registry

## Quick Start

### 1. Apply Base Resources

```bash
# Apply namespace and configmaps
kubectl apply -k infrastructure/kubernetes/base/

# Verify
kubectl get namespace ml-inference
kubectl get configmaps -n ml-inference
```

### 2. Apply Inference Resources

```bash
# Apply inference deployment, service, and HPA
kubectl apply -k infrastructure/kubernetes/inference/

# Verify
kubectl get deployments -n ml-inference
kubectl get services -n ml-inference
kubectl get pods -n ml-inference
```

### 3. Check Deployment Status

```bash
# Watch pods
kubectl get pods -n ml-inference -w

# Check logs
kubectl logs -n ml-inference -l app=inference-api

# Check service endpoint
kubectl get svc inference-api -n ml-inference
```

### 4. Test the API

```bash
# Get external IP
EXTERNAL_IP=$(kubectl get svc inference-api -n ml-inference -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# Health check
curl http://$EXTERNAL_IP/health

# Test prediction
curl -X POST http://$EXTERNAL_IP/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1.0, 0.1, -0.1, 0.2]]}'
```

## Environment-Specific Deployment

### Development

```bash
# Deploy to dev namespace with 1 replica
kubectl apply -k infrastructure/kubernetes/overlays/dev/

# Port forward for local testing
kubectl port-forward -n ml-inference-dev svc/inference-api-dev 8080:80

# Test locally
curl http://localhost:8080/health
```

### Production

```bash
# Deploy to production with 3 replicas and higher resources
kubectl apply -k infrastructure/kubernetes/overlays/prod/
```

## Architecture

```
infrastructure/kubernetes/
├── base/
│   ├── namespace.yaml      # ml-inference namespace
│   ├── configmap.yaml      # Environment configuration
│   └── kustomization.yaml  # Base kustomization
├── inference/
│   ├── serviceaccount.yaml # GKE workload identity
│   ├── deployment.yaml     # Inference API deployment
│   ├── service.yaml        # LoadBalancer service
│   ├── hpa.yaml           # Horizontal Pod Autoscaler
│   └── kustomization.yaml # Inference kustomization
└── overlays/
    ├── dev/
    │   └── kustomization.yaml  # Dev configuration
    └── prod/
        └── kustomization.yaml  # Production configuration
```

## Security Features

- **Non-root containers**: Runs as user 1000
- **Read-only root filesystem**: Enabled
- **Security context**: Drop all capabilities
- **Network policies**: Can be added for pod-to-pod communication
- **Workload Identity**: Service account ready for GCP IAM integration

## Scaling

The HorizontalPodAutoscaler automatically scales based on:
- **CPU**: Scales when >70% utilization
- **Memory**: Scales when >80% utilization
- **Min replicas**: 2 (production)
- **Max replicas**: 10

## Monitoring

### Health Checks

- **Liveness probe**: `/health` (30s interval)
- **Readiness probe**: `/ready` (10s interval)

### Resource Limits

| Resource | Request | Limit |
|----------|---------|-------|
| CPU | 250m | 500m |
| Memory | 256Mi | 512Mi |

## Troubleshooting

### Pod not starting

```bash
# Check events
kubectl describe pod -n ml-inference -l app=inference-api

# Check logs
kubectl logs -n ml-inference -l app=inference-api --previous
```

### Service not getting external IP

```bash
# Check service status
kubectl get svc inference-api -n ml-inference

# Check for errors
kubectl describe svc inference-api -n ml-inference
```

### Autoscaling not working

```bash
# Check HPA status
kubectl get hpa -n ml-inference

# Describe HPA
kubectl describe hpa inference-api-hpa -n ml-inference
```

## Cleanup

```bash
# Remove all resources
kubectl delete -k infrastructure/kubernetes/inference/
kubectl delete -k infrastructure/kubernetes/base/

# Or delete namespace (removes everything)
kubectl delete namespace ml-inference
```
