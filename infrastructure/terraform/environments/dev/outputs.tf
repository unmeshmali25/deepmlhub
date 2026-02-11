# Dev Environment - Outputs
# These values are shown after terraform apply

# GCS Bucket Outputs
output "dvc_bucket_url" {
  description = "GCS URL for DVC storage bucket"
  value       = module.gcs.dvc_bucket_url
}

output "dvc_bucket_name" {
  description = "Name of the DVC storage bucket"
  value       = module.gcs.dvc_bucket_name
}

output "mlflow_artifacts_bucket_url" {
  description = "GCS URL for MLflow artifacts bucket"
  value       = module.gcs.mlflow_bucket_url
}

output "mlflow_artifacts_bucket_name" {
  description = "Name of the MLflow artifacts bucket"
  value       = module.gcs.mlflow_bucket_name
}

# Artifact Registry Outputs
output "docker_repository_url" {
  description = "URL for pushing/pulling Docker images"
  value       = module.artifact_registry.repository_url
}

output "artifact_repository_id" {
  description = "Artifact Registry repository ID"
  value       = module.artifact_registry.repository_id
}

# MLflow Outputs
output "mlflow_tracking_url" {
  description = "URL for MLflow tracking server"
  value       = module.mlflow.service_url
}

output "mlflow_service_name" {
  description = "Name of the MLflow Cloud Run service"
  value       = module.mlflow.service_name
}

output "mlflow_service_account" {
  description = "Service account email for MLflow"
  value       = module.mlflow.service_account_email
}

# VPC Outputs
output "vpc_network_id" {
  description = "VPC network ID"
  value       = module.vpc.network_id
}

output "vpc_network_name" {
  description = "VPC network name"
  value       = module.vpc.network_name
}

output "vpc_subnetwork_id" {
  description = "VPC subnetwork ID"
  value       = module.vpc.subnetwork_id
}

output "vpc_subnetwork_name" {
  description = "VPC subnetwork name"
  value       = module.vpc.subnetwork_name
}

# GKE Outputs
output "gke_cluster_name" {
  description = "GKE cluster name"
  value       = module.gke.cluster_name
}

output "gke_cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = module.gke.cluster_endpoint
  sensitive   = true
}

# Summary Output
output "infrastructure_summary" {
  description = "Summary of created infrastructure"
  value       = <<EOT

🎉 Dev Infrastructure Created Successfully!

📦 Storage:
   - DVC Bucket: ${module.gcs.dvc_bucket_name}
   - MLflow Artifacts: ${module.gcs.mlflow_bucket_name}

🐳 Docker Registry:
   - Repository: ${module.artifact_registry.repository_url}

📊 MLflow Tracking:
   - URL: ${module.mlflow.service_url}
   - Service Account: ${module.mlflow.service_account_email}

🌐 Network:
   - VPC: ${module.vpc.network_name}
   - Subnet: ${module.vpc.subnetwork_name}

☸️  GKE Cluster:
   - Name: ${module.gke.cluster_name}
   - Endpoint: ${module.gke.cluster_endpoint}

📝 Next Steps:
   1. Configure DVC remote: dvc remote add -d gcs gs://${module.gcs.dvc_bucket_name}
   2. Authenticate Docker: gcloud auth configure-docker ${var.region}-docker.pkg.dev
   3. Set MLflow tracking URI: export MLFLOW_TRACKING_URI=${module.mlflow.service_url}
   4. Configure kubectl: gcloud container clusters get-credentials ${module.gke.cluster_name} --region ${var.region}
EOT
}
