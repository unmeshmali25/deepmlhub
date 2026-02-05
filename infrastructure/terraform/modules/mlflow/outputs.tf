output "service_name" {
  description = "Name of the Cloud Run service"
  value       = google_cloud_run_service.mlflow.name
}

output "service_url" {
  description = "URL of the deployed MLflow server"
  value       = google_cloud_run_service.mlflow.status[0].url
}

output "service_account_email" {
  description = "Email of the MLflow service account"
  value       = google_service_account.mlflow_sa.email
}

output "artifacts_bucket" {
  description = "GCS bucket used for MLflow artifacts"
  value       = var.artifacts_bucket_name
}
