output "dvc_bucket_name" {
  description = "Name of the DVC storage bucket"
  value       = google_storage_bucket.dvc_storage.name
}

output "dvc_bucket_url" {
  description = "URL of the DVC storage bucket (gs://)"
  value       = "gs://${google_storage_bucket.dvc_storage.name}"
}

output "mlflow_bucket_name" {
  description = "Name of the MLflow artifacts bucket"
  value       = google_storage_bucket.mlflow_artifacts.name
}

output "mlflow_bucket_url" {
  description = "URL of the MLflow artifacts bucket (gs://)"
  value       = "gs://${google_storage_bucket.mlflow_artifacts.name}"
}
