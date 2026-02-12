variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region for Cloud Run"
  type        = string
  default     = "us-central1"
}

variable "service_name" {
  description = "Name of the Cloud Run service"
  type        = string
  default     = "mlflow-server"
}

variable "mlflow_image" {
  description = "MLflow Docker image to deploy"
  type        = string
  default     = "gcr.io/cloudrun/hello" # Placeholder - replace with actual MLflow image
}

variable "artifacts_bucket_name" {
  description = "GCS bucket name for MLflow artifacts"
  type        = string
}

variable "min_instances" {
  description = "Minimum number of instances (0 for scale-to-zero)"
  type        = string
  default     = "0"
}

variable "max_instances" {
  description = "Maximum number of instances"
  type        = string
  default     = "2"
}

variable "labels" {
  description = "Labels to apply to Cloud Run service"
  type        = map(string)
  default     = {}
}

variable "database_url" {
  description = "PostgreSQL connection string for MLflow backend store"
  type        = string
  sensitive   = true
}
