# Dev Environment - Variables

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

# GCS Module Variables
variable "lifecycle_delete_days" {
  description = "Days after which old object versions are deleted"
  type        = number
  default     = 30
}

# Artifact Registry Variables
variable "artifact_repository_id" {
  description = "Artifact Registry repository ID"
  type        = string
  default     = "ml-images"
}

# MLflow Variables
variable "mlflow_service_name" {
  description = "Name of the MLflow Cloud Run service"
  type        = string
  default     = "mlflow-server"
}

variable "mlflow_image" {
  description = "MLflow Docker image to deploy"
  type        = string
  default     = "gcr.io/cloudrun/hello"
}

variable "mlflow_min_instances" {
  description = "Minimum MLflow instances (0 for scale-to-zero)"
  type        = string
  default     = "0"
}

variable "mlflow_max_instances" {
  description = "Maximum MLflow instances"
  type        = string
  default     = "2"
}

# GKE Module Variables
variable "gke_cluster_name" {
  description = "Name of the GKE cluster"
  type        = string
  default     = "deepmlhub-cluster"
}

variable "gke_enable_private_nodes" {
  description = "Enable private nodes for GKE (recommended for production)"
  type        = bool
  default     = false
}
