# Dev Environment - Main Configuration
# This file calls all the modules to create the dev infrastructure

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Common labels for all resources
locals {
  common_labels = {
    environment = var.environment
    managed_by  = "terraform"
    project     = "deepmlhub"
  }
}

# GCS Module - Creates DVC and MLflow buckets
module "gcs" {
  source = "../../modules/gcs"

  project_id            = var.project_id
  region                = var.region
  lifecycle_delete_days = var.lifecycle_delete_days
  labels                = local.common_labels
}

# Artifact Registry Module - Creates Docker repository
module "artifact_registry" {
  source = "../../modules/artifact-registry"

  project_id    = var.project_id
  region        = var.region
  repository_id = var.artifact_repository_id
  labels        = local.common_labels
}

# MLflow Cloud Run Module - Deploys MLflow tracking server
module "mlflow" {
  source = "../../modules/mlflow"

  project_id            = var.project_id
  region                = var.region
  service_name          = var.mlflow_service_name
  mlflow_image          = var.mlflow_image
  artifacts_bucket_name = module.gcs.mlflow_bucket_name
  min_instances         = var.mlflow_min_instances
  max_instances         = var.mlflow_max_instances
  labels                = local.common_labels
}
