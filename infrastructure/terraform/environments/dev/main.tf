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

# Cloud SQL Module - Creates PostgreSQL database for MLflow
module "cloud_sql" {
  source = "../../modules/cloud-sql"

  project_id        = var.project_id
  region            = var.region
  instance_name     = "mlflow-db"
  database_password = var.mlflow_db_password
  authorized_ip     = var.authorized_ip
  labels            = local.common_labels
}

# MLflow Cloud Run Module - Deploys MLflow tracking server
module "mlflow" {
  source = "../../modules/mlflow"

  project_id                         = var.project_id
  region                             = var.region
  service_name                       = var.mlflow_service_name
  mlflow_image                       = var.mlflow_image
  artifacts_bucket_name              = module.gcs.mlflow_bucket_name
  cloud_sql_instance_connection_name = module.cloud_sql.instance_connection_name
  database_user                      = "mlflow"
  database_password                  = var.mlflow_db_password
  min_instances                      = var.mlflow_min_instances
  max_instances                      = var.mlflow_max_instances
  labels                             = local.common_labels
}

# VPC Module - Creates network infrastructure for GKE
module "vpc" {
  source = "../../modules/vpc"

  project_id   = var.project_id
  region       = var.region
  network_name = "deepmlhub-vpc"
  subnet_name  = "deepmlhub-subnet"
}

# GKE Module - Creates Kubernetes cluster for ML workloads
# Note: GKE Autopilot clusters have a control plane cost (~$70/month)
# but no node management overhead. Nodes scale to zero when not in use.
module "gke" {
  source = "../../modules/gke"

  project_id           = var.project_id
  region               = var.region
  cluster_name         = var.gke_cluster_name
  enable_private_nodes = var.gke_enable_private_nodes
  labels               = local.common_labels
  network_id           = module.vpc.network_id
  subnetwork_id        = module.vpc.subnetwork_id
}
