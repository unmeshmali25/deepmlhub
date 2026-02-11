# GKE Module - Creates Kubernetes cluster for ML workloads
# Using GKE Autopilot for simplified management

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# Enable required APIs
resource "google_project_service" "container" {
  service            = "container.googleapis.com"
  disable_on_destroy = false
}

# GKE Autopilot Cluster
resource "google_container_cluster" "primary" {
  name     = var.cluster_name
  location = var.region
  project  = var.project_id

  enable_autopilot = true

  network    = var.network_id
  subnetwork = var.subnetwork_id

  # Release channel
  release_channel {
    channel = "REGULAR"
  }

  # Network configuration
  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  # Private cluster configuration (recommended for production)
  private_cluster_config {
    enable_private_nodes    = var.enable_private_nodes
    enable_private_endpoint = false
    master_ipv4_cidr_block  = var.enable_private_nodes ? "172.16.0.0/28" : null
  }

  # Master authorized networks (allow access from anywhere for dev)
  master_authorized_networks_config {
    cidr_blocks {
      cidr_block   = "0.0.0.0/0"
      display_name = "All"
    }
  }

  # Maintenance window
  maintenance_policy {
    recurring_window {
      start_time = "2024-01-01T06:00:00Z"
      end_time   = "2024-01-01T12:00:00Z"
      recurrence = "FREQ=WEEKLY;BYDAY=SA,SU"
    }
  }

  # Logging and monitoring
  logging_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]
  }

  monitoring_config {
    enable_components = ["SYSTEM_COMPONENTS"]
    managed_prometheus {
      enabled = true
    }
  }

  resource_labels = var.labels

  depends_on = [google_project_service.container]
}

# Create a node pool for ML workloads (optional with Autopilot, but useful for specific configurations)
# Note: With Autopilot, node pools are managed automatically

# Output the cluster endpoint and credentials
output "cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.primary.name
}

output "cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = google_container_cluster.primary.endpoint
}

output "cluster_location" {
  description = "GKE cluster location"
  value       = google_container_cluster.primary.location
}

output "cluster_ca_certificate" {
  description = "GKE cluster CA certificate"
  value       = base64decode(google_container_cluster.primary.master_auth[0].cluster_ca_certificate)
  sensitive   = true
}
