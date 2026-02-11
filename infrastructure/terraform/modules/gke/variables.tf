# GKE Module Variables

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for the cluster"
  type        = string
  default     = "us-central1"
}

variable "cluster_name" {
  description = "Name of the GKE cluster"
  type        = string
  default     = "deepmlhub-cluster"
}

variable "enable_private_nodes" {
  description = "Enable private nodes (recommended for production)"
  type        = bool
  default     = false
}

variable "labels" {
  description = "Labels to apply to resources"
  type        = map(string)
  default     = {}
}

variable "network_id" {
  description = "The VPC network ID"
  type        = string
}

variable "subnetwork_id" {
  description = "The subnetwork ID"
  type        = string
}
