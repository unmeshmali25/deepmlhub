# VPC Module Variables

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "network_name" {
  description = "Name of the VPC network"
  type        = string
  default     = "deepmlhub-vpc"
}

variable "subnet_name" {
  description = "Name of the subnet"
  type        = string
  default     = "deepmlhub-subnet"
}

variable "subnet_cidr" {
  description = "Primary CIDR range for the subnet"
  type        = string
  default     = "10.0.0.0/24"
}

variable "pods_cidr" {
  description = "Secondary CIDR range for pods"
  type        = string
  default     = "10.4.0.0/14"
}

variable "services_cidr" {
  description = "Secondary CIDR range for services"
  type        = string
  default     = "10.0.32.0/20"
}
