variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region for Artifact Registry"
  type        = string
  default     = "us-central1"
}

variable "repository_id" {
  description = "Repository ID/name"
  type        = string
  default     = "ml-images"
}

variable "labels" {
  description = "Labels to apply to repository"
  type        = map(string)
  default     = {}
}
