variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region for bucket creation"
  type        = string
  default     = "us-central1"
}

variable "lifecycle_delete_days" {
  description = "Days after which old object versions are deleted"
  type        = number
  default     = 30
}

variable "labels" {
  description = "Labels to apply to buckets"
  type        = map(string)
  default     = {}
}
