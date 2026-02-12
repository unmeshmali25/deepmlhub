variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region for Cloud SQL"
  type        = string
  default     = "us-central1"
}

variable "instance_name" {
  description = "Name of the Cloud SQL instance"
  type        = string
  default     = "mlflow-db"
}

variable "database_name" {
  description = "Name of the database to create"
  type        = string
  default     = "mlflow"
}

variable "database_user" {
  description = "Username for the database"
  type        = string
  default     = "mlflow"
}

variable "database_password" {
  description = "Password for the database user"
  type        = string
  sensitive   = true
}

variable "authorized_ip" {
  description = "IP address to allow access to the database"
  type        = string
}

variable "tier" {
  description = "Machine type for the database (db-f1-micro is cheapest)"
  type        = string
  default     = "db-f1-micro"
}

variable "disk_size" {
  description = "Disk size in GB"
  type        = number
  default     = 10
}

variable "disk_type" {
  description = "Disk type (PD_SSD or PD_HDD)"
  type        = string
  default     = "PD_SSD"
}

variable "disk_autoresize" {
  description = "Enable automatic disk resize"
  type        = bool
  default     = false
}

variable "enable_backups" {
  description = "Enable automated backups"
  type        = bool
  default     = false
}

variable "deletion_protection" {
  description = "Enable deletion protection"
  type        = bool
  default     = false
}

variable "labels" {
  description = "Labels to apply to the instance"
  type        = map(string)
  default     = {}
}
