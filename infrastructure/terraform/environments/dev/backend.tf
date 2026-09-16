# Terraform Backend Configuration
# Copy this file to backend.tf and fill in your state bucket name

terraform {
  backend "gcs" {
    bucket = "deepmlhub-deepmlhub-voiceoffers-tfstate"
    prefix = "terraform/state/dev"
  }
}
