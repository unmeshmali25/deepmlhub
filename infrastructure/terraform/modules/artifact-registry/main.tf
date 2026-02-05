# Artifact Registry Module - Creates Docker repository for ML images

resource "google_artifact_registry_repository" "docker_repo" {
  location      = var.region
  repository_id = var.repository_id
  description   = "Docker repository for ML training and inference images"
  format        = "DOCKER"

  labels = merge(var.labels, {
    purpose = "ml-docker-images"
  })
}
