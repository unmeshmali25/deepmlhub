output "repository_id" {
  description = "Artifact Registry repository ID"
  value       = google_artifact_registry_repository.docker_repo.repository_id
}

output "repository_url" {
  description = "Full repository URL for Docker images"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker_repo.repository_id}"
}

output "repository_name" {
  description = "Repository name (same as repository_id)"
  value       = google_artifact_registry_repository.docker_repo.name
}
