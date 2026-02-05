# MLflow Cloud Run Module - Deploys MLflow server on Cloud Run

# Service account for MLflow
resource "google_service_account" "mlflow_sa" {
  account_id   = "mlflow-server"
  display_name = "MLflow Server Service Account"
  description  = "Service account for MLflow Cloud Run service"
}

# Grant MLflow service account access to GCS bucket
resource "google_storage_bucket_iam_member" "mlflow_gcs_access" {
  bucket = var.artifacts_bucket_name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.mlflow_sa.email}"
}

# Cloud Run service for MLflow
resource "google_cloud_run_service" "mlflow" {
  name     = var.service_name
  location = var.region

  template {
    spec {
      service_account_name = google_service_account.mlflow_sa.email
      
      containers {
        image = var.mlflow_image
        
        ports {
          container_port = 5000
        }
        
        env {
          name  = "BACKEND_STORE_URI"
          value = "sqlite:///tmp/mlflow.db"
        }
        
        env {
          name  = "DEFAULT_ARTIFACT_ROOT"
          value = "gs://${var.artifacts_bucket_name}"
        }
        
        env {
          name  = "MLFLOW_TRACKING_URI"
          value = ""
        }
        
        resources {
          limits = {
            cpu    = "1"
            memory = "512Mi"
          }
        }
      }
      
      container_concurrency = 80
      timeout_seconds       = 300
    }
    
    metadata {
      annotations = {
        "autoscaling.knative.dev/minScale" = var.min_instances
        "autoscaling.knative.dev/maxScale" = var.max_instances
        "run.googleapis.com/cpu-throttling" = var.min_instances == "0" ? "true" : "false"
      }
      
      labels = merge(var.labels, {
        purpose = "mlflow-tracking"
      })
    }
  }
  
  depends_on = [google_storage_bucket_iam_member.mlflow_gcs_access]
}

# Allow unauthenticated invocations (for dev/testing)
# In production, you might want to restrict this
resource "google_cloud_run_service_iam_member" "invoker" {
  location = google_cloud_run_service.mlflow.location
  project  = google_cloud_run_service.mlflow.project
  service  = google_cloud_run_service.mlflow.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
