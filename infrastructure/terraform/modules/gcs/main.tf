# GCS Module - Creates buckets for DVC and MLflow

resource "google_storage_bucket" "dvc_storage" {
  name     = "${var.project_id}-dvc-storage"
  location = var.region

  uniform_bucket_level_access = true
  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = var.lifecycle_delete_days
    }
    action {
      type = "Delete"
    }
  }

  labels = merge(var.labels, {
    purpose = "dvc-storage"
  })
}

resource "google_storage_bucket" "mlflow_artifacts" {
  name     = "${var.project_id}-mlflow-artifacts"
  location = var.region

  uniform_bucket_level_access = true
  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = var.lifecycle_delete_days
    }
    action {
      type = "Delete"
    }
  }

  labels = merge(var.labels, {
    purpose = "mlflow-artifacts"
  })
}
