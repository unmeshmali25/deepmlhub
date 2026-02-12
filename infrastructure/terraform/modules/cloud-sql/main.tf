# Cloud SQL PostgreSQL Module for MLflow
# Creates a minimal PostgreSQL instance for tracking ML experiments

# Cloud SQL PostgreSQL instance
resource "google_sql_database_instance" "mlflow" {
  name             = var.instance_name
  database_version = "POSTGRES_15"
  region           = var.region
  project          = var.project_id

  settings {
    tier = var.tier # db-f1-micro for dev (cheapest)

    ip_configuration {
      ipv4_enabled = true

      # Restrict access to user's IP only
      authorized_networks {
        name  = "developer-ip"
        value = var.authorized_ip
      }
    }

    backup_configuration {
      enabled = var.enable_backups
    }

    disk_autoresize = var.disk_autoresize
    disk_size       = var.disk_size
    disk_type       = var.disk_type

    user_labels = var.labels
  }

  deletion_protection = var.deletion_protection
}

# Create mlflow database
resource "google_sql_database" "mlflow" {
  name     = var.database_name
  instance = google_sql_database_instance.mlflow.name
  project  = var.project_id
}

# Create mlflow user
resource "google_sql_user" "mlflow" {
  name     = var.database_user
  instance = google_sql_database_instance.mlflow.name
  password = var.database_password
  project  = var.project_id
}
