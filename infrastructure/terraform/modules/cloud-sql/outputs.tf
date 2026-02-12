output "instance_name" {
  description = "Name of the Cloud SQL instance"
  value       = google_sql_database_instance.mlflow.name
}

output "instance_connection_name" {
  description = "Connection name for the Cloud SQL instance"
  value       = google_sql_database_instance.mlflow.connection_name
}

output "public_ip_address" {
  description = "Public IP address of the Cloud SQL instance"
  value       = google_sql_database_instance.mlflow.public_ip_address
}

output "database_name" {
  description = "Name of the created database"
  value       = google_sql_database.mlflow.name
}

output "database_user" {
  description = "Database username"
  value       = google_sql_user.mlflow.name
}

output "connection_string" {
  description = "PostgreSQL connection string for MLflow"
  value       = "postgresql://${var.database_user}:${var.database_password}@${google_sql_database_instance.mlflow.public_ip_address}:5432/${var.database_name}"
  sensitive   = true
}
