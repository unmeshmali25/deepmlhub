# Training Dockerfile for ML Pipeline
# Builds containerized training environment with DVC and MLflow support

FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r mluser && useradd -r -g mluser mluser

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY configs/ configs/
COPY dvc.yaml ./
COPY .dvc/ .dvc/

# Create necessary directories
RUN mkdir -p data/raw data/processed models metrics mlruns

# Set ownership
RUN chown -R mluser:mluser /app

# Switch to non-root user
USER mluser

# Default command runs the training pipeline
CMD ["python", "-m", "src.model.train"]
