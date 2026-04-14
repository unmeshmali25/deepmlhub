FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl gcc libpq-dev && rm -rf /var/lib/apt/lists/*

# Security: non-root user
RUN groupadd -r mluser && useradd -r -g mluser mluser

WORKDIR /app

# Copy requirements first (caching optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY configs/ configs/
COPY feast/ feast/
COPY dvc.yaml ./
COPY .dvc/ .dvc/

# Create directories
RUN mkdir -p data/raw data/processed models metrics mlruns

# Set ownership and switch user
RUN chown -R mluser:mluser /app
USER mluser

# Default command
CMD ["python", "-m", "src.model.train"]
