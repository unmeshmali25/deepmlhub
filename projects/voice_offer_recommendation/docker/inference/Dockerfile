FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends curl libpq-dev

# Non-root user
RUN groupadd -r mluser && useradd -r -g mluser mluser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source, config, Feast repo, AND pre-trained model
COPY src/ src/
COPY configs/ configs/
COPY feast/ feast/
COPY models/ models/

RUN chown -R mluser:mluser /app
USER mluser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.inference.server:app", "--host", "0.0.0.0", "--port", "8000"]
