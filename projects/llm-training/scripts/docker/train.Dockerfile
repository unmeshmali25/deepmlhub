FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

CMD ["python", "-m", "src.training.pretrain", "--config", "configs/gpt2_124m_pretrain.yaml"]
