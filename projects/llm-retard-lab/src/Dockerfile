# RunPon cold-start CoT SFT training image
# Build context - this directory (courses/deeplearning_posttraining/src/)
# Using RunPod's PyTorch 2.8.0 + CUDA 12.8.1 base image template
# This image ships correct CUDA build of torch, plus nginx/openssh/jupyterlab and RunPod's startup script. 
# docker build -t runpod-cot-sft .

FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2204

WORKDIR /workspace

# Minimal system deps (git lets you pull code inside the pod). 

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/* 

# Training stack only - torch intentionally absent (already in base image)
RUN pip install --no-cache-dir \ 
  transformers \
  "trl >= 0.11" \
  peft \
  accelerate \
  datasets \
  matplotlib

# The src module hosting the codebase
COPY . /workspace

# Making the src package importable from the notebook / CLI
ENV PYTHONPATH=/workspace 

# Headless run (once the training module exists): 
#   docker run --gpus all runpod-cot-sft python -m <pkg>.training.sft_cot \ 
#     --data /workspace/..data.jsonl --output /workspace/adapter_qwen_cot_full
#
# Defaut CMD intentially not overridder: the RunPod base image starts jupyterlab + openssh
# so the pod gives us an interactive session to upload data.jsonl + notebook 



