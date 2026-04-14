#!/bin/bash
set -e

CONFIG=${1:-configs/gpt2_124m_pretrain.yaml}
JOB_NAME=${2:-llm-pretrain}

kubectl exec -it deploy/${JOB_NAME} -- python -m src.training.pretrain --config ${CONFIG}
