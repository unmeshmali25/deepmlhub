# llm-training

End-to-end LLM training pipeline following Karpathy's nanoGPT philosophy.

## Quick Start

```bash
# Install dependencies
make install

# Install in editable mode (recommended for development)
make install-dev

# Prepare Shakespeare data
python data/shakespeare/prepare.py

# Run pretraining
make pretrain

# Run SFT
make sft

# Run DPO (on M1 Pro - no GPU needed!)
make dpo

# Evaluate
make eval
```

## Project Structure

```
llm-training/
├── src/llm_training/      # Shared library (pip installable)
│   ├── models/gpt2/       # GPT-2 implementation
│   ├── training/          # Training loops (pretrain, SFT, DPO)
│   ├── data/              # Tokenizer, datasets
│   └── utils/             # Checkpointing, logging
├── courses/               # Course-specific sandboxes
│   ├── karpathy_zero2hero/
│   └── deeplearning_posttraining/
├── experiments/           # Your own cross-course research
├── configs/               # YAML configs
├── scripts/               # Docker, kubectl
├── tests/                 # Unit tests
├── Makefile               # Easy commands
├── setup.py               # Package installer
└── requirements.txt
```

## Using as a Library

Install the package to import it from course notebooks or experiments:

```bash
pip install -e .
```

```python
from llm_training.data.tokenizer import Tokenizer
from llm_training.models.gpt2 import GPT2Model, GPT2Config
from llm_training.training.dpo import DPOTrainer
```

## Hardware

- **Pretrain/SFT**: GKE with NVIDIA T4 (16GB)
- **DPO**: M1 Pro MacBook (zero cloud cost)

## Checkpoints

Saved to: `gs://deepmlhub-voiceoffers-dvc/checkpoints/llm-training`
