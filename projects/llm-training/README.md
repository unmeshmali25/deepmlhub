# llm-training

End-to-end LLM training pipeline following Karpathy's nanoGPT philosophy.

## Quick Start

```bash
# Install dependencies
make install

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
├── src/
│   ├── models/gpt2/      # GPT-2 implementation
│   ├── training/         # Training loops (pretrain, SFT, DPO)
│   ├── data/            # Tokenizer, datasets
│   └── utils/           # Checkpointing, logging
├── configs/             # YAML configs
├── scripts/             # Docker, kubectl
└── tests/               # Unit tests
```

## Hardware

- **Pretrain/SFT**: GKE with NVIDIA T4 (16GB)
- **DPO**: M1 Pro MacBook (zero cloud cost)

## Checkpoints

Saved to: `gs://deepmlhub-voiceoffers-dvc/checkpoints/llm-training`
