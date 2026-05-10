.PHONY: help install install-dev test lint format pretrain sft dpo eval clean

help:
	@echo "llm-retard-lab Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  install     - Install dependencies"
	@echo "  install-dev - Install package in editable mode with dev deps"
	@echo "  test        - Run tests"
	@echo "  lint        - Run ruff check"
	@echo "  format      - Format code with ruff"
	@echo "  pretrain    - Run pretraining"
	@echo "  sft         - Run SFT"
	@echo "  dpo         - Run DPO"
	@echo "  eval        - Run evaluation"
	@echo "  clean       - Clean artifacts"

install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check src/llm_training/

format:
	ruff format src/llm_training/

pretrain:
	PYTHONPATH=src python -m llm_training.training.pretrain --config configs/gpt2_124m_pretrain.yaml

sft:
	PYTHONPATH=src python -m llm_training.training.sft --config configs/gpt2_124m_sft.yaml

dpo:
	PYTHONPATH=src python -m llm_training.training.dpo --config configs/gpt2_124m_dpo.yaml

eval:
	PYTHONPATH=src python -m llm_training.training.eval

clean:
	rm -rf __pycache__ src/**/__pycache__ .pytest_cache
	rm -rf outputs/ logs/ *.pt
