.PHONY: help install test lint format pretrain sft dpo eval clean

help:
	@echo "llm-training Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  install   - Install dependencies"
	@echo "  test      - Run tests"
	@echo "  lint      - Run ruff check"
	@echo "  format    - Format code with ruff"
	@echo "  pretrain  - Run pretraining"
	@echo "  sft       - Run SFT"
	@echo "  dpo       - Run DPO"
	@echo "  eval      - Run evaluation"
	@echo "  clean     - Clean artifacts"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

lint:
	ruff check src/

format:
	ruff format src/

pretrain:
	python -m src.training.pretrain --config configs/gpt2_124m_pretrain.yaml

sft:
	python -m src.training.sft --config configs/gpt2_124m_sft.yaml

dpo:
	python -m src.training.dpo --config configs/gpt2_124m_dpo.yaml

eval:
	python -m src.training.eval

clean:
	rm -rf __pycache__ src/**/__pycache__ .pytest_cache
	rm -rf outputs/ logs/ *.pt
