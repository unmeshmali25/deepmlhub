from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import yaml


@dataclass
class TrainConfig:
    """
    All knobs for a single LoRA SFT run.
    """

    model_name: str = "Qwen/Qwen2.5-1.5B"
    data_path: Path = Path("data.jsonl")
    output_dir: Path = Path("./adapter_out")

    # --- LoRA ---
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Tuple (not list) so the default is hashable — safe as a dataclass default.
    lora_target_modules: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    # --- SFT loop ---
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_length: int = 1024
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 200
    save_total_limit: int = 2
    # None = use epochs; set to int (e.g. 2) for fast integration tests.
    max_steps: Optional[int] = None

    # --- dtype ---
    fp16: bool = False
    bf16: bool = True  # A5000 supports bf16 — preferred over fp16

    # --- reproducibility + integrations ---
    seed: int = 42
    report_to: str = "none"


def load_train_config(config_path: Path) -> TrainConfig:
    """Load a training configuration from YAML."""

    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    project_root = config_path.parent.parent

    return TrainConfig(
        model_name=raw["model"]["name"],
        data_path=project_root / raw["data"]["path"],
        output_dir=project_root / raw["output"]["directory"],
        lora_r=raw["peft"]["r"],
        lora_alpha=raw["peft"]["alpha"],
        lora_dropout=raw["peft"]["dropout"],
        lora_target_modules=tuple(raw["peft"]["target_modules"]),
        num_train_epochs=raw["training"]["num_train_epochs"],
        per_device_train_batch_size=raw["training"]["batch_size"],
        gradient_accumulation_steps=raw["training"]["gradient_accumulation_steps"],
        learning_rate=raw["training"]["learning_rate"],
        max_length=raw["training"]["max_length"],
        seed=raw["training"]["seed"],
    )
