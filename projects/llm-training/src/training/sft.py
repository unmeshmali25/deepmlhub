import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import SFTDataset
from src.data.tokenizer import Tokenizer
from src.models.gpt2.config import GPT2Config
from src.models.gpt2.model import GPT2Model
from src.utils.checkpoint import CheckpointManager, MLflowLogger
from src.utils.device import get_default_device


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: dict,
    logger: MLflowLogger,
) -> None:
    device = get_default_device()
    model.train()

    max_iters = config["training"]["max_iters"]
    grad_clip = config["training"]["grad_clip"]
    log_interval = config["logging"]["log_interval"]

    checkpoint_manager = CheckpointManager(
        save_dir=config["checkpoint"]["save_dir"],
        GCS_bucket=config["checkpoint"].get("GCS_bucket"),
    )

    step = 0
    train_iter = iter(train_loader)

    while step < max_iters:
        t0 = time.time()

        batch = next(train_iter)
        x, y = [b.to(device) for b in batch]

        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        t1 = time.time()
        dt = t1 - t0

        if step % log_interval == 0:
            print(f"Step {step:5d} | loss: {loss.item():.4f} | dt: {dt * 1000:.2f}ms")
            logger.log_metrics({"sft/loss": loss.item()}, step=step)

        if step > 0 and step % config["logging"]["save_interval"] == 0:
            checkpoint_manager.save_best(
                model, optimizer.state_dict(), step, loss.item()
            )

        step += 1

    checkpoint_manager.save_best(model, optimizer.state_dict(), step, loss.item())
    print("SFT training complete!")


def main(config_path: str = "configs/gpt2_124m_sft.yaml") -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = get_default_device()
    print(f"Using device: {device}")

    checkpoint_path = config["model"].get("checkpoint_path")
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_config = GPT2Config.gpt2_124m()
        model = GPT2Model.from_config(model_config)
        model.load_state_dict(checkpoint["model_state"])
    else:
        print("Starting from scratch with GPT-2 124M")
        model_config = GPT2Config.gpt2_124m()
        model = GPT2Model.from_config(model_config)

    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    tokenizer = Tokenizer("gpt2")
    train_path = Path(config["data"]["train_path"])
    train_dataset = SFTDataset(
        train_path, tokenizer, block_size=config["training"]["sequence_length"]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=(config["training"]["beta1"], config["training"]["beta2"]),
    )

    logger = MLflowLogger(
        experiment_name="llm-training-sft",
        tracking_uri=config["logging"].get("mlflow_tracking_uri"),
    )
    logger.log_params({"config": config})

    train(model, train_loader, optimizer, config, logger)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gpt2_124m_sft.yaml")
    args = parser.parse_args()

    main(args.config)
