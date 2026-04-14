import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import ShakespeareDataset
from src.models.gpt2.config import GPT2Config
from src.models.gpt2.model import GPT2Model
from src.utils.checkpoint import CheckpointManager, MLflowLogger
from src.utils.device import get_default_device


def get_lr(
    it: int, warmup_iters: int, lr_decay_iters: int, min_lr: float, learning_rate: float
) -> float:
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) * (
        learning_rate - min_lr
    )


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
    eval_interval = config["logging"]["eval_interval"]

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

        lr = get_lr(
            step,
            warmup_iters=config["training"]["warmup_iters"],
            lr_decay_iters=config["training"]["lr_decay_iters"],
            min_lr=config["training"]["min_lr"],
            learning_rate=config["training"]["learning_rate"],
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if step % log_interval == 0:
            tokens_per_sec = x.numel() / dt
            print(
                f"Step {step:5d} | loss: {loss.item():.4f} | lr: {lr:.2e} | dt: {dt * 1000:.2f}ms | tok/sec: {tokens_per_sec:.0f}"
            )
            logger.log_metrics(
                {
                    "train/loss": loss.item(),
                    "train/lr": lr,
                    "train/tokens_per_sec": tokens_per_sec,
                },
                step=step,
            )

        if step > 0 and step % eval_interval == 0:
            print(f"Eval at step {step}...")

        if step > 0 and step % config["logging"]["save_interval"] == 0:
            checkpoint_manager.save_best(
                model, optimizer.state_dict(), step, loss.item()
            )

        step += 1

    checkpoint_manager.save_best(model, optimizer.state_dict(), step, loss.item())
    print("Training complete!")


def main(config_path: str = "configs/gpt2_124m_pretrain.yaml") -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = get_default_device()
    print(f"Using device: {device}")

    model_config = GPT2Config.gpt2_124m()
    model = GPT2Model.from_config(model_config).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    train_path = Path(config["data"]["train_path"])
    train_dataset = ShakespeareDataset(
        train_path, block_size=config["training"]["sequence_length"]
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
        experiment_name="llm-training-pretrain",
        tracking_uri=config["logging"].get("mlflow_tracking_uri"),
    )
    logger.log_params({k: v for k, v in config.items()})
    logger.log_params({"model/params": sum(p.numel() for p in model.parameters())})

    train(model, train_loader, optimizer, config, logger)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gpt2_124m_pretrain.yaml")
    args = parser.parse_args()

    main(args.config)
