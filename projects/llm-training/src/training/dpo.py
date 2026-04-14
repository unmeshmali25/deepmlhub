import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import DPODataset
from src.data.tokenizer import Tokenizer
from src.models.gpt2.config import GPT2Config
from src.models.gpt2.model import GPT2Model
from src.utils.checkpoint import CheckpointManager, MLflowLogger
from src.utils.device import get_default_device


def dpo_loss(
    policy_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
    beta: float = 0.3,
) -> torch.Tensor:
    chosen_logps = policy_logps[: len(policy_logps) // 2]
    rejected_logps = policy_logps[len(policy_logps) // 2 :]

    chosen_ref_logps = ref_logps[: len(ref_logps) // 2]
    rejected_ref_logps = ref_logps[len(ref_logps) // 2 :]

    chosen_logratios = chosen_logps - chosen_ref_logps
    rejected_logratios = rejected_logps - rejected_ref_logps

    logratios = chosen_logratios - rejected_logratios
    loss = -F.logsigmoid(beta * logratios).mean()

    return loss


def compute_logps(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    return -F.cross_entropy(
        logits[:, :-1].transpose(1, 2), input_ids[:, 1:], reduction="none"
    ).sum(dim=-1)


def train(
    model: nn.Module,
    ref_model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: dict,
    logger: MLflowLogger,
) -> None:
    device = get_default_device()
    model.train()

    max_iters = config["training"]["max_iters"]
    beta = config["training"]["beta"]
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
        chosen = batch["chosen"].to(device)
        rejected = batch["rejected"].to(device)

        input_ids = torch.cat([chosen, rejected], dim=0)

        logits, _ = model(input_ids)
        with torch.no_grad():
            ref_logits, _ = ref_model(input_ids)

        policy_logps = compute_logps(logits, input_ids)
        ref_logps = compute_logps(ref_logits, input_ids)

        loss = dpo_loss(policy_logps, ref_logps, None, None, beta=beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t1 = time.time()
        dt = t1 - t0

        if step % log_interval == 0:
            print(f"Step {step:5d} | loss: {loss.item():.4f} | dt: {dt * 1000:.2f}ms")
            logger.log_metrics({"dpo/loss": loss.item()}, step=step)

        if step > 0 and step % config["logging"]["save_interval"] == 0:
            checkpoint_manager.save_best(
                model, optimizer.state_dict(), step, loss.item()
            )

        step += 1

    checkpoint_manager.save_best(model, optimizer.state_dict(), step, loss.item())
    print("DPO training complete!")


def main(config_path: str = "configs/gpt2_124m_dpo.yaml") -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = get_default_device()
    print(f"Using device: {device}")

    checkpoint_path = config["model"].get("checkpoint_path")
    print(f"Loading SFT checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_config = GPT2Config.gpt2_124m()
    model = GPT2Model.from_config(model_config)
    model.load_state_dict(checkpoint["model_state"])

    ref_model = GPT2Model.from_config(model_config)
    ref_model.load_state_dict(checkpoint["model_state"])
    ref_model.eval()

    model.to(device)
    ref_model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    tokenizer = Tokenizer("gpt2")
    train_path = Path(config["data"]["train_path"])
    train_dataset = DPODataset(
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
        weight_decay=0.1,
    )

    logger = MLflowLogger(
        experiment_name="llm-training-dpo",
        tracking_uri=config["logging"].get("mlflow_tracking_uri"),
    )
    logger.log_params({"config": config})

    train(model, ref_model, train_loader, optimizer, config, logger)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gpt2_124m_dpo.yaml")
    args = parser.parse_args()

    main(args.config)
