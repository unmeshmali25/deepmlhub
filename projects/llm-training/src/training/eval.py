from pathlib import Path

import torch
import yaml

from src.data.dataset import ShakespeareDataset
from src.data.tokenizer import Tokenizer
from src.models.gpt2.config import GPT2Config
from src.models.gpt2.model import GPT2Model
from src.utils.device import get_default_device


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = None,
) -> str:
    device = get_default_device()
    model.eval()

    input_ids = (
        torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    )

    for _ in range(max_new_tokens):
        if input_ids.size(1) > model.context_length:
            input_ids = input_ids[:, -model.context_length :]

        logits, _ = model(input_ids)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(input_ids[0].tolist())


def compute_perplexity(
    model: torch.nn.Module, dataset: ShakespeareDataset, device: torch.device
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i in range(min(100, len(dataset))):
        x, y = dataset[i]
        x = x.unsqueeze(0).to(device)
        y = y.to(device)

        _, loss = model(x, y)
        total_loss += loss.item()
        total_tokens += x.size(1)

    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


def main(
    checkpoint_path: str = "outputs/pretrain/best.pt",
    config_path: str = "configs/gpt2_124m_pretrain.yaml",
) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = get_default_device()
    print(f"Using device: {device}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_config = GPT2Config.gpt2_124m()
    model = GPT2Model.from_config(model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")

    tokenizer = Tokenizer("gpt2")

    val_path = Path(config["data"]["val_path"])
    val_dataset = ShakespeareDataset(
        val_path, block_size=config["training"]["sequence_length"]
    )
    perplexity = compute_perplexity(model, val_dataset, device)
    print(f"Validation perplexity: {perplexity:.4f}")

    prompts = [
        "ROMEO:",
        "JULIET:",
        "KING HENRY V:",
    ]

    print("\n--- Generated samples ---")
    for prompt in prompts:
        text = generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8)
        print(f"\n[{prompt}]\n{text}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/pretrain/best.pt")
    parser.add_argument("--config", type=str, default="configs/gpt2_124m_pretrain.yaml")
    args = parser.parse_args()

    main(args.checkpoint, args.config)
