from pathlib import Path

import torch
from torch.utils.data import Dataset


class ShakespeareDataset(Dataset):
    def __init__(self, data_path: Path, block_size: int = 128):
        self.data = torch.fromfile(data_path, dtype=torch.uint32)
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.block_size].long()
        y = self.data[idx + 1 : idx + self.block_size + 1].long()
        return x, y


class SFTDataset(Dataset):
    def __init__(self, jsonl_path: Path, tokenizer, block_size: int = 512):
        self.examples = []
        with open(jsonl_path) as f:
            for line in f:
                obj = eval(line)
                text = obj["prompt"] + obj["completion"]
                self.examples.append(tokenizer.encode(text))

        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.examples[idx]
        enc = enc[: self.block_size + 1]
        x = torch.tensor(enc[:-1], dtype=torch.long)
        y = torch.tensor(enc[1:], dtype=torch.long)
        return x, y


class DPODataset(Dataset):
    def __init__(self, jsonl_path: Path, tokenizer, block_size: int = 512):
        self.examples = []
        with open(jsonl_path) as f:
            for line in f:
                obj = eval(line)
                chosen = tokenizer.encode(obj["chosen"])
                rejected = tokenizer.encode(obj["rejected"])
                self.examples.append({"chosen": chosen, "rejected": rejected})

        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        item = self.examples[idx]
        chosen = item["chosen"][: self.block_size]
        rejected = item["rejected"][: self.block_size]
        return {
            "chosen": torch.tensor(chosen, dtype=torch.long),
            "rejected": torch.tensor(rejected, dtype=torch.long),
        }
