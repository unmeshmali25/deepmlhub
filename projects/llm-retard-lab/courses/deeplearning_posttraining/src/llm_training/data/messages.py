from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    from datasets import Dataset


def to_message(example: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Convert {prompt, response} -> {message: [user, assistant]} for TRL.
    Pure: no I/O, no globals, deterministic
    """
    return {
        "messages": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["response"]},
        ]
    }


def parse_pred(text: str) -> dict[str, Optional[str]]:
    """
    Slit a model generation into {thinking, reply}.
    Input format: "<thinking> ... </thinking>\\n\\n<reply text>"
    if no <thinking> block, thinking=None and reply=text.strip().

    Pure: regex only, no I/O.
    """
    pattern = r"<thinking>(.*?)</thinking>\s*(.*)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return {"thinking": None, "reply": text.strip()}
    return {
        "thinking": match.group(1).strip(),
        "reply": match.group(2).strip(),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """
    Read a .jsonl file into a list of dicts. Raises FileNotFoundError if missing
    Pure-ish: deterministic I/O, no globals, easy to test with tmp_path
    """
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_num}: invalid JSON: {e}") from e
    return records


def to_sft_dataset(
    path: Path,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Tuple["Dataset", "Dataset"]:
    """
    Load jsonl, map to messages, Split into train val hf datasets.

    Impure: depends on HF datasets library. Integration tests only.
    """
    from datasets import load_dataset

    ds = load_dataset("json", data_files=str(path), split="train")
    ds = ds.map(to_message)
    split = ds.train_test_split(test_size=val_frac, seed=seed)
    return split["train"], split["test"]
