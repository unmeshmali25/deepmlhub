import urllib.request
from pathlib import Path

import tiktoken


def download_shakespeare() -> Path:
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_dir = Path("data/shakespeare")
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / "input.txt"

    if dest.exists():
        print(f"Already downloaded to {dest}")
        return dest

    print(f"Downloading Shakespeare from {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest}")
    return dest


def tokenize_shakespeare(
    input_path: Path, output_dir: Path, train_ratio: float = 0.9
) -> None:
    with open(input_path) as f:
        text = f.read()

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(text)
    tokens = [int(t) for t in tokens]

    n = len(tokens)
    split = int(n * train_ratio)
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]

    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"

    import array as arr

    train_arr = arr.array("I", train_tokens)
    val_arr = arr.array("I", val_tokens)

    train_arr.tofile(open(train_path, "wb"))
    val_arr.tofile(open(val_path, "wb"))

    print(f"Train tokens: {len(train_tokens)}")
    print(f"Val tokens: {len(val_tokens)}")
    print(f"Train saved to {train_path}")
    print(f"Val saved to {val_path}")


def prepare_shakespeare_data() -> None:
    data_dir = Path("data/shakespeare")
    input_path = download_shakespeare()
    tokenize_shakespeare(input_path, data_dir)


if __name__ == "__main__":
    prepare_shakespeare_data()
