"""Generate synthetic classification data."""

from typing import cast
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_classification


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parents[2] / "configs" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_data() -> pd.DataFrame:
    """Generate synthetic classification data and save to CSV."""
    config = load_config()
    data_config = config["data"]
    paths = config["paths"]

    print(f"Generating {data_config['n_samples']} samples...")
    print(f"Features: {data_config['n_features']}, Classes: {data_config['n_classes']}")

    # Generate synthetic data
    X, y = make_classification(
        n_samples=data_config["n_samples"],
        n_features=data_config["n_features"],
        n_informative=max(2, data_config["n_features"] // 2),
        n_redundant=max(1, data_config["n_features"] // 4),
        n_classes=data_config["n_classes"],
        random_state=data_config["random_seed"],
        flip_y=0.1,  # Add some noise
    )
    X = cast(np.ndarray, X)
    y = cast(np.ndarray, y)

    # Create DataFrame
    feature_cols = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=pd.Index(feature_cols))
    df["target"] = y

    # Save to CSV
    output_path = Path(__file__).parents[2] / paths["raw_data"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} samples to {output_path}")
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")

    return df


if __name__ == "__main__":
    generate_data()
