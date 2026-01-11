"""Preprocess data: split into train/test sets."""

import yaml
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parents[2] / "configs" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def preprocess_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw data, split into train/test, and save."""
    config = load_config()
    data_config = config["data"]
    paths = config["paths"]
    base_path = Path(__file__).parents[2]

    # Load raw data
    raw_path = base_path / paths["raw_data"]
    print(f"Loading data from {raw_path}...")
    df = pd.read_csv(raw_path)

    # Split features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=data_config["test_size"],
        random_state=data_config["random_seed"],
        stratify=y,
    )

    # Optional: Scale features (uncomment if needed)
    # scaler = StandardScaler()
    # X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    # X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    # joblib.dump(scaler, base_path / "models" / "scaler.joblib")

    # Combine back to DataFrames
    train_df = X_train.copy()
    train_df["target"] = y_train.values

    test_df = X_test.copy()
    test_df["target"] = y_test.values

    # Save
    train_path = base_path / paths["train_data"]
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)

    test_path = base_path / paths["test_data"]
    test_df.to_csv(test_path, index=False)

    print(f"Train set: {len(train_df)} samples -> {train_path}")
    print(f"Test set: {len(test_df)} samples -> {test_path}")

    return train_df, test_df


if __name__ == "__main__":
    preprocess_data()
