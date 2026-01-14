"""Make predictions with trained model."""

import yaml
import pandas as pd
import joblib
from pathlib import Path
from typing import Union
import numpy as np


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parents[2] / "configs" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model():
    """Load the trained model."""
    config = load_config()
    paths = config["paths"]
    base_path = Path(__file__).parents[2]
    model_path = base_path / paths["model"]

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run training first.")

    return joblib.load(model_path)


def predict(features: Union[list, np.ndarray]) -> tuple[list[int], list[list[float]]]:
    """
    Make predictions for given features.

    Args:
        features: List of feature vectors, e.g., [[1.0, 2.0, ...], [3.0, 4.0, ...]]

    Returns:
        Tuple of (predictions, probabilities)
    """
    model = load_model()
    predictions = model.predict(features).tolist()
    probabilities = model.predict_proba(features).tolist()
    return predictions, probabilities


def main():
    """Demo: predict on a few samples from test set."""
    config = load_config()
    paths = config["paths"]
    base_path = Path(__file__).parents[2]

    # Load a few test samples
    test_path = base_path / paths["test_data"]

    if not test_path.exists():
        print(f"Test data not found at {test_path}. Run preprocessing first.")
        return

    test_df = pd.read_csv(test_path)

    X_test = test_df.drop("target", axis=1).head(5)
    y_true = test_df["target"].head(5).tolist()

    # Make predictions
    predictions, probabilities = predict(X_test.values.tolist())

    print("=" * 50)
    print("PREDICTION DEMO")
    print("=" * 50)
    for i, (pred, true, prob) in enumerate(zip(predictions, y_true, probabilities)):
        status = "+" if pred == true else "x"
        confidence = max(prob) * 100
        print(
            f"Sample {i}: predicted={pred}, actual={true} {status} (confidence: {confidence:.1f}%)"
        )


if __name__ == "__main__":
    main()
