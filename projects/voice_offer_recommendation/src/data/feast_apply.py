"""Wraps feast apply for DVC pipeline integration.

Registers FeatureViews to the Feast registry (data/registry.db)
and optionally sets up online store tables in the feast schema on pgBouncer.

This must run at least once before fetch_features.py, and again whenever
feature_definitions.py changes.
"""

from pathlib import Path

import yaml
from feast import FeatureStore


def load_config() -> dict:
    config_path = Path(__file__).parents[2] / "configs" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def apply_feast() -> None:
    config = load_config()
    feast_config = config.get("feast", {})
    repo_path = feast_config.get("repo_path", "./feature_repo")
    abs_repo_path = Path(__file__).parents[2] / repo_path

    print(f"Applying Feast feature repo at {abs_repo_path}...")
    store = FeatureStore(repo_path=str(abs_repo_path))
    store.apply()
    print("Feast apply complete.")


if __name__ == "__main__":
    apply_feast()
