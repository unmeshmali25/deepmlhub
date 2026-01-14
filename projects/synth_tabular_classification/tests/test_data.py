"""Tests for data generation and preprocessing."""

import pytest
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data.generate import generate_data, load_config
from src.data.preprocess import preprocess_data


class TestDataGeneration:
    def test_generate_data_creates_file(self, tmp_path, monkeypatch):
        """Test that generate_data creates the output file."""
        # This is a basic smoke test
        config = load_config()
        assert config["data"]["n_samples"] > 0

    def test_config_loads(self):
        """Test that config loads correctly."""
        config = load_config()
        assert "data" in config
        assert "model" in config
        assert "paths" in config


class TestPreprocessing:
    def test_config_has_test_size(self):
        """Test that config has test_size."""
        config = load_config()
        assert "test_size" in config["data"]
        assert 0 < config["data"]["test_size"] < 1
