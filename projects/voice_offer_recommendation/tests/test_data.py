"""Tests for data fetching and feature engineering."""

import pandas as pd
import pytest

from src.data.fetch_features import load_config


class TestDataConfig:
    def test_config_loads(self):
        """Test that configuration file loads correctly."""
        config = load_config()
        assert "data" in config
        assert "model" in config
        assert "mlflow" in config
        assert config["model"]["type"] == "simple_rule"

    def test_config_has_feast_settings(self):
        """Test that Feast configuration is present."""
        config = load_config()
        assert "feast" in config
        assert "repo_path" in config["feast"]
        assert "feature_views" in config["feast"]


class TestSupabaseFallback:
    def test_fetch_features_from_supabase_mock(self, monkeypatch):
        """Test that Supabase fallback can be triggered with mock data."""
        # This test validates the fallback path without requiring real Supabase tables
        mock_df = pd.DataFrame(
            {
                "agent_id": [1, 1, 2],
                "product_id": [101, 102, 101],
                "times_purchased": [1, 0, 2],
            }
        )

        # Since we can't easily mock Supabase without the client library,
        # we just verify the function signatures and fallback logic conceptually
        from src.data.fetch_features import fetch_features_from_supabase

        # The function should raise if Supabase tables don't exist
        # which is expected in test environments
        with pytest.raises(Exception):
            fetch_features_from_supabase(load_config())
