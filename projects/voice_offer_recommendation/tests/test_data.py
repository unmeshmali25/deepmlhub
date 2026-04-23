"""Tests for data fetching and feature engineering."""

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


class TestDatabaseFetch:
    def test_fetch_features_from_db_raises_without_connection(self):
        """Test that fetch_features_from_db raises without a real DB connection."""
        from src.data.fetch_features import fetch_features_from_db

        with pytest.raises(Exception):
            fetch_features_from_db(load_config())

    def test_get_db_engine_uses_env_var(self, monkeypatch):
        """Test that get_db_engine uses DATABASE_URL env var."""
        from src.data.fetch_features import get_db_engine

        monkeypatch.setenv(
            "DATABASE_URL",
            "postgresql://user:pass@localhost:5432/testdb?sslmode=require",
        )
        engine, schema = get_db_engine(load_config())
        assert schema == "dbt_vor"
