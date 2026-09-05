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


class TestEntityDataFrame:
    def test_get_entity_dataframe_raises_without_db_env(self, monkeypatch):
        """get_entity_dataframe raises a clear error when no DB config exists."""
        from src.data import fetch_features

        for var in (
            "DATABASE_URL",
            "SUPABASE_DB_HOST",
            "SUPABASE_DB_USER",
            "SUPABASE_DB_PASSWORD",
        ):
            monkeypatch.delenv(var, raising=False)

        with pytest.raises(RuntimeError, match="Database connection not configured"):
            fetch_features.get_entity_dataframe(load_config())


class TestFeatureRefs:
    def test_all_feature_refs_counts_and_format(self):
        """_all_feature_refs returns all 79 features as view:name strings."""
        from src.data.fetch_features import _all_feature_refs

        refs = _all_feature_refs()
        assert len(refs) == 79
        views = {ref.split(":")[0] for ref in refs}
        assert views == {
            "agent_product_interaction",
            "agent_features",
            "product_features",
        }
        assert all(":" in ref for ref in refs)

    def test_interaction_features_come_first(self):
        """Interaction view has priority (training rows own conflicting columns)."""
        from src.data.fetch_features import _all_feature_refs

        refs = _all_feature_refs()
        assert refs[0].startswith("agent_product_interaction:")
        assert refs[-1].startswith("product_features:")


class TestRenameMap:
    def _make_df(self, columns):
        import pandas as pd

        return pd.DataFrame({c: [0] for c in columns})

    def test_strips_feast_prefixes(self):
        from src.data.fetch_features import _build_rename_map

        df = self._make_df(["agent_features__age", "product_features__price"])
        rename_map = _build_rename_map(df)
        assert rename_map["agent_features__age"] == "age"
        assert rename_map["product_features__price"] == "price"

    def test_conflicting_columns_get_view_suffixes(self):
        from src.data.fetch_features import _build_rename_map

        df = self._make_df(
            [
                "agent_product_interaction__coupon_redemption_rate",
                "agent_features__coupon_redemption_rate",
                "product_features__product_category",
                "agent_product_interaction__product_category",
            ]
        )
        rename_map = _build_rename_map(df)
        assert (
            rename_map["agent_product_interaction__coupon_redemption_rate"]
            == "coupon_redemption_rate"
        )
        assert (
            rename_map["agent_features__coupon_redemption_rate"]
            == "coupon_redemption_rate_agent"
        )
        assert rename_map["product_features__product_category"] == "product_category"
        assert (
            rename_map["agent_product_interaction__product_category"]
            == "product_category_dup"
        )

    def test_event_timestamp_dropped(self):
        from src.data.fetch_features import _build_rename_map

        df = self._make_df(["agent_features__age", "event_timestamp"])
        rename_map = _build_rename_map(df)
        assert "event_timestamp" not in rename_map.values()
        assert list(rename_map.values()) == ["age"]