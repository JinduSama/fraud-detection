"""Unit tests for configuration module with Pydantic v2 validation."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import (
    DBSCANConfig,
    EnsembleConfig,
    FraudDetectionConfig,
    IsolationForestConfig,
    PathsConfig,
    get_default_config,
    load_config,
)


class TestDBSCANConfig:
    """Tests for DBSCAN configuration validation."""

    def test_default_values(self):
        """Test default DBSCAN configuration values."""
        config = DBSCANConfig()
        assert config.enabled is True
        assert config.eps == 0.35
        assert config.min_samples == 2
        assert config.distance_metric == "jaro_winkler"

    def test_eps_range_validation(self):
        """Test eps must be between 0 and 1."""
        with pytest.raises(ValueError):
            DBSCANConfig(eps=1.5)
        with pytest.raises(ValueError):
            DBSCANConfig(eps=-0.1)

    def test_valid_distance_metrics(self):
        """Test only valid distance metrics are accepted."""
        for metric in ["jaro_winkler", "levenshtein", "damerau"]:
            config = DBSCANConfig(distance_metric=metric)
            assert config.distance_metric == metric

    def test_invalid_distance_metric(self):
        """Test invalid distance metric raises error."""
        with pytest.raises(ValueError):
            DBSCANConfig(distance_metric="invalid")


class TestIsolationForestConfig:
    """Tests for Isolation Forest configuration validation."""

    def test_contamination_auto(self):
        """Test 'auto' is valid for contamination."""
        config = IsolationForestConfig(contamination="auto")
        assert config.contamination == "auto"

    def test_contamination_float_string(self):
        """Test valid float string for contamination."""
        config = IsolationForestConfig(contamination="0.1")
        assert config.contamination == "0.1"

    def test_contamination_invalid_range(self):
        """Test contamination outside valid range raises error."""
        with pytest.raises(ValueError):
            IsolationForestConfig(contamination="0.6")  # > 0.5
        with pytest.raises(ValueError):
            IsolationForestConfig(contamination="0.0")  # <= 0

    def test_n_estimators_minimum(self):
        """Test n_estimators must be at least 1."""
        with pytest.raises(ValueError):
            IsolationForestConfig(n_estimators=0)


class TestEnsembleConfig:
    """Tests for ensemble configuration validation."""

    def test_valid_strategies(self):
        """Test all valid fusion strategies are accepted."""
        for strategy in ["max", "weighted_avg", "voting", "stacking"]:
            config = EnsembleConfig(strategy=strategy)
            assert config.strategy == strategy

    def test_invalid_strategy(self):
        """Test invalid strategy raises error."""
        with pytest.raises(ValueError):
            EnsembleConfig(strategy="invalid")

    def test_threshold_range(self):
        """Test threshold must be between 0 and 1."""
        config = EnsembleConfig(threshold=0.5)
        assert config.threshold == 0.5

        with pytest.raises(ValueError):
            EnsembleConfig(threshold=1.5)

    def test_negative_weight_validation(self):
        """Test negative weights are rejected."""
        with pytest.raises(ValueError):
            EnsembleConfig(weights={"dbscan": -0.1})


class TestPathsConfig:
    """Tests for paths configuration."""

    def test_get_path(self):
        """Test get_path returns correct full path."""
        config = PathsConfig(data_dir="data", detected_fraud="output.csv")
        path = config.get_path("detected_fraud")
        assert path == Path("data/output.csv")

    def test_default_paths(self):
        """Test default path values."""
        config = PathsConfig()
        assert config.data_dir == "data"
        assert config.detected_fraud == "detected_fraud.csv"
        assert config.customer_dataset == "customer_dataset.csv"


class TestFraudDetectionConfig:
    """Tests for main configuration class."""

    def test_default_config(self):
        """Test default configuration is valid."""
        config = get_default_config()
        assert config.detectors.dbscan.enabled is True
        assert config.ensemble.strategy == "weighted_avg"
        assert config.paths.data_dir == "data"

    def test_config_sync_paths(self):
        """Test data paths are synced from paths config."""
        config = FraudDetectionConfig(
            paths=PathsConfig(data_dir="custom_data")
        )
        assert "custom_data" in config.data.input_path
        assert "custom_data" in config.data.output_path

    def test_extra_fields_forbidden(self):
        """Test unknown fields are rejected."""
        with pytest.raises(ValueError):
            FraudDetectionConfig.model_validate({"unknown_field": "value"})


class TestLoadConfig:
    """Tests for config loading functionality."""

    def test_load_nonexistent_file(self):
        """Test loading with nonexistent file returns defaults."""
        config = load_config("/nonexistent/path.yaml")
        assert config.detectors.dbscan.eps == 0.35

    def test_env_override_dbscan_eps(self):
        """Test environment variable override for DBSCAN eps."""
        with patch.dict(os.environ, {"FRAUD_DBSCAN_EPS": "0.5"}):
            config = load_config()
            assert config.detectors.dbscan.eps == 0.5

    def test_env_override_ensemble_threshold(self):
        """Test environment variable override for ensemble threshold."""
        with patch.dict(os.environ, {"FRAUD_ENSEMBLE_THRESHOLD": "0.8"}):
            config = load_config()
            assert config.ensemble.threshold == 0.8

    def test_env_override_enabled_flag(self):
        """Test environment variable override for enabled flags."""
        with patch.dict(os.environ, {"FRAUD_DBSCAN_ENABLED": "false"}):
            config = load_config()
            assert config.detectors.dbscan.enabled is False
