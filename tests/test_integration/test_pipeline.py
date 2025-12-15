"""Integration tests for the complete fraud detection pipeline."""

import pandas as pd
import pytest

from src.models import (
    DBSCANDetector,
    EnsembleDetector,
    FusionStrategy,
    IsolationForestDetector,
    LocalOutlierFactorDetector,
)
from src.models.features import FeatureExtractor


class TestIsolationForestDetector:
    """Integration tests for Isolation Forest detector."""

    def test_fit_predict(self, sample_customer_data):
        """Test basic fit and predict cycle."""
        detector = IsolationForestDetector(contamination=0.3, random_state=42)
        detector.fit(sample_customer_data)
        results = detector.predict(sample_customer_data)

        assert "is_fraud" in results.columns
        assert "score" in results.columns
        assert len(results) == len(sample_customer_data)
        assert results["is_fraud"].dtype == bool

    def test_detects_outliers(self, sample_customer_data):
        """Test that obvious outliers are flagged."""
        detector = IsolationForestDetector(contamination=0.3, random_state=42)
        detector.fit(sample_customer_data)
        results = detector.predict(sample_customer_data)

        # At least one record should be flagged
        assert results["is_fraud"].sum() >= 1

    def test_score_range(self, sample_customer_data):
        """Test scores are in valid range."""
        detector = IsolationForestDetector(contamination=0.3, random_state=42)
        detector.fit(sample_customer_data)
        results = detector.predict(sample_customer_data)

        assert results["score"].min() >= 0.0
        assert results["score"].max() <= 1.0


class TestDBSCANDetector:
    """Integration tests for DBSCAN detector."""

    def test_fit_predict(self, sample_customer_data):
        """Test basic fit and predict cycle."""
        detector = DBSCANDetector(eps=0.4, min_samples=2)
        detector.fit(sample_customer_data)
        results = detector.predict(sample_customer_data)

        assert "is_fraud" in results.columns
        assert "score" in results.columns
        assert len(results) == len(sample_customer_data)

    def test_finds_clusters(self, sample_customer_data):
        """Test that similar records are clustered."""
        detector = DBSCANDetector(eps=0.4, min_samples=2)
        detector.fit(sample_customer_data)

        # Should find at least one cluster (similar records exist)
        assert len(detector.clusters) >= 1

    def test_cluster_membership(self, sample_customer_data):
        """Test cluster membership is recorded."""
        detector = DBSCANDetector(eps=0.4, min_samples=2)
        detector.fit(sample_customer_data)
        results = detector.predict(sample_customer_data)

        # Flagged records should have reason
        flagged = results[results["is_fraud"]]
        if len(flagged) > 0:
            assert all(flagged["reason"].str.len() > 0)


class TestEnsembleDetector:
    """Integration tests for ensemble detector."""

    def test_weighted_avg_strategy(self, sample_customer_data):
        """Test weighted average fusion strategy."""
        iso = IsolationForestDetector(contamination=0.3, random_state=42)
        dbscan = DBSCANDetector(eps=0.4, min_samples=2)

        ensemble = EnsembleDetector(
            detectors=[(iso, 0.5), (dbscan, 0.5)],
            strategy=FusionStrategy.WEIGHTED_AVG,
        )
        ensemble.fit(sample_customer_data)
        results = ensemble.predict(sample_customer_data)

        assert "is_fraud" in results.columns
        assert "score" in results.columns
        assert len(results) == len(sample_customer_data)

    def test_voting_strategy(self, sample_customer_data):
        """Test voting fusion strategy."""
        iso = IsolationForestDetector(contamination=0.3, random_state=42)
        dbscan = DBSCANDetector(eps=0.4, min_samples=2)
        lof = LocalOutlierFactorDetector(n_neighbors=3, contamination=0.3)

        ensemble = EnsembleDetector(
            detectors=[(iso, 1.0), (dbscan, 1.0), (lof, 1.0)],
            strategy=FusionStrategy.VOTING,
        )
        ensemble.fit(sample_customer_data)
        results = ensemble.predict(sample_customer_data)

        assert "is_fraud" in results.columns
        assert len(results) == len(sample_customer_data)

    def test_threshold_adjustment(self, sample_customer_data):
        """Test threshold affects flagging."""
        iso = IsolationForestDetector(contamination=0.3, random_state=42)

        # High threshold = fewer flags
        ensemble_high = EnsembleDetector(
            detectors=[(iso, 1.0)],
            strategy=FusionStrategy.WEIGHTED_AVG,
        )
        ensemble_high.set_threshold(0.9)
        ensemble_high.fit(sample_customer_data)
        results_high = ensemble_high.predict(sample_customer_data)

        # Low threshold = more flags
        iso2 = IsolationForestDetector(contamination=0.3, random_state=42)
        ensemble_low = EnsembleDetector(
            detectors=[(iso2, 1.0)],
            strategy=FusionStrategy.WEIGHTED_AVG,
        )
        ensemble_low.set_threshold(0.1)
        ensemble_low.fit(sample_customer_data)
        results_low = ensemble_low.predict(sample_customer_data)

        assert results_low["is_fraud"].sum() >= results_high["is_fraud"].sum()


class TestFeatureExtractor:
    """Integration tests for feature extraction."""

    def test_extract_features(self, sample_customer_data):
        """Test feature extraction produces features."""
        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_customer_data)

        assert len(features) == len(sample_customer_data)
        assert len(features.columns) > 10  # Should have many features

    def test_feature_names(self, sample_customer_data):
        """Test expected feature names are present."""
        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_customer_data)

        # Check for expected feature categories
        columns = features.columns.tolist()
        assert any("len_" in c for c in columns)  # Length features
        assert any("entropy" in c for c in columns)  # Entropy features

    def test_empty_dataframe(self, empty_dataframe):
        """Test handling of empty dataframe."""
        extractor = FeatureExtractor()
        features = extractor.extract_features(empty_dataframe)

        assert len(features) == 0


class TestLOFDetector:
    """Integration tests for Local Outlier Factor detector."""

    def test_fit_predict(self, sample_customer_data):
        """Test basic fit and predict cycle."""
        lof = LocalOutlierFactorDetector(n_neighbors=3, contamination=0.3)
        lof.fit(sample_customer_data)
        results = lof.predict(sample_customer_data)

        assert "is_fraud" in results.columns
        assert "score" in results.columns
        assert len(results) == len(sample_customer_data)
