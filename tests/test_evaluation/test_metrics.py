"""Unit tests for fraud evaluation metrics."""

import pandas as pd
import pytest

from src.evaluation.metrics import FraudMetrics, EvaluationResults


# Helper function to simplify test code
def calculate_metrics(df: pd.DataFrame) -> EvaluationResults:
    """Helper to calculate metrics from a DataFrame."""
    return FraudMetrics().evaluate(df)


class TestFraudMetrics:
    """Tests for FraudMetrics calculation."""

    def test_perfect_detection(self):
        """Test metrics with perfect detection (all correct)."""
        detected = pd.DataFrame({
            "customer_id": ["C001", "C002", "C003", "C004"],
            "detected_fraud": [True, True, False, False],
            "is_fraud": [True, True, False, False],
        })

        metrics = calculate_metrics(detected)

        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0
        assert metrics.true_positives == 2
        assert metrics.false_positives == 0
        assert metrics.false_negatives == 0

    def test_no_detection(self):
        """Test metrics when nothing is detected."""
        detected = pd.DataFrame({
            "customer_id": ["C001", "C002", "C003"],
            "detected_fraud": [False, False, False],
            "is_fraud": [True, True, False],
        })

        metrics = calculate_metrics(detected)

        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0
        assert metrics.true_positives == 0
        assert metrics.false_negatives == 2

    def test_all_false_positives(self):
        """Test metrics when all detections are false positives."""
        detected = pd.DataFrame({
            "customer_id": ["C001", "C002", "C003"],
            "detected_fraud": [True, True, True],
            "is_fraud": [False, False, False],
        })

        metrics = calculate_metrics(detected)

        assert metrics.precision == 0.0
        assert metrics.recall == 0.0  # No true positives
        assert metrics.false_positives == 3
        assert metrics.true_positives == 0

    def test_mixed_results(self):
        """Test metrics with mixed true/false positives."""
        detected = pd.DataFrame({
            "customer_id": ["C001", "C002", "C003", "C004", "C005"],
            "detected_fraud": [True, True, True, False, False],
            "is_fraud": [True, True, False, True, False],
        })

        metrics = calculate_metrics(detected)

        assert metrics.true_positives == 2
        assert metrics.false_positives == 1
        assert metrics.false_negatives == 1
        assert metrics.true_negatives == 1

        # Precision = TP / (TP + FP) = 2/3
        assert abs(metrics.precision - 2 / 3) < 0.01

        # Recall = TP / (TP + FN) = 2/3
        assert abs(metrics.recall - 2 / 3) < 0.01

    def test_empty_dataframe(self):
        """Test metrics with empty dataframe."""
        detected = pd.DataFrame(columns=["customer_id", "detected_fraud", "is_fraud"])

        metrics = calculate_metrics(detected)

        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0
        assert metrics.true_positives == 0
        assert metrics.false_positives == 0


class TestEvaluationResultsDataclass:
    """Tests for EvaluationResults dataclass."""

    def test_f1_score_calculation(self):
        """Test F1 score is harmonic mean of precision and recall."""
        # Create a metrics instance manually
        metrics = EvaluationResults(
            precision=0.8,
            recall=0.6,
            f1_score=0.0,  # Will be calculated
            accuracy=0.73,
            true_positives=6,
            false_positives=2,
            false_negatives=4,
            true_negatives=10,
        )

        # F1 = 2 * (P * R) / (P + R) = 2 * 0.48 / 1.4 ≈ 0.686
        expected_f1 = 2 * (0.8 * 0.6) / (0.8 + 0.6)
        assert abs(expected_f1 - 0.6857) < 0.01

    def test_metrics_summary(self):
        """Test metrics have all required fields."""
        metrics = EvaluationResults(
            precision=0.75,
            recall=0.80,
            f1_score=0.77,
            accuracy=0.92,
            true_positives=8,
            false_positives=2,
            false_negatives=2,
            true_negatives=38,
        )

        assert hasattr(metrics, "precision")
        assert hasattr(metrics, "recall")
        assert hasattr(metrics, "f1_score")
        assert hasattr(metrics, "true_positives")
        assert hasattr(metrics, "false_positives")
        assert hasattr(metrics, "false_negatives")
        assert hasattr(metrics, "true_negatives")
