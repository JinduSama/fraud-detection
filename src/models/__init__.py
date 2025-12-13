"""Models package for fraud detection algorithms."""
from .preprocessing import DataPreprocessor
from .clustering import FraudDetector
from .base import BaseDetector, DetectionResult
from .features import FeatureExtractor
from .ensemble import EnsembleDetector, FusionStrategy
from .explainer import FraudExplainer
from .detectors import (
    IsolationForestDetector,
    LocalOutlierFactorDetector,
    DBSCANDetector,
    GraphDetector,
)

__all__ = [
    # Legacy
    "DataPreprocessor",
    "FraudDetector",
    # Base classes
    "BaseDetector",
    "DetectionResult",
    # Feature extraction
    "FeatureExtractor",
    # Detectors
    "IsolationForestDetector",
    "LocalOutlierFactorDetector",
    "DBSCANDetector",
    "GraphDetector",
    # Ensemble
    "EnsembleDetector",
    "FusionStrategy",
    # Explainability
    "FraudExplainer",
]
