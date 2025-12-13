"""
Base Detector Interface Module.

Defines the abstract base class that all fraud detectors must implement.
This ensures consistent interface across different detection algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class DetectionResult:
    """Container for detection results from a single detector."""
    
    scores: np.ndarray  # Anomaly scores (0-1, higher = more anomalous)
    is_fraud: np.ndarray  # Boolean predictions
    reasons: list[str]  # Explanation for each prediction
    probabilities: Optional[np.ndarray] = None  # Calibrated probabilities if available
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a DataFrame."""
        return pd.DataFrame({
            "score": self.scores,
            "is_fraud": self.is_fraud,
            "reason": self.reasons,
            "probability": self.probabilities if self.probabilities is not None else self.scores
        })


class BaseDetector(ABC):
    """
    Abstract base class for all fraud detectors.
    
    This class defines the interface that all detector implementations
    must follow, enabling consistent usage and ensemble combinations.
    
    Each detector should:
    1. Learn patterns from training data (fit)
    2. Score new records for anomalies (predict)
    3. Provide calibrated probabilities (predict_proba)
    4. Explain individual predictions (explain)
    """
    
    def __init__(self, name: str = "BaseDetector"):
        """
        Initialize the detector.
        
        Args:
            name: Human-readable name for the detector.
        """
        self.name = name
        self._is_fitted = False
        self._threshold = 0.5
    
    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseDetector":
        """
        Fit the detector on training data.
        
        Args:
            df: DataFrame with customer records to learn from.
            
        Returns:
            self: The fitted detector instance.
        """
        pass
    
    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict fraud scores for each record.
        
        Args:
            df: DataFrame with customer records to score.
            
        Returns:
            DataFrame with columns: 'score', 'is_fraud', 'reason'
        """
        pass
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get calibrated probability estimates.
        
        Default implementation uses the raw scores.
        Subclasses should override for proper calibration.
        
        Args:
            df: DataFrame with customer records.
            
        Returns:
            Array of probabilities (0-1) for each record.
        """
        predictions = self.predict(df)
        return predictions["score"].values
    
    def explain(self, df: pd.DataFrame, idx: int) -> dict:
        """
        Explain why a specific record was flagged.
        
        Args:
            df: DataFrame with customer records.
            idx: Index of the record to explain.
            
        Returns:
            Dictionary with explanation details including feature contributions.
        """
        # Default implementation - subclasses should provide detailed explanations
        predictions = self.predict(df.iloc[[idx]])
        return {
            "index": idx,
            "score": float(predictions["score"].iloc[0]),
            "is_fraud": bool(predictions["is_fraud"].iloc[0]),
            "reason": predictions["reason"].iloc[0],
            "detector": self.name,
            "feature_contributions": {}
        }
    
    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the detector and return predictions.
        
        Convenience method combining fit and predict.
        
        Args:
            df: DataFrame with customer records.
            
        Returns:
            DataFrame with predictions.
        """
        self.fit(df)
        return self.predict(df)
    
    def set_threshold(self, threshold: float) -> None:
        """
        Set the decision threshold for fraud classification.
        
        Args:
            threshold: Score threshold above which records are flagged as fraud.
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        self._threshold = threshold
    
    def get_threshold(self) -> float:
        """Get the current decision threshold."""
        return self._threshold
    
    @property
    def is_fitted(self) -> bool:
        """Check if the detector has been fitted."""
        return self._is_fitted
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self._is_fitted})"
