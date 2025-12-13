"""
Isolation Forest Detector.

Implements anomaly detection using Isolation Forest algorithm with
feature engineering and optional probability calibration.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ..base import BaseDetector
from ..features import FeatureExtractor


class IsolationForestDetector(BaseDetector):
    """
    Fraud detector using Isolation Forest algorithm.
    
    Isolation Forest works by randomly selecting a feature and then
    randomly selecting a split value. Anomalies are isolated in fewer
    steps, giving them a shorter path length in the tree.
    """
    
    def __init__(
        self,
        contamination: str | float = "auto",
        n_estimators: int = 100,
        max_samples: str | int = "auto",
        random_state: Optional[int] = None,
        name: str = "IsolationForest"
    ):
        """
        Initialize the Isolation Forest detector.
        
        Args:
            contamination: Expected proportion of outliers. 'auto' uses
                          heuristics to determine.
            n_estimators: Number of trees in the forest.
            max_samples: Number of samples to draw for each tree.
            random_state: Random seed for reproducibility.
            name: Human-readable name for the detector.
        """
        super().__init__(name=name)
        
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        
        self._model: Optional[IsolationForest] = None
        self._scaler: Optional[StandardScaler] = None
        self._feature_extractor: Optional[FeatureExtractor] = None
        self._feature_names: list[str] = []
    
    def fit(self, df: pd.DataFrame) -> "IsolationForestDetector":
        """
        Fit the Isolation Forest model on training data.
        
        Args:
            df: DataFrame with customer records.
            
        Returns:
            self: The fitted detector.
        """
        # Extract features
        self._feature_extractor = FeatureExtractor(compute_network_features=True)
        feature_df = self._feature_extractor.extract_features(df)
        numeric_features = self._feature_extractor.get_numeric_features(feature_df)
        
        self._feature_names = list(numeric_features.columns)
        
        # Handle missing values
        X = numeric_features.fillna(0).values
        
        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        
        # Train Isolation Forest
        self._model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1
        )
        self._model.fit(X_scaled)
        
        self._is_fitted = True
        return self
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict fraud scores for each record.
        
        Args:
            df: DataFrame with customer records.
            
        Returns:
            DataFrame with 'score', 'is_fraud', 'reason' columns.
        """
        if not self._is_fitted:
            raise RuntimeError("Detector must be fitted before prediction")
        
        # Extract features
        feature_df = self._feature_extractor.extract_features(df)
        numeric_features = self._feature_extractor.get_numeric_features(feature_df)
        
        # Handle missing values
        X = numeric_features.fillna(0).values
        
        # Scale features
        X_scaled = self._scaler.transform(X)
        
        # Get predictions
        # decision_function returns negative for outliers
        raw_scores = -self._model.decision_function(X_scaled)
        
        # Normalize to 0-1 range (using min-max from training bounds)
        # Higher score = more anomalous
        scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-10)
        
        # Get binary predictions
        predictions = self._model.predict(X_scaled)
        is_fraud = predictions == -1  # -1 indicates anomaly
        
        # Generate reasons
        reasons = []
        for idx, (fraud, score) in enumerate(zip(is_fraud, scores)):
            if fraud:
                # Find top contributing features
                top_features = self._get_top_features(X_scaled[idx])
                reasons.append(f"isolation_forest_anomaly (top: {', '.join(top_features[:3])})")
            else:
                reasons.append("")
        
        return pd.DataFrame({
            "score": scores,
            "is_fraud": is_fraud,
            "reason": reasons
        }, index=df.index)
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get probability-like scores.
        
        Note: IsolationForest doesn't provide true probabilities.
        These are normalized decision function values.
        
        Args:
            df: DataFrame with customer records.
            
        Returns:
            Array of probability-like scores.
        """
        return self.predict(df)["score"].values
    
    def _get_top_features(self, x_scaled: np.ndarray, top_n: int = 5) -> list[str]:
        """
        Get features with highest absolute deviation.
        
        Args:
            x_scaled: Scaled feature vector.
            top_n: Number of top features to return.
            
        Returns:
            List of feature names with highest deviation.
        """
        if not self._feature_names:
            return []
        
        # Features with highest absolute scaled values are most unusual
        abs_values = np.abs(x_scaled)
        top_indices = np.argsort(abs_values)[-top_n:][::-1]
        
        return [self._feature_names[i] for i in top_indices if i < len(self._feature_names)]
    
    def explain(self, df: pd.DataFrame, idx: int) -> dict:
        """
        Explain why a specific record was flagged.
        
        Args:
            df: DataFrame with customer records.
            idx: Index of the record to explain.
            
        Returns:
            Dictionary with explanation details.
        """
        if not self._is_fitted:
            raise RuntimeError("Detector must be fitted before explanation")
        
        # Get prediction for this record
        single_df = df.iloc[[idx]] if isinstance(idx, int) else df.loc[[idx]]
        predictions = self.predict(single_df)
        
        # Extract features for this record
        feature_df = self._feature_extractor.extract_features(single_df)
        numeric_features = self._feature_extractor.get_numeric_features(feature_df)
        X = numeric_features.fillna(0).values
        X_scaled = self._scaler.transform(X)
        
        # Get feature contributions (based on deviation from mean)
        feature_contributions = {}
        for i, name in enumerate(self._feature_names):
            # Higher absolute scaled value = more unusual = higher contribution
            feature_contributions[name] = float(abs(X_scaled[0, i]))
        
        # Sort by contribution
        sorted_contributions = dict(
            sorted(feature_contributions.items(), key=lambda x: x[1], reverse=True)
        )
        
        return {
            "index": idx,
            "score": float(predictions["score"].iloc[0]),
            "is_fraud": bool(predictions["is_fraud"].iloc[0]),
            "reason": predictions["reason"].iloc[0],
            "detector": self.name,
            "feature_contributions": sorted_contributions,
            "raw_features": {name: float(X[0, i]) for i, name in enumerate(self._feature_names)}
        }
    
    @property
    def model(self) -> Optional[IsolationForest]:
        """Access the underlying sklearn model."""
        return self._model


if __name__ == "__main__":
    # Test the detector
    test_data = pd.DataFrame({
        "customer_id": ["C001", "C002", "C003", "C004", "C005"],
        "surname": ["Mueller", "Smith", "Johnson", "RandomXYZ", "Williams"],
        "first_name": ["Hans", "John", "Jane", "Test123", "Bob"],
        "address": ["Main St 1", "Oak Ave 5", "Pine Rd 10", "123 Fake", "Elm St 20"],
        "email": ["hans.mueller@gmail.com", "john@yahoo.com", "jane@gmail.com", 
                 "x1y2z3@fake.net", "bob.w@outlook.com"],
        "iban": ["DE89370400440532013000", "GB82WEST12345698765432", 
                "DE11111111111111111111", "XX00000000000000000000", "FR7612345678901234567890"],
        "date_of_birth": ["1990-01-01", "1985-06-15", "1992-03-20", "2000-01-01", "1988-12-25"],
        "nationality": ["German", "British", "German", "Unknown", "French"],
    })
    
    detector = IsolationForestDetector(contamination=0.2, random_state=42)
    detector.fit(test_data)
    results = detector.predict(test_data)
    
    print("Isolation Forest Results:")
    print(results)
    
    # Test explanation
    print("\nExplanation for record 3:")
    print(detector.explain(test_data, 3))
