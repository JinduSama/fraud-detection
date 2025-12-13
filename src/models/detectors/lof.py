"""
Local Outlier Factor Detector.

Implements anomaly detection using LOF algorithm which measures
local density deviation of samples compared to their neighbors.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from ..base import BaseDetector
from ..features import FeatureExtractor


class LocalOutlierFactorDetector(BaseDetector):
    """
    Fraud detector using Local Outlier Factor algorithm.
    
    LOF measures the local density deviation of a given sample with respect
    to its neighbors. It is effective at detecting samples that have a 
    substantially lower density than their neighbors.
    """
    
    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: str | float = "auto",
        metric: str = "minkowski",
        novelty: bool = True,
        n_jobs: int = -1,
        name: str = "LocalOutlierFactor"
    ):
        """
        Initialize the LOF detector.
        
        Args:
            n_neighbors: Number of neighbors for kNN queries.
            contamination: Expected proportion of outliers.
            metric: Distance metric to use.
            novelty: If True, enables predict() on new data.
            n_jobs: Number of parallel jobs.
            name: Human-readable name for the detector.
        """
        super().__init__(name=name)
        
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric
        self.novelty = novelty
        self.n_jobs = n_jobs
        
        self._model: Optional[LocalOutlierFactor] = None
        self._scaler: Optional[StandardScaler] = None
        self._feature_extractor: Optional[FeatureExtractor] = None
        self._feature_names: list[str] = []
        self._training_lof_scores: Optional[np.ndarray] = None
    
    def fit(self, df: pd.DataFrame) -> "LocalOutlierFactorDetector":
        """
        Fit the LOF model on training data.
        
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
        
        # Adjust n_neighbors if dataset is small
        actual_neighbors = min(self.n_neighbors, len(X) - 1)
        
        # Train LOF
        self._model = LocalOutlierFactor(
            n_neighbors=actual_neighbors,
            contamination=self.contamination,
            metric=self.metric,
            novelty=self.novelty,
            n_jobs=self.n_jobs
        )
        self._model.fit(X_scaled)
        
        # Store training LOF scores for reference
        self._training_lof_scores = -self._model.negative_outlier_factor_
        
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
        
        # Get LOF scores (negative_outlier_factor_ gives lower values for outliers)
        raw_scores = -self._model.decision_function(X_scaled)
        
        # Normalize to 0-1 range
        min_score = min(raw_scores.min(), self._training_lof_scores.min())
        max_score = max(raw_scores.max(), self._training_lof_scores.max())
        scores = (raw_scores - min_score) / (max_score - min_score + 1e-10)
        
        # Get binary predictions
        predictions = self._model.predict(X_scaled)
        is_fraud = predictions == -1  # -1 indicates outlier
        
        # Generate reasons
        reasons = []
        for idx, (fraud, score) in enumerate(zip(is_fraud, scores)):
            if fraud:
                # Find top contributing features
                top_features = self._get_top_features(X_scaled[idx])
                reasons.append(f"local_density_anomaly (top: {', '.join(top_features[:3])})")
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
        
        Args:
            df: DataFrame with customer records.
            
        Returns:
            Array of probability-like scores.
        """
        return self.predict(df)["score"].values
    
    def _get_top_features(self, x_scaled: np.ndarray, top_n: int = 5) -> list[str]:
        """
        Get features with highest absolute deviation from the mean.
        
        Args:
            x_scaled: Scaled feature vector.
            top_n: Number of top features to return.
            
        Returns:
            List of feature names with highest deviation.
        """
        if not self._feature_names:
            return []
        
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
        
        # Extract features
        feature_df = self._feature_extractor.extract_features(single_df)
        numeric_features = self._feature_extractor.get_numeric_features(feature_df)
        X = numeric_features.fillna(0).values
        X_scaled = self._scaler.transform(X)
        
        # Get feature contributions
        feature_contributions = {}
        for i, name in enumerate(self._feature_names):
            feature_contributions[name] = float(abs(X_scaled[0, i]))
        
        sorted_contributions = dict(
            sorted(feature_contributions.items(), key=lambda x: x[1], reverse=True)
        )
        
        return {
            "index": idx,
            "score": float(predictions["score"].iloc[0]),
            "is_fraud": bool(predictions["is_fraud"].iloc[0]),
            "reason": predictions["reason"].iloc[0],
            "detector": self.name,
            "n_neighbors": self.n_neighbors,
            "feature_contributions": sorted_contributions,
            "explanation": "Record has unusual local density compared to neighbors"
        }
    
    @property
    def model(self) -> Optional[LocalOutlierFactor]:
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
    
    detector = LocalOutlierFactorDetector(n_neighbors=3, contamination=0.2)
    detector.fit(test_data)
    results = detector.predict(test_data)
    
    print("LOF Results:")
    print(results)
