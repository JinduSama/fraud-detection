"""
Ensemble Detector Module.

Combines multiple fraud detectors using configurable fusion strategies
for improved detection performance.
"""

from enum import Enum
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

from .base import BaseDetector


class FusionStrategy(str, Enum):
    """Available fusion strategies for combining detector outputs."""
    MAX = "max"                    # Flag if any detector flags (high recall)
    WEIGHTED_AVG = "weighted_avg"  # Weighted average of scores
    VOTING = "voting"              # Majority voting
    STACKING = "stacking"          # Meta-classifier on detector outputs


class EnsembleDetector(BaseDetector):
    """
    Ensemble fraud detector combining multiple detection algorithms.
    
    Supports various fusion strategies to combine individual detector
    outputs into a unified fraud score.
    """
    
    FUSION_STRATEGIES = list(FusionStrategy)
    
    def __init__(
        self,
        detectors: list[tuple[BaseDetector, float]],
        strategy: FusionStrategy | str = FusionStrategy.WEIGHTED_AVG,
        voting_threshold: float = 0.5,
        name: str = "Ensemble"
    ):
        """
        Initialize the ensemble detector.
        
        Args:
            detectors: List of (detector, weight) tuples.
            strategy: Fusion strategy to use.
            voting_threshold: Threshold for voting strategy (fraction of detectors).
            name: Human-readable name.
        """
        super().__init__(name=name)
        
        self.detectors = detectors
        self.strategy = FusionStrategy(strategy) if isinstance(strategy, str) else strategy
        self.voting_threshold = voting_threshold
        
        self._meta_classifier: Optional[LogisticRegression] = None
        self._detector_predictions: dict[str, pd.DataFrame] = {}
        self._optimal_threshold: float = 0.5
    
    def fit(self, df: pd.DataFrame) -> "EnsembleDetector":
        """
        Fit all detectors in the ensemble.
        
        Args:
            df: DataFrame with customer records.
            
        Returns:
            self: The fitted ensemble.
        """
        for detector, _ in self.detectors:
            if not detector.is_fitted:
                detector.fit(df)
        
        # For stacking strategy, collect predictions to train meta-classifier
        if self.strategy == FusionStrategy.STACKING:
            self._prepare_stacking_features(df)
        
        self._is_fitted = True
        return self
    
    def _prepare_stacking_features(self, df: pd.DataFrame) -> None:
        """Prepare features for stacking meta-classifier."""
        self._detector_predictions = {}
        
        for detector, _ in self.detectors:
            preds = detector.predict(df)
            self._detector_predictions[detector.name] = preds
    
    def _get_stacking_features(self, df: pd.DataFrame) -> np.ndarray:
        """Get feature matrix for stacking from individual detector predictions."""
        features = []
        
        for detector, _ in self.detectors:
            preds = detector.predict(df)
            features.append(preds["score"].values.reshape(-1, 1))
        
        return np.hstack(features)
    
    def fit_stacking(self, df: pd.DataFrame, y_true: pd.Series) -> "EnsembleDetector":
        """
        Fit the stacking meta-classifier.
        
        Args:
            df: DataFrame with customer records.
            y_true: Ground truth labels.
            
        Returns:
            self: The fitted ensemble.
        """
        # First fit all base detectors
        self.fit(df)
        
        # Get stacking features
        X_stack = self._get_stacking_features(df)
        
        # Train meta-classifier
        self._meta_classifier = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight="balanced"
        )
        self._meta_classifier.fit(X_stack, y_true)
        
        return self
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict fraud using the ensemble.
        
        Args:
            df: DataFrame with customer records.
            
        Returns:
            DataFrame with 'score', 'is_fraud', 'reason' columns.
        """
        if not self._is_fitted:
            raise RuntimeError("Ensemble must be fitted before prediction")
        
        # Collect predictions from all detectors
        all_predictions = []
        all_weights = []
        
        for detector, weight in self.detectors:
            preds = detector.predict(df)
            all_predictions.append(preds)
            all_weights.append(weight)
        
        # Fuse predictions based on strategy
        if self.strategy == FusionStrategy.MAX:
            return self._fuse_max(all_predictions, all_weights, df.index)
        elif self.strategy == FusionStrategy.WEIGHTED_AVG:
            return self._fuse_weighted_avg(all_predictions, all_weights, df.index)
        elif self.strategy == FusionStrategy.VOTING:
            return self._fuse_voting(all_predictions, all_weights, df.index)
        elif self.strategy == FusionStrategy.STACKING:
            return self._fuse_stacking(df, all_predictions)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.strategy}")
    
    def _fuse_max(
        self, 
        predictions: list[pd.DataFrame],
        weights: list[float],
        index: pd.Index
    ) -> pd.DataFrame:
        """Fuse using max strategy (flag if any detector flags)."""
        n = len(predictions[0])
        scores = np.zeros(n)
        is_fraud = np.zeros(n, dtype=bool)
        reasons = [""] * n
        
        for preds in predictions:
            scores = np.maximum(scores, preds["score"].values)
            is_fraud = is_fraud | preds["is_fraud"].values
            
            for i, reason in enumerate(preds["reason"]):
                if reason:
                    if reasons[i]:
                        reasons[i] += f"; {reason}"
                    else:
                        reasons[i] = reason
        
        return pd.DataFrame({
            "score": scores,
            "is_fraud": is_fraud,
            "reason": reasons
        }, index=index)
    
    def _fuse_weighted_avg(
        self,
        predictions: list[pd.DataFrame],
        weights: list[float],
        index: pd.Index
    ) -> pd.DataFrame:
        """Fuse using weighted average of scores."""
        n = len(predictions[0])
        weighted_scores = np.zeros(n)
        total_weight = sum(weights)
        
        all_reasons = [[] for _ in range(n)]
        
        for preds, weight in zip(predictions, weights):
            weighted_scores += weight * preds["score"].values
            
            for i, reason in enumerate(preds["reason"]):
                if reason:
                    all_reasons[i].append(reason)
        
        scores = weighted_scores / total_weight
        is_fraud = scores >= self._threshold
        
        reasons = ["; ".join(r) if r else "" for r in all_reasons]
        
        return pd.DataFrame({
            "score": scores,
            "is_fraud": is_fraud,
            "reason": reasons
        }, index=index)
    
    def _fuse_voting(
        self,
        predictions: list[pd.DataFrame],
        weights: list[float],
        index: pd.Index
    ) -> pd.DataFrame:
        """Fuse using majority voting."""
        n = len(predictions[0])
        vote_counts = np.zeros(n)
        scores = np.zeros(n)
        
        all_reasons = [[] for _ in range(n)]
        
        for preds, weight in zip(predictions, weights):
            vote_counts += preds["is_fraud"].values.astype(float) * weight
            scores += preds["score"].values * weight
            
            for i, reason in enumerate(preds["reason"]):
                if reason:
                    all_reasons[i].append(reason)
        
        total_weight = sum(weights)
        vote_fraction = vote_counts / total_weight
        
        is_fraud = vote_fraction >= self.voting_threshold
        scores = scores / total_weight
        
        reasons = ["; ".join(r) if r else "" for r in all_reasons]
        
        return pd.DataFrame({
            "score": scores,
            "is_fraud": is_fraud,
            "reason": reasons
        }, index=index)
    
    def _fuse_stacking(
        self,
        df: pd.DataFrame,
        predictions: list[pd.DataFrame]
    ) -> pd.DataFrame:
        """Fuse using stacking meta-classifier."""
        if self._meta_classifier is None:
            # Fall back to weighted average if meta-classifier not trained
            weights = [w for _, w in self.detectors]
            return self._fuse_weighted_avg(predictions, weights, df.index)
        
        X_stack = self._get_stacking_features(df)
        
        # Get probabilities from meta-classifier
        proba = self._meta_classifier.predict_proba(X_stack)[:, 1]
        is_fraud = proba >= self._threshold
        
        # Collect reasons
        n = len(df)
        all_reasons = [[] for _ in range(n)]
        for preds in predictions:
            for i, reason in enumerate(preds["reason"]):
                if reason:
                    all_reasons[i].append(reason)
        
        reasons = ["; ".join(r) if r else "" for r in all_reasons]
        
        return pd.DataFrame({
            "score": proba,
            "is_fraud": is_fraud,
            "reason": reasons
        }, index=df.index)
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Get ensemble probability scores."""
        return self.predict(df)["score"].values
    
    def optimize_threshold(
        self,
        df: pd.DataFrame,
        y_true: pd.Series,
        metric: str = "f1"
    ) -> float:
        """
        Find optimal threshold using ground truth.
        
        Args:
            df: DataFrame with customer records.
            y_true: Ground truth labels.
            metric: Metric to optimize ('f1', 'precision', 'recall').
            
        Returns:
            Optimal threshold value.
        """
        scores = self.predict(df)["score"].values
        
        precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
        
        # Calculate F1 scores
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        
        if metric == "f1":
            optimal_idx = np.argmax(f1_scores)
        elif metric == "precision":
            # Find threshold with precision >= 0.9 and max recall
            mask = precisions >= 0.9
            if mask.any():
                optimal_idx = np.argmax(recalls[mask])
                optimal_idx = np.where(mask)[0][optimal_idx]
            else:
                optimal_idx = np.argmax(precisions)
        elif metric == "recall":
            # Find threshold with recall >= 0.9 and max precision
            mask = recalls >= 0.9
            if mask.any():
                optimal_idx = np.argmax(precisions[mask])
                optimal_idx = np.where(mask)[0][optimal_idx]
            else:
                optimal_idx = np.argmax(recalls)
        else:
            optimal_idx = np.argmax(f1_scores)
        
        if optimal_idx < len(thresholds):
            self._optimal_threshold = float(thresholds[optimal_idx])
            self._threshold = self._optimal_threshold
        
        return self._optimal_threshold
    
    def explain(self, df: pd.DataFrame, idx: int) -> dict:
        """Explain ensemble prediction for a specific record."""
        if not self._is_fitted:
            raise RuntimeError("Ensemble must be fitted before explanation")
        
        single_df = df.iloc[[idx]] if isinstance(idx, int) else df.loc[[idx]]
        ensemble_pred = self.predict(single_df)
        
        # Get individual detector predictions
        detector_results = {}
        for detector, weight in self.detectors:
            preds = detector.predict(single_df)
            detector_results[detector.name] = {
                "score": float(preds["score"].iloc[0]),
                "is_fraud": bool(preds["is_fraud"].iloc[0]),
                "reason": preds["reason"].iloc[0],
                "weight": weight
            }
        
        return {
            "index": idx,
            "score": float(ensemble_pred["score"].iloc[0]),
            "is_fraud": bool(ensemble_pred["is_fraud"].iloc[0]),
            "reason": ensemble_pred["reason"].iloc[0],
            "detector": self.name,
            "strategy": self.strategy.value,
            "threshold": self._threshold,
            "detector_results": detector_results
        }
    
    def get_detector_weights(self) -> dict[str, float]:
        """Get current detector weights."""
        return {detector.name: weight for detector, weight in self.detectors}
    
    def set_detector_weights(self, weights: dict[str, float]) -> None:
        """Update detector weights."""
        new_detectors = []
        for detector, old_weight in self.detectors:
            new_weight = weights.get(detector.name, old_weight)
            new_detectors.append((detector, new_weight))
        self.detectors = new_detectors


if __name__ == "__main__":
    from .detectors import IsolationForestDetector, DBSCANDetector
    
    # Test the ensemble
    test_data = pd.DataFrame({
        "customer_id": ["C001", "C002", "C003", "C004", "C005"],
        "surname": ["Mueller", "Muller", "Smith", "RandomXYZ", "Mueller"],
        "first_name": ["Hans", "Hans", "John", "Test123", "Hans"],
        "address": ["Main St 1", "Main St 1", "Oak Ave 5", "123 Fake", "Main St 1"],
        "email": ["hans@test.com", "hans@test.com", "john@test.com", 
                 "x1y2z3@fake.net", "h@test.com"],
        "iban": ["DE123", "DE123", "DE456", "XX000", "DE123"],
        "date_of_birth": ["1990-01-01", "1990-01-01", "1985-06-15", "2000-01-01", "1990-01-01"],
        "nationality": ["German", "German", "British", "Unknown", "German"],
    })
    
    # Create detectors
    iso_forest = IsolationForestDetector(contamination=0.3, random_state=42)
    dbscan = DBSCANDetector(eps=0.4, min_samples=2)
    
    # Create ensemble
    ensemble = EnsembleDetector(
        detectors=[
            (iso_forest, 0.5),
            (dbscan, 0.5),
        ],
        strategy=FusionStrategy.WEIGHTED_AVG
    )
    
    ensemble.fit(test_data)
    results = ensemble.predict(test_data)
    
    print("Ensemble Results:")
    print(results)
    
    print("\nExplanation for record 0:")
    print(ensemble.explain(test_data, 0))
