"""Real-time fraud scoring combining intrinsic features and similarity search.

This module provides production-ready scoring that combines:
1. Intrinsic feature analysis (instant, no comparison needed)
2. FAISS similarity search (fast, against historical records)
3. Combined scoring with configurable thresholds

Example:
    # Training (monthly)
    scorer = RealTimeScorer()
    scorer.train(historical_df)
    scorer.save("models/production_v1/")

    # Production (per request, <50ms)
    scorer = RealTimeScorer.load("models/production_v1/")
    result = scorer.score(new_application)
    if result.alert_level == AlertLevel.HIGH:
        send_to_review_queue(new_application)
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .intrinsic_features import IntrinsicFeatureExtractor, IntrinsicFeatures
from .similarity_index import SimilarityIndex, SimilarRecord


class AlertLevel(Enum):
    """Alert levels for fraud scoring."""
    LOW = "LOW"          # < 0.5: Likely legitimate
    MEDIUM = "MEDIUM"    # 0.5 - 0.8: Needs review
    HIGH = "HIGH"        # >= 0.8: Likely fraud


@dataclass
class ScoringResult:
    """Result of scoring a single record."""
    
    # Scores
    intrinsic_score: float  # 0-1, from Isolation Forest on intrinsic features
    similarity_score: float  # 0-1, max similarity to historical records
    combined_score: float  # 0-1, weighted combination
    
    # Alert level based on combined score
    alert_level: AlertLevel
    
    # Details
    intrinsic_features: IntrinsicFeatures
    similar_records: list[SimilarRecord]
    flags: list[str]  # List of triggered flags
    
    # Timing
    scoring_time_ms: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "intrinsic_score": self.intrinsic_score,
            "similarity_score": self.similarity_score,
            "combined_score": self.combined_score,
            "alert_level": self.alert_level.value,
            "flags": self.flags,
            "similar_records": [
                {
                    "customer_id": r.customer_id,
                    "similarity": r.similarity,
                }
                for r in self.similar_records
            ],
            "scoring_time_ms": self.scoring_time_ms,
        }


@dataclass
class ScorerConfig:
    """Configuration for the real-time scorer."""
    
    # Thresholds
    high_threshold: float = 0.8
    medium_threshold: float = 0.5
    
    # Weights for combining scores
    intrinsic_weight: float = 0.6
    similarity_weight: float = 0.4
    
    # Similarity search settings
    similarity_top_k: int = 5
    similarity_min_score: float = 0.3
    high_similarity_threshold: float = 0.85  # Very similar = suspicious
    
    # Isolation Forest settings
    contamination: float = 0.1
    n_estimators: int = 100
    random_state: int = 42


class RealTimeScorer:
    """Production scorer combining intrinsic analysis and similarity search.
    
    This scorer provides fast (<50ms) fraud scoring by:
    1. Extracting intrinsic features that don't require comparison
    2. Scoring with a pre-trained Isolation Forest
    3. Finding similar historical records with FAISS
    4. Combining scores into a final alert level
    
    The scorer must be trained on historical data before use.
    Training should be done monthly to capture evolving patterns.
    """
    
    def __init__(self, config: Optional[ScorerConfig] = None):
        """Initialize the scorer.
        
        Args:
            config: Scorer configuration. Uses defaults if not provided.
        """
        self.config = config or ScorerConfig()
        
        self.feature_extractor = IntrinsicFeatureExtractor()
        self.similarity_index: Optional[SimilarityIndex] = None
        self.isolation_forest: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        
        self._is_trained = False
    
    def train(
        self,
        df: pd.DataFrame,
        customer_id_col: str = "customer_id",
    ) -> "RealTimeScorer":
        """Train the scorer on historical data.
        
        This fits the Isolation Forest on intrinsic features and builds
        the FAISS similarity index.
        
        Args:
            df: DataFrame with historical customer records.
            customer_id_col: Name of the customer ID column.
            
        Returns:
            Self for method chaining.
        """
        import time
        start = time.time()
        
        # Extract intrinsic features for all records
        print(f"Extracting intrinsic features for {len(df)} records...")
        feature_arrays = []
        for _, row in df.iterrows():
            features = self.feature_extractor.extract(row)
            feature_arrays.append(features.to_array())
        
        X = np.array(feature_arrays)
        
        # Fit scaler
        print("Fitting feature scaler...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest
        print("Training Isolation Forest...")
        self.isolation_forest = IsolationForest(
            contamination=self.config.contamination,
            n_estimators=self.config.n_estimators,
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        self.isolation_forest.fit(X_scaled)
        
        # Build similarity index
        print("Building FAISS similarity index...")
        self.similarity_index = SimilarityIndex()
        self.similarity_index.build(df, customer_id_col=customer_id_col)
        
        self._is_trained = True
        
        elapsed = time.time() - start
        print(f"Training complete in {elapsed:.1f}s. Index size: {len(self.similarity_index)}")
        
        return self
    
    def score(self, record: dict | pd.Series) -> ScoringResult:
        """Score a single record for fraud risk.
        
        Args:
            record: Customer record to score.
            
        Returns:
            ScoringResult with scores, alert level, and details.
        """
        import time
        start = time.time()
        
        if not self._is_trained:
            raise RuntimeError("Scorer must be trained before scoring. Call train() first.")
        
        # Extract intrinsic features
        intrinsic = self.feature_extractor.extract(record)
        
        # Score with Isolation Forest
        X = intrinsic.to_array().reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Get anomaly score (convert from IF's -1 to 1 scale to 0-1)
        # IF returns: -1 for anomalies, +1 for normal
        # We want: high score = more suspicious
        raw_score = -self.isolation_forest.score_samples(X_scaled)[0]
        intrinsic_score = self._normalize_if_score(raw_score)
        
        # Find similar records
        similar = self.similarity_index.find_similar(
            record,
            top_k=self.config.similarity_top_k,
            min_similarity=self.config.similarity_min_score,
        )
        
        # Compute similarity score (high similarity to existing = suspicious)
        if similar:
            max_similarity = max(r.similarity for r in similar)
            # If very similar to existing records, that's suspicious
            similarity_score = max_similarity if max_similarity >= self.config.high_similarity_threshold else max_similarity * 0.5
        else:
            similarity_score = 0.0
        
        # Combine scores
        combined_score = (
            self.config.intrinsic_weight * intrinsic_score +
            self.config.similarity_weight * similarity_score
        )
        combined_score = min(1.0, combined_score)  # Cap at 1.0
        
        # Determine alert level
        if combined_score >= self.config.high_threshold:
            alert_level = AlertLevel.HIGH
        elif combined_score >= self.config.medium_threshold:
            alert_level = AlertLevel.MEDIUM
        else:
            alert_level = AlertLevel.LOW
        
        # Collect flags
        flags = self._collect_flags(intrinsic, similar)
        
        elapsed_ms = (time.time() - start) * 1000
        
        return ScoringResult(
            intrinsic_score=intrinsic_score,
            similarity_score=similarity_score,
            combined_score=combined_score,
            alert_level=alert_level,
            intrinsic_features=intrinsic,
            similar_records=similar,
            flags=flags,
            scoring_time_ms=elapsed_ms,
        )
    
    def score_batch(
        self,
        df: pd.DataFrame,
    ) -> list[ScoringResult]:
        """Score multiple records.
        
        Args:
            df: DataFrame with records to score.
            
        Returns:
            List of ScoringResult objects.
        """
        return [self.score(row) for _, row in df.iterrows()]
    
    def _normalize_if_score(self, raw_score: float) -> float:
        """Normalize Isolation Forest score to 0-1 range.
        
        The raw score from score_samples is typically in [-0.5, 0.5].
        We map this to [0, 1] where higher = more anomalous.
        """
        # Typical range is around -0.5 (normal) to 0.5 (anomaly)
        # Shift and scale to [0, 1]
        normalized = (raw_score + 0.5)
        return max(0.0, min(1.0, normalized))
    
    def _collect_flags(
        self,
        intrinsic: IntrinsicFeatures,
        similar: list[SimilarRecord],
    ) -> list[str]:
        """Collect human-readable flags for the scoring result."""
        flags = []
        
        # Intrinsic flags
        if not intrinsic.iban_valid:
            flags.append("invalid_iban")
        if not intrinsic.iban_country_matches_address:
            flags.append("iban_country_mismatch")
        if intrinsic.email_is_disposable:
            flags.append("disposable_email")
        if intrinsic.email_numeric_ratio > 0.5:
            flags.append("high_email_numeric_ratio")
        if intrinsic.name_has_digits:
            flags.append("digits_in_name")
        if intrinsic.name_keyboard_pattern_score > 0.3:
            flags.append("keyboard_pattern_detected")
        if not intrinsic.postal_code_valid:
            flags.append("invalid_postal_code")
        if intrinsic.address_has_po_box:
            flags.append("po_box_address")
        if intrinsic.email_entropy > 4.0:
            flags.append("high_email_entropy")
        if intrinsic.surname_entropy > 3.5:
            flags.append("high_name_entropy")
        if intrinsic.field_completeness < 0.5:
            flags.append("low_field_completeness")
        
        # Similarity flags
        if similar:
            max_sim = max(r.similarity for r in similar)
            if max_sim >= 0.95:
                flags.append("near_duplicate_found")
            elif max_sim >= self.config.high_similarity_threshold:
                flags.append("high_similarity_match")
        
        return flags
    
    def save(self, directory: str | Path) -> None:
        """Save the trained scorer to disk.
        
        Args:
            directory: Directory to save model files.
        """
        if not self._is_trained:
            raise RuntimeError("Cannot save untrained scorer.")
        
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save Isolation Forest and scaler
        joblib.dump({
            "isolation_forest": self.isolation_forest,
            "scaler": self.scaler,
            "config": self.config,
        }, directory / "intrinsic_model.joblib")
        
        # Save similarity index
        self.similarity_index.save(directory)
        
        print(f"Scorer saved to {directory}")
    
    @classmethod
    def load(cls, directory: str | Path) -> "RealTimeScorer":
        """Load a trained scorer from disk.
        
        Args:
            directory: Directory containing saved model files.
            
        Returns:
            Loaded RealTimeScorer instance.
        """
        directory = Path(directory)
        
        # Load intrinsic model
        model_data = joblib.load(directory / "intrinsic_model.joblib")
        
        # Create instance
        instance = cls(config=model_data["config"])
        instance.isolation_forest = model_data["isolation_forest"]
        instance.scaler = model_data["scaler"]
        
        # Load similarity index
        instance.similarity_index = SimilarityIndex.load(directory)
        
        instance._is_trained = True
        
        return instance
    
    @property
    def is_trained(self) -> bool:
        """Whether the scorer has been trained."""
        return self._is_trained
    
    def __repr__(self) -> str:
        status = "trained" if self._is_trained else "untrained"
        size = len(self.similarity_index) if self.similarity_index else 0
        return f"RealTimeScorer({status}, index_size={size})"
