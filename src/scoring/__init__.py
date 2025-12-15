"""Production scoring package for real-time fraud detection.

This package provides:
- IntrinsicFeatureExtractor: Extract features from single records (no comparison needed)
- SimilarityIndex: FAISS-based fast similarity search
- RealTimeScorer: Combined scorer for production use

Usage:
    # Training (monthly)
    from src.scoring import RealTimeScorer
    scorer = RealTimeScorer()
    scorer.train(historical_df)
    scorer.save("models/production_v1/")

    # Production scoring (<50ms per record)
    scorer = RealTimeScorer.load("models/production_v1/")
    result = scorer.score(new_application)
"""

from .intrinsic_features import IntrinsicFeatureExtractor
from .similarity_index import SimilarityIndex
from .realtime import RealTimeScorer, AlertLevel

__all__ = [
    "IntrinsicFeatureExtractor",
    "SimilarityIndex",
    "RealTimeScorer",
    "AlertLevel",
]
