"""
DBSCAN Detector with Sparse Matrix Optimization.

Implements scalable DBSCAN-based fraud detection using sparse matrices
and blocking for efficient computation on large datasets.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
import jellyfish
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.cluster import DBSCAN

from ..base import BaseDetector
from ..preprocessing import DataPreprocessor


@dataclass
class SuspiciousCluster:
    """Represents a cluster of suspicious records."""
    
    cluster_id: int
    record_indices: list[int]
    customer_ids: list[str]
    similarity_score: float
    detection_reason: str


class StringDistanceMetrics:
    """Collection of string distance/similarity metrics."""
    
    @staticmethod
    def jaro_winkler_distance(s1: str, s2: str) -> float:
        """Jaro-Winkler distance between two strings."""
        if not s1 or not s2:
            return 1.0
        similarity = jellyfish.jaro_winkler_similarity(s1, s2)
        return 1.0 - similarity
    
    @staticmethod
    def levenshtein_distance_normalized(s1: str, s2: str) -> float:
        """Normalized Levenshtein distance."""
        if not s1 and not s2:
            return 0.0
        if not s1 or not s2:
            return 1.0
        
        distance = jellyfish.levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        return distance / max_len
    
    @staticmethod
    def damerau_levenshtein_normalized(s1: str, s2: str) -> float:
        """Normalized Damerau-Levenshtein distance."""
        if not s1 and not s2:
            return 0.0
        if not s1 or not s2:
            return 1.0
        
        distance = jellyfish.damerau_levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        return distance / max_len


class DBSCANDetector(BaseDetector):
    """
    Scalable DBSCAN-based fraud detector.
    
    Uses sparse matrices and blocking to efficiently compute
    pairwise distances only where needed.
    """
    
    DEFAULT_FIELD_WEIGHTS = {
        "surname": 0.25,
        "first_name": 0.20,
        "address": 0.20,
        "email": 0.15,
        "iban": 0.20,
    }
    
    def __init__(
        self,
        eps: float = 0.35,
        min_samples: int = 2,
        field_weights: Optional[dict[str, float]] = None,
        distance_metric: str = "jaro_winkler",
        use_sparse: bool = True,
        eps_threshold: Optional[float] = None,
        name: str = "DBSCAN"
    ):
        """
        Initialize the DBSCAN detector.
        
        Args:
            eps: DBSCAN epsilon (max distance for clustering).
            min_samples: Minimum samples to form a cluster.
            field_weights: Weights for each field in similarity.
            distance_metric: Distance metric to use.
            use_sparse: Whether to use sparse matrix (memory efficient).
            eps_threshold: Only store distances below this threshold in sparse matrix.
            name: Human-readable name.
        """
        super().__init__(name=name)
        
        self.eps = eps
        self.min_samples = min_samples
        self.field_weights = field_weights or self.DEFAULT_FIELD_WEIGHTS
        self.use_sparse = use_sparse
        self.eps_threshold = eps_threshold or (eps * 1.5)  # Store nearby pairs
        
        # Select distance function
        distance_functions = {
            "jaro_winkler": StringDistanceMetrics.jaro_winkler_distance,
            "levenshtein": StringDistanceMetrics.levenshtein_distance_normalized,
            "damerau": StringDistanceMetrics.damerau_levenshtein_normalized,
        }
        self.distance_fn = distance_functions.get(
            distance_metric,
            StringDistanceMetrics.jaro_winkler_distance
        )
        
        self.preprocessor = DataPreprocessor()
        self._distance_matrix: Optional[np.ndarray | csr_matrix] = None
        self._clusters: list[SuspiciousCluster] = []
        self._labels: Optional[np.ndarray] = None
        self._processed_df: Optional[pd.DataFrame] = None
    
    def _compute_record_distance(self, row1: pd.Series, row2: pd.Series) -> float:
        """Compute weighted distance between two records."""
        total_distance = 0.0
        total_weight = 0.0
        
        for field, weight in self.field_weights.items():
            norm_field = f"{field}_normalized"
            
            if norm_field in row1.index and norm_field in row2.index:
                val1 = str(row1[norm_field]) if pd.notna(row1[norm_field]) else ""
                val2 = str(row2[norm_field]) if pd.notna(row2[norm_field]) else ""
                
                if field == "iban":
                    if row1.get("iban", "") == row2.get("iban", ""):
                        field_distance = 0.0
                    else:
                        field_distance = 1.0
                else:
                    field_distance = self.distance_fn(val1, val2)
                
                total_distance += weight * field_distance
                total_weight += weight
        
        if total_weight == 0:
            return 1.0
        
        return total_distance / total_weight
    
    def _get_block_pairs(
        self, 
        df: pd.DataFrame, 
        block_columns: list[str]
    ) -> set[tuple[int, int]]:
        """
        Get pairs of record indices that should be compared based on blocking.
        
        Args:
            df: DataFrame with blocking columns.
            block_columns: Columns to use for blocking.
            
        Returns:
            Set of (idx_i, idx_j) pairs to compare.
        """
        pairs = set()
        
        for block_column in block_columns:
            if block_column not in df.columns:
                continue
            
            blocks = df.groupby(block_column).indices
            
            for block_key, indices in blocks.items():
                if len(indices) < 2:
                    continue
                
                indices_list = list(indices)
                for i in range(len(indices_list)):
                    for j in range(i + 1, len(indices_list)):
                        idx_i = indices_list[i]
                        idx_j = indices_list[j]
                        pairs.add((min(idx_i, idx_j), max(idx_i, idx_j)))
        
        # Also add IBAN-based pairs
        if "iban" in df.columns:
            iban_groups = df.groupby("iban").indices
            for iban, indices in iban_groups.items():
                if len(indices) < 2:
                    continue
                indices_list = list(indices)
                for i in range(len(indices_list)):
                    for j in range(i + 1, len(indices_list)):
                        idx_i = indices_list[i]
                        idx_j = indices_list[j]
                        pairs.add((min(idx_i, idx_j), max(idx_i, idx_j)))
        
        return pairs
    
    def _compute_distance_matrix_sparse(
        self, 
        df: pd.DataFrame,
        block_columns: list[str]
    ) -> csr_matrix:
        """
        Compute sparse distance matrix using blocking.
        
        Only stores distances below eps_threshold to save memory.
        
        Args:
            df: Preprocessed DataFrame.
            block_columns: Columns for blocking.
            
        Returns:
            CSR sparse matrix of distances.
        """
        n = len(df)
        sparse_dist = lil_matrix((n, n), dtype=np.float32)
        
        # Get pairs to compare
        pairs = self._get_block_pairs(df, block_columns)
        
        # Compute distances for blocked pairs
        for idx_i, idx_j in pairs:
            dist = self._compute_record_distance(df.iloc[idx_i], df.iloc[idx_j])
            
            if dist < self.eps_threshold:
                sparse_dist[idx_i, idx_j] = dist
                sparse_dist[idx_j, idx_i] = dist
        
        # Fill diagonal with 0
        for i in range(n):
            sparse_dist[i, i] = 0
        
        return sparse_dist.tocsr()
    
    def _compute_distance_matrix_dense(
        self, 
        df: pd.DataFrame,
        block_columns: list[str]
    ) -> np.ndarray:
        """
        Compute dense distance matrix with blocking optimization.
        
        Args:
            df: Preprocessed DataFrame.
            block_columns: Columns for blocking.
            
        Returns:
            Dense NxN distance matrix.
        """
        n = len(df)
        distance_matrix = np.ones((n, n), dtype=np.float32)
        np.fill_diagonal(distance_matrix, 0)
        
        pairs = self._get_block_pairs(df, block_columns)
        
        for idx_i, idx_j in pairs:
            dist = self._compute_record_distance(df.iloc[idx_i], df.iloc[idx_j])
            distance_matrix[idx_i, idx_j] = dist
            distance_matrix[idx_j, idx_i] = dist
        
        return distance_matrix
    
    def fit(self, df: pd.DataFrame) -> "DBSCANDetector":
        """
        Fit the DBSCAN detector on training data.
        
        Args:
            df: DataFrame with customer records.
            
        Returns:
            self: The fitted detector.
        """
        # Preprocess data
        processed = self.preprocessor.preprocess_dataframe(df)
        processed = self.preprocessor.create_blocking_key(processed)
        self._processed_df = processed

        # This detector stores cluster membership as positional indices in the
        # fitted dataset. It is not intended for out-of-sample prediction
        # without refitting.
        self._fit_n = len(df)
        
        # Define blocking columns
        block_columns = [
            "block_surname_soundex",
            "block_iban_country",
            "block_email_domain",
        ]
        
        # Compute distance matrix
        if self.use_sparse:
            self._distance_matrix = self._compute_distance_matrix_sparse(
                processed, block_columns
            )
            # For DBSCAN with sparse, we need to convert appropriately
            # DBSCAN expects dense or uses different algorithm for sparse
            distance_for_dbscan = self._distance_matrix.toarray()
        else:
            distance_for_dbscan = self._compute_distance_matrix_dense(
                processed, block_columns
            )
            self._distance_matrix = distance_for_dbscan
        
        # Run DBSCAN
        dbscan = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric="precomputed"
        )
        self._labels = dbscan.fit_predict(distance_for_dbscan)
        
        # Extract clusters
        self._clusters = []
        unique_labels = set(self._labels)
        unique_labels.discard(-1)
        
        for label in unique_labels:
            indices = np.where(self._labels == label)[0].tolist()
            
            if len(indices) < 2:
                continue
            
            customer_ids = df.iloc[indices]["customer_id"].tolist() if "customer_id" in df.columns else []
            
            # Calculate average similarity
            avg_distance = 0.0
            count = 0
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    if self.use_sparse:
                        avg_distance += self._distance_matrix[indices[i], indices[j]]
                    else:
                        avg_distance += self._distance_matrix[indices[i], indices[j]]
                    count += 1
            
            if count > 0:
                avg_distance /= count
            
            detection_reason = self._infer_detection_reason(
                df.iloc[indices], processed.iloc[indices]
            )
            
            cluster = SuspiciousCluster(
                cluster_id=int(label),
                record_indices=indices,
                customer_ids=customer_ids,
                similarity_score=1.0 - avg_distance,
                detection_reason=detection_reason
            )
            self._clusters.append(cluster)
        
        self._is_fitted = True
        return self
    
    def _infer_detection_reason(
        self, 
        original_df: pd.DataFrame,
        processed_df: pd.DataFrame
    ) -> str:
        """Infer why records were clustered together."""
        reasons = []
        
        if "iban" in original_df.columns:
            unique_ibans = original_df["iban"].nunique()
            if unique_ibans < len(original_df):
                reasons.append("shared_iban")
        
        if "address_normalized" in processed_df.columns:
            unique_addresses = processed_df["address_normalized"].nunique()
            if unique_addresses < len(processed_df):
                reasons.append("same_address")
        
        if "surname_soundex" in processed_df.columns:
            unique_soundex = processed_df["surname_soundex"].nunique()
            if unique_soundex == 1:
                reasons.append("similar_names")
        
        if "email_local" in processed_df.columns:
            local_parts = processed_df["email_local"].tolist()
            if len(set(local_parts)) < len(local_parts):
                reasons.append("similar_emails")
        
        if not reasons:
            reasons.append("high_overall_similarity")
        
        return ", ".join(reasons)
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict fraud for each record based on clusters.
        
        Args:
            df: DataFrame with customer records.
            
        Returns:
            DataFrame with 'score', 'is_fraud', 'reason' columns.
        """
        if not self._is_fitted:
            raise RuntimeError("Detector must be fitted before prediction")

        if hasattr(self, "_fit_n") and len(df) != self._fit_n:
            raise ValueError(
                "DBSCANDetector.predict expects the same dataset size used during fit "
                f"(fit_n={self._fit_n}, got_n={len(df)}). Re-fit to score a different dataset."
            )
        
        n = len(df)
        scores = np.zeros(n)
        is_fraud = np.zeros(n, dtype=bool)
        reasons = [""] * n
        
        # Mark clustered records as fraud
        for cluster in self._clusters:
            for idx in cluster.record_indices:
                scores[idx] = cluster.similarity_score
                is_fraud[idx] = True
                reasons[idx] = cluster.detection_reason
        
        # Also flag direct indicators
        if "iban" in df.columns:
            iban_counts = df["iban"].value_counts()
            shared_ibans = iban_counts[iban_counts > 1].index.tolist()
            
            for iban in shared_ibans:
                mask = df["iban"] == iban
                indices = df[mask].index.tolist()
                for idx in indices:
                    if isinstance(idx, int) and idx < n:
                        if not is_fraud[idx]:
                            scores[idx] = 0.8
                            is_fraud[idx] = True
                            reasons[idx] = "shared_iban"
                        elif "shared_iban" not in reasons[idx]:
                            reasons[idx] += ", shared_iban"
        
        return pd.DataFrame({
            "score": scores,
            "is_fraud": is_fraud,
            "reason": reasons
        }, index=df.index)
    
    def explain(self, df: pd.DataFrame, idx: int) -> dict:
        """Explain why a specific record was flagged."""
        if not self._is_fitted:
            raise RuntimeError("Detector must be fitted before explanation")

        if hasattr(self, "_fit_n") and len(df) != self._fit_n:
            raise ValueError(
                "DBSCANDetector.explain expects the same dataset size used during fit "
                f"(fit_n={self._fit_n}, got_n={len(df)})."
            )

        idx_label = idx
        if isinstance(idx, int) and 0 <= idx < len(df):
            idx_pos = idx
        elif idx in df.index:
            idx_pos = int(df.index.get_loc(idx))
        else:
            raise KeyError(f"Index {idx!r} not found in input DataFrame")

        predictions_all = self.predict(df)
        pred_row = predictions_all.loc[idx_label] if idx_label in predictions_all.index else predictions_all.iloc[idx_pos]
        
        # Find which cluster this record belongs to
        cluster_info = None
        for cluster in self._clusters:
            if idx_pos in cluster.record_indices:
                cluster_info = {
                    "cluster_id": cluster.cluster_id,
                    "cluster_size": len(cluster.record_indices),
                    "similarity_score": cluster.similarity_score,
                    "other_members": [i for i in cluster.record_indices if i != idx_pos][:5]
                }
                break
        
        # Get field-level similarity breakdown if in cluster
        field_similarities = {}
        if cluster_info and self._processed_df is not None:
            other_idx = cluster_info["other_members"][0] if cluster_info["other_members"] else None
            if other_idx is not None:
                for field in self.field_weights.keys():
                    norm_field = f"{field}_normalized"
                    if norm_field in self._processed_df.columns:
                        val1 = str(self._processed_df.iloc[idx_pos][norm_field])
                        val2 = str(self._processed_df.iloc[other_idx][norm_field])
                        sim = 1.0 - self.distance_fn(val1, val2)
                        field_similarities[field] = float(sim)
        
        return {
            "index": idx_label,
            "score": float(pred_row["score"]),
            "is_fraud": bool(pred_row["is_fraud"]),
            "reason": pred_row["reason"],
            "detector": self.name,
            "cluster_info": cluster_info,
            "field_similarities": field_similarities
        }
    
    def get_field_contributions(self, df: pd.DataFrame, idx: int) -> dict:
        """Get field-level similarity contributions for a record."""
        if self._processed_df is None:
            return {}
        
        # Find cluster members
        cluster_members = []
        for cluster in self._clusters:
            if idx in cluster.record_indices:
                cluster_members = [i for i in cluster.record_indices if i != idx]
                break
        
        if not cluster_members:
            return {}
        
        # Compute average similarity per field
        field_sims = {field: [] for field in self.field_weights.keys()}
        
        for other_idx in cluster_members:
            for field in self.field_weights.keys():
                norm_field = f"{field}_normalized"
                if norm_field in self._processed_df.columns:
                    val1 = str(self._processed_df.iloc[idx][norm_field])
                    val2 = str(self._processed_df.iloc[other_idx][norm_field])
                    sim = 1.0 - self.distance_fn(val1, val2)
                    field_sims[field].append(sim)
        
        return {field: np.mean(sims) if sims else 0.0 for field, sims in field_sims.items()}
    
    @property
    def clusters(self) -> list[SuspiciousCluster]:
        """Get detected clusters."""
        return self._clusters
    
    @property
    def labels(self) -> Optional[np.ndarray]:
        """Get cluster labels for each record."""
        return self._labels


if __name__ == "__main__":
    # Test the detector
    test_data = pd.DataFrame({
        "customer_id": ["C001", "C002", "C003", "C004", "C005"],
        "surname": ["Mueller", "Muller", "Smith", "Johnson", "Muller"],
        "first_name": ["Hans", "Hans", "John", "Jane", "Hans"],
        "address": ["Main St 1", "Main St 1", "Oak Ave 5", "Pine Rd 10", "Main St 1"],
        "email": ["hans.m@test.com", "hans.mueller@test.com", "john@test.com", 
                 "jane@test.com", "h.muller@test.com"],
        "iban": ["DE123", "DE123", "DE456", "DE789", "DE123"],
    })
    
    detector = DBSCANDetector(eps=0.4, min_samples=2, use_sparse=True)
    detector.fit(test_data)
    results = detector.predict(test_data)
    
    print("DBSCAN Results:")
    print(results)
    print(f"\nClusters found: {len(detector.clusters)}")
    for c in detector.clusters:
        print(f"  Cluster {c.cluster_id}: {c.customer_ids}")
