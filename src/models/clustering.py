"""
Fraud Detection Clustering Module.

Implements DBSCAN-based clustering with custom string distance metrics
for identifying suspicious patterns in customer data.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from .preprocessing import DataPreprocessor
from ..utils.text import StringDistanceMetrics


@dataclass
class SuspiciousCluster:
    """Represents a cluster of suspicious records."""
    
    cluster_id: int
    record_indices: list[int]
    customer_ids: list[str]
    similarity_score: float
    detection_reason: str


class FraudDetector:
    """
    Fraud detection using clustering and string similarity.
    
    Uses a multi-field similarity approach with DBSCAN clustering
    to identify groups of similar/suspicious records.
    """
    
    # Field weights for composite similarity calculation
    DEFAULT_FIELD_WEIGHTS = {
        "surname": 0.20,
        "first_name": 0.16,
        "email": 0.12,
        "iban": 0.17,
        # Prefer structured address fields when present
        "strasse": 0.08,
        "hausnummer": 0.04,
        "plz": 0.03,
        "stadt": 0.02,
        # Fallback when structured fields are missing
        "address": 0.18,
    }
    
    def __init__(
        self,
        eps: float = 0.3,
        min_samples: int = 2,
        field_weights: Optional[dict[str, float]] = None,
        distance_metric: str = "jaro_winkler"
    ):
        """
        Initialize the fraud detector.
        
        Args:
            eps: DBSCAN epsilon (max distance between samples in a cluster).
            min_samples: Minimum samples to form a cluster.
            field_weights: Weights for each field in similarity calculation.
            distance_metric: Which distance metric to use 
                            ('jaro_winkler', 'levenshtein', 'damerau').
        """
        self.eps = eps
        self.min_samples = min_samples
        self.field_weights = field_weights or self.DEFAULT_FIELD_WEIGHTS
        
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
        self._distance_matrix: Optional[np.ndarray] = None
        self._clusters: list[SuspiciousCluster] = []
    
    def _compute_record_distance(
        self, 
        row1: pd.Series, 
        row2: pd.Series
    ) -> float:
        """
        Compute weighted distance between two records.
        
        Args:
            row1: First record.
            row2: Second record.
            
        Returns:
            Weighted composite distance (0-1).
        """
        total_distance = 0.0
        total_weight = 0.0

        has_structured_address = all(
            f"{c}_normalized" in row1.index and f"{c}_normalized" in row2.index
            for c in ["strasse", "hausnummer", "plz", "stadt"]
        )
        
        for field, weight in self.field_weights.items():
            if field == "address" and has_structured_address:
                continue
            norm_field = f"{field}_normalized"
            
            if norm_field in row1.index and norm_field in row2.index:
                val1 = str(row1[norm_field]) if pd.notna(row1[norm_field]) else ""
                val2 = str(row2[norm_field]) if pd.notna(row2[norm_field]) else ""
                
                # Special handling for IBAN - exact match is very significant
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
    
    def compute_distance_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute pairwise distance matrix for all records.
        
        Warning: This is O(n^2) - use blocking for large datasets.
        
        Args:
            df: Preprocessed DataFrame with normalized columns.
            
        Returns:
            NxN distance matrix.
        """
        n = len(df)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self._compute_record_distance(df.iloc[i], df.iloc[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist  # Symmetric
        
        self._distance_matrix = distance_matrix
        return distance_matrix
    
    def compute_distance_matrix_blocked(
        self, 
        df: pd.DataFrame,
        block_columns: list[str] = None
    ) -> np.ndarray:
        """
        Compute distance matrix using multiple blocking strategies.
        
        Uses multiple blocking keys to ensure records that might be 
        related through different fields (e.g., same IBAN vs same name)
        are compared to each other.
        
        Args:
            df: Preprocessed DataFrame with blocking keys.
            block_columns: Columns to use for blocking. Uses multiple strategies by default.
            
        Returns:
            NxN distance matrix.
        """
        if block_columns is None:
            block_columns = [
                "block_surname_soundex",
                "block_iban_country",
                "block_email_domain",
            ]
        
        n = len(df)
        distance_matrix = np.ones((n, n))  # Default to max distance
        np.fill_diagonal(distance_matrix, 0)  # Self-distance is 0
        
        # Track which pairs we've already compared
        compared_pairs = set()
        
        for block_column in block_columns:
            if block_column not in df.columns:
                continue
            
            # Group by block key
            blocks = df.groupby(block_column).indices
            
            for block_key, indices in blocks.items():
                if len(indices) < 2:
                    continue
                
                # Compare within block
                indices_list = list(indices)
                for i in range(len(indices_list)):
                    for j in range(i + 1, len(indices_list)):
                        idx_i = indices_list[i]
                        idx_j = indices_list[j]
                        
                        # Skip if already compared
                        pair = (min(idx_i, idx_j), max(idx_i, idx_j))
                        if pair in compared_pairs:
                            continue
                        compared_pairs.add(pair)
                        
                        dist = self._compute_record_distance(
                            df.iloc[idx_i], 
                            df.iloc[idx_j]
                        )
                        distance_matrix[idx_i, idx_j] = dist
                        distance_matrix[idx_j, idx_i] = dist
        
        # Also check for exact IBAN matches (critical for shared_iban fraud)
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
                        pair = (min(idx_i, idx_j), max(idx_i, idx_j))
                        if pair in compared_pairs:
                            continue
                        compared_pairs.add(pair)
                        dist = self._compute_record_distance(
                            df.iloc[idx_i], 
                            df.iloc[idx_j]
                        )
                        distance_matrix[idx_i, idx_j] = dist
                        distance_matrix[idx_j, idx_i] = dist
        
        self._distance_matrix = distance_matrix
        return distance_matrix
    
    def detect_clusters(
        self, 
        df: pd.DataFrame,
        use_blocking: bool = True
    ) -> list[SuspiciousCluster]:
        """
        Detect suspicious clusters in the dataset.
        
        Args:
            df: Input DataFrame (will be preprocessed).
            use_blocking: Whether to use blocking to speed up computation.
            
        Returns:
            List of SuspiciousCluster objects.
        """
        # Preprocess data
        print("  - Preprocessing data...")
        processed = self.preprocessor.preprocess_dataframe(df)
        processed = self.preprocessor.create_blocking_key(processed)
        
        # Compute distance matrix
        print("  - Computing distance matrix...")
        if use_blocking:
            distance_matrix = self.compute_distance_matrix_blocked(processed)
        else:
            distance_matrix = self.compute_distance_matrix(processed)
        
        # Run DBSCAN
        print(f"  - Running DBSCAN (eps={self.eps}, min_samples={self.min_samples})...")
        dbscan = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric="precomputed"
        )
        cluster_labels = dbscan.fit_predict(distance_matrix)
        
        # Extract clusters (ignore noise label -1)
        clusters = []
        unique_labels = set(cluster_labels)
        unique_labels.discard(-1)  # Remove noise label
        
        for label in unique_labels:
            indices = np.where(cluster_labels == label)[0].tolist()
            
            if len(indices) < 2:
                continue
            
            # Get customer IDs
            customer_ids = df.iloc[indices]["customer_id"].tolist()
            
            # Calculate average similarity within cluster
            avg_distance = 0.0
            count = 0
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    avg_distance += distance_matrix[indices[i], indices[j]]
                    count += 1
            
            if count > 0:
                avg_distance /= count
            
            avg_similarity = 1.0 - avg_distance
            
            # Determine detection reason
            detection_reason = self._infer_detection_reason(
                df.iloc[indices], 
                processed.iloc[indices]
            )
            
            cluster = SuspiciousCluster(
                cluster_id=int(label),
                record_indices=indices,
                customer_ids=customer_ids,
                similarity_score=avg_similarity,
                detection_reason=detection_reason
            )
            clusters.append(cluster)
        
        self._clusters = clusters
        print(f"  - Found {len(clusters)} suspicious clusters")
        
        return clusters
    
    def _infer_detection_reason(
        self, 
        original_df: pd.DataFrame,
        processed_df: pd.DataFrame
    ) -> str:
        """
        Infer why records were clustered together.
        
        Args:
            original_df: Original records in cluster.
            processed_df: Preprocessed records.
            
        Returns:
            Human-readable detection reason.
        """
        reasons = []
        
        # Check for shared IBAN
        if "iban" in original_df.columns:
            unique_ibans = original_df["iban"].nunique()
            if unique_ibans < len(original_df):
                reasons.append("shared_iban")
        
        # Check for same address
        if "address_normalized" in processed_df.columns:
            unique_addresses = processed_df["address_normalized"].nunique()
            if unique_addresses < len(processed_df):
                reasons.append("same_address")
        
        # Check for similar names
        if "surname_soundex" in processed_df.columns:
            unique_soundex = processed_df["surname_soundex"].nunique()
            if unique_soundex == 1:
                reasons.append("similar_names")
        
        # Check for similar emails
        if "email_local" in processed_df.columns:
            local_parts = processed_df["email_local"].tolist()
            if len(set(local_parts)) < len(local_parts):
                reasons.append("similar_emails")
        
        if not reasons:
            reasons.append("high_overall_similarity")
        
        return ", ".join(reasons)
    
    def get_flagged_records(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get DataFrame with fraud flags based on detected clusters.
        
        Also flags records with direct fraud indicators:
        - Shared IBAN with different customer
        - Same address with different name
        
        Args:
            df: Original DataFrame.
            
        Returns:
            DataFrame with additional columns:
            - detected_fraud: Boolean flag
            - cluster_id: Assigned cluster (-1 if not in cluster)
            - detection_reason: Why flagged
        """
        result = df.copy()
        result["detected_fraud"] = False
        result["detected_cluster_id"] = -1
        result["detection_reason"] = ""
        
        # Flag clusters from DBSCAN
        for cluster in self._clusters:
            for idx in cluster.record_indices:
                result.loc[idx, "detected_fraud"] = True
                result.loc[idx, "detected_cluster_id"] = cluster.cluster_id
                result.loc[idx, "detection_reason"] = cluster.detection_reason
        
        # Direct detection: Shared IBAN (different customers with same IBAN)
        if "iban" in result.columns:
            iban_counts = result["iban"].value_counts()
            shared_ibans = iban_counts[iban_counts > 1].index.tolist()
            
            for iban in shared_ibans:
                mask = result["iban"] == iban
                indices = result[mask].index.tolist()
                for idx in indices:
                    if not result.loc[idx, "detected_fraud"]:
                        result.loc[idx, "detected_fraud"] = True
                        result.loc[idx, "detection_reason"] = "shared_iban"
                    elif "shared_iban" not in result.loc[idx, "detection_reason"]:
                        result.loc[idx, "detection_reason"] += ", shared_iban"
        
        # Direct detection: Same address (different customers at same address)
        if "address" in result.columns:
            addr_counts = result["address"].value_counts()
            shared_addrs = addr_counts[addr_counts > 1].index.tolist()
            
            for addr in shared_addrs:
                mask = result["address"] == addr
                indices = result[mask].index.tolist()
                for idx in indices:
                    if not result.loc[idx, "detected_fraud"]:
                        result.loc[idx, "detected_fraud"] = True
                        result.loc[idx, "detection_reason"] = "same_address"
                    elif "same_address" not in result.loc[idx, "detection_reason"]:
                        result.loc[idx, "detection_reason"] += ", same_address"
        
        return result


if __name__ == "__main__":
    # Test with sample data
    test_data = pd.DataFrame({
        "customer_id": ["C001", "C002", "C003", "C004", "C005"],
        "surname": ["Mueller", "Muller", "Smith", "Johnson", "Muller"],
        "first_name": ["Hans", "Hans", "John", "Jane", "Hans"],
        "address": ["Main St 1", "Main St 1", "Oak Ave 5", "Pine Rd 10", "Main St 1"],
        "email": ["hans.m@test.com", "hans.mueller@test.com", "john@test.com", 
                 "jane@test.com", "h.muller@test.com"],
        "iban": ["DE123", "DE123", "DE456", "DE789", "DE123"],
        "is_fraud": [False, True, False, False, True],
    })
    
    detector = FraudDetector(eps=0.4, min_samples=2)
    clusters = detector.detect_clusters(test_data, use_blocking=False)
    
    print(f"\nFound {len(clusters)} clusters:")
    for c in clusters:
        print(f"  Cluster {c.cluster_id}: {c.customer_ids} ({c.detection_reason})")
