"""FAISS-based similarity index for fast approximate nearest neighbor search.

This module provides efficient similarity search against a database of historical
records using TF-IDF vectorization and FAISS indexing.

Features:
- Character n-gram TF-IDF for fuzzy string matching
- FAISS IndexFlatIP for exact inner product similarity
- Incremental index updates (add new records without rebuilding)
- Persistence (save/load index to disk)
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


@dataclass
class SimilarRecord:
    """A record similar to the query."""
    customer_id: str
    similarity: float
    record_index: int
    matched_text: str


class SimilarityIndex:
    """FAISS-based similarity index for fast record matching.
    
    Uses TF-IDF on character n-grams to create dense vectors, then uses
    FAISS for efficient similarity search.
    
    Example:
        index = SimilarityIndex()
        index.build(historical_df)
        
        similar = index.find_similar(new_record, top_k=5)
        for match in similar:
            print(f"{match.customer_id}: {match.similarity:.2f}")
    """
    
    # Fields to concatenate for similarity matching
    DEFAULT_FIELDS = [
        "surname", "first_name", "email", "iban",
        "street", "postal_code", "city", "date_of_birth",
    ]
    
    def __init__(
        self,
        fields: Optional[list[str]] = None,
        ngram_range: tuple[int, int] = (2, 4),
        max_features: int = 10000,
        use_gpu: bool = False,
    ):
        """Initialize the similarity index.
        
        Args:
            fields: List of field names to include in similarity matching.
            ngram_range: Character n-gram range for TF-IDF.
            max_features: Maximum number of TF-IDF features.
            use_gpu: Whether to use GPU acceleration (requires faiss-gpu).
        """
        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS is required for SimilarityIndex. "
                "Install with: pip install faiss-cpu"
            )
        
        self.fields = fields or self.DEFAULT_FIELDS
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.use_gpu = use_gpu
        
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.index: Optional[faiss.Index] = None
        self.customer_ids: list[str] = []
        self.record_texts: list[str] = []
        self._dimension: int = 0
        self._is_trained: bool = False
    
    def _record_to_text(self, record: dict | pd.Series) -> str:
        """Convert a record to a single text string for vectorization."""
        if isinstance(record, pd.Series):
            record = record.to_dict()
        
        parts = []
        for field in self.fields:
            value = record.get(field, "")
            if value and pd.notna(value):
                # Normalize the value
                text = str(value).strip().lower()
                # Add field marker to help distinguish fields
                parts.append(f"{field}:{text}")
        
        return " ".join(parts)
    
    def build(self, df: pd.DataFrame, customer_id_col: str = "customer_id") -> "SimilarityIndex":
        """Build the similarity index from a DataFrame.
        
        Args:
            df: DataFrame with customer records.
            customer_id_col: Name of the customer ID column.
            
        Returns:
            Self for method chaining.
        """
        # Convert records to text
        texts = []
        self.customer_ids = []
        
        for _, row in df.iterrows():
            text = self._record_to_text(row)
            texts.append(text)
            self.customer_ids.append(str(row.get(customer_id_col, f"idx_{len(self.customer_ids)}")))
        
        self.record_texts = texts
        
        # Fit TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            lowercase=True,
            dtype=np.float32,
        )
        
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Convert to dense numpy array and normalize for cosine similarity
        vectors = tfidf_matrix.toarray().astype(np.float32)
        faiss.normalize_L2(vectors)
        
        self._dimension = vectors.shape[1]
        
        # Build FAISS index
        self.index = faiss.IndexFlatIP(self._dimension)  # Inner product = cosine after L2 norm
        self.index.add(vectors)
        
        self._is_trained = True
        
        return self
    
    def add(self, records: pd.DataFrame, customer_id_col: str = "customer_id") -> int:
        """Add new records to the index without rebuilding.
        
        Args:
            records: DataFrame with new records to add.
            customer_id_col: Name of the customer ID column.
            
        Returns:
            Number of records added.
        """
        if not self._is_trained:
            raise RuntimeError("Index must be built before adding records. Call build() first.")
        
        texts = []
        new_ids = []
        
        for _, row in records.iterrows():
            text = self._record_to_text(row)
            texts.append(text)
            new_ids.append(str(row.get(customer_id_col, f"idx_{len(self.customer_ids) + len(new_ids)}")))
        
        # Transform using fitted vectorizer
        tfidf_matrix = self.vectorizer.transform(texts)
        vectors = tfidf_matrix.toarray().astype(np.float32)
        faiss.normalize_L2(vectors)
        
        # Add to index
        self.index.add(vectors)
        self.customer_ids.extend(new_ids)
        self.record_texts.extend(texts)
        
        return len(texts)
    
    def find_similar(
        self,
        record: dict | pd.Series,
        top_k: int = 10,
        min_similarity: float = 0.3,
    ) -> list[SimilarRecord]:
        """Find similar records in the index.
        
        Args:
            record: Query record to find matches for.
            top_k: Maximum number of similar records to return.
            min_similarity: Minimum similarity threshold (0-1).
            
        Returns:
            List of SimilarRecord objects, sorted by similarity descending.
        """
        if not self._is_trained:
            raise RuntimeError("Index must be built before searching. Call build() first.")
        
        # Convert query to vector
        query_text = self._record_to_text(record)
        query_vector = self.vectorizer.transform([query_text]).toarray().astype(np.float32)
        faiss.normalize_L2(query_vector)
        
        # Search
        similarities, indices = self.index.search(query_vector, top_k)
        
        # Filter and format results
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for not found
                continue
            if sim < min_similarity:
                continue
            
            results.append(SimilarRecord(
                customer_id=self.customer_ids[idx],
                similarity=float(sim),
                record_index=int(idx),
                matched_text=self.record_texts[idx],
            ))
        
        return results
    
    def find_similar_batch(
        self,
        records: pd.DataFrame,
        top_k: int = 10,
        min_similarity: float = 0.3,
    ) -> list[list[SimilarRecord]]:
        """Find similar records for multiple queries.
        
        Args:
            records: DataFrame with query records.
            top_k: Maximum matches per query.
            min_similarity: Minimum similarity threshold.
            
        Returns:
            List of result lists, one per query record.
        """
        if not self._is_trained:
            raise RuntimeError("Index must be built before searching. Call build() first.")
        
        # Convert all queries to vectors
        texts = [self._record_to_text(row) for _, row in records.iterrows()]
        query_vectors = self.vectorizer.transform(texts).toarray().astype(np.float32)
        faiss.normalize_L2(query_vectors)
        
        # Batch search
        similarities, indices = self.index.search(query_vectors, top_k)
        
        # Format results
        all_results = []
        for query_idx in range(len(texts)):
            results = []
            for sim, idx in zip(similarities[query_idx], indices[query_idx]):
                if idx < 0 or sim < min_similarity:
                    continue
                results.append(SimilarRecord(
                    customer_id=self.customer_ids[idx],
                    similarity=float(sim),
                    record_index=int(idx),
                    matched_text=self.record_texts[idx],
                ))
            all_results.append(results)
        
        return all_results
    
    def save(self, directory: str | Path) -> None:
        """Save the index and vectorizer to disk.
        
        Args:
            directory: Directory to save files to.
        """
        if not self._is_trained:
            raise RuntimeError("Cannot save untrained index.")
        
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(directory / "faiss_index.bin"))
        
        # Save vectorizer and metadata
        joblib.dump({
            "vectorizer": self.vectorizer,
            "customer_ids": self.customer_ids,
            "record_texts": self.record_texts,
            "fields": self.fields,
            "dimension": self._dimension,
            "ngram_range": self.ngram_range,
            "max_features": self.max_features,
        }, directory / "similarity_metadata.joblib")
    
    @classmethod
    def load(cls, directory: str | Path) -> "SimilarityIndex":
        """Load a saved index from disk.
        
        Args:
            directory: Directory containing saved files.
            
        Returns:
            Loaded SimilarityIndex instance.
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is required. Install with: pip install faiss-cpu")
        
        directory = Path(directory)
        
        # Load metadata
        metadata = joblib.load(directory / "similarity_metadata.joblib")
        
        # Create instance
        instance = cls(
            fields=metadata["fields"],
            ngram_range=metadata["ngram_range"],
            max_features=metadata["max_features"],
        )
        
        # Restore state
        instance.vectorizer = metadata["vectorizer"]
        instance.customer_ids = metadata["customer_ids"]
        instance.record_texts = metadata["record_texts"]
        instance._dimension = metadata["dimension"]
        
        # Load FAISS index
        instance.index = faiss.read_index(str(directory / "faiss_index.bin"))
        instance._is_trained = True
        
        return instance
    
    @property
    def size(self) -> int:
        """Number of records in the index."""
        return len(self.customer_ids)
    
    @property
    def is_trained(self) -> bool:
        """Whether the index has been built."""
        return self._is_trained
    
    def __len__(self) -> int:
        return self.size
    
    def __repr__(self) -> str:
        status = "trained" if self._is_trained else "untrained"
        return f"SimilarityIndex({status}, size={self.size}, dim={self._dimension})"
