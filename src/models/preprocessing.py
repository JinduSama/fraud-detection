"""
Data Preprocessing Module.

Provides data cleaning and feature engineering for fraud detection,
including text normalization and similarity feature extraction.

TASK-005: Create src/models/preprocessing.py for data cleaning and feature engineering.
"""

import re
import unicodedata
from typing import Optional

import numpy as np
import pandas as pd
import jellyfish


class DataPreprocessor:
    """
    Preprocessor for customer data before fraud detection.
    
    Handles text normalization, cleaning, and feature engineering
    for string similarity-based fraud detection.
    """
    
    # Columns to preprocess for similarity matching
    # Prefer structured address fields when present; keep `address` for backward compatibility.
    TEXT_COLUMNS = [
        "surname",
        "first_name",
        "email",
        "address",
        "strasse",
        "hausnummer",
        "plz",
        "stadt",
    ]
    
    def __init__(self):
        """Initialize the preprocessor."""
        self._processed_df: Optional[pd.DataFrame] = None
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text for comparison.
        
        Operations:
        - Convert to lowercase
        - Remove accents/diacritics
        - Remove special characters
        - Normalize whitespace
        
        Args:
            text: Input text to normalize.
            
        Returns:
            Normalized text string.
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Normalize unicode (decompose accents)
        text = unicodedata.normalize('NFKD', text)
        
        # Remove diacritical marks
        text = ''.join(c for c in text if not unicodedata.combining(c))
        
        # Remove special characters except alphanumeric and space
        text = re.sub(r'[^a-z0-9\s]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    @staticmethod
    def extract_email_components(email: str) -> tuple[str, str]:
        """
        Extract local part and domain from email.
        
        Args:
            email: Email address string.
            
        Returns:
            Tuple of (local_part, domain).
        """
        if pd.isna(email) or '@' not in str(email):
            return "", ""
        
        parts = str(email).lower().split('@')
        if len(parts) == 2:
            return parts[0], parts[1]
        return "", ""
    
    @staticmethod
    def get_soundex(text: str) -> str:
        """
        Get Soundex phonetic encoding of text.
        
        Args:
            text: Input text.
            
        Returns:
            Soundex code.
        """
        if not text or pd.isna(text):
            return ""
        
        # Clean text first
        clean = DataPreprocessor.normalize_text(text)
        words = clean.split()
        
        if not words:
            return ""
        
        # Get soundex for first word (typically the name)
        try:
            return jellyfish.soundex(words[0])
        except Exception:
            return ""
    
    @staticmethod
    def get_metaphone(text: str) -> str:
        """
        Get Metaphone phonetic encoding of text.
        
        Args:
            text: Input text.
            
        Returns:
            Metaphone code.
        """
        if not text or pd.isna(text):
            return ""
        
        clean = DataPreprocessor.normalize_text(text)
        words = clean.split()
        
        if not words:
            return ""
        
        try:
            return jellyfish.metaphone(words[0])
        except Exception:
            return ""
    
    @staticmethod
    def get_ngrams(text: str, n: int = 2) -> set[str]:
        """
        Extract character n-grams from text.
        
        Args:
            text: Input text.
            n: Size of n-grams (default: 2 for bigrams).
            
        Returns:
            Set of n-gram strings.
        """
        if not text or len(text) < n:
            return set()
        
        return {text[i:i+n] for i in range(len(text) - n + 1)}
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the entire DataFrame for fraud detection.
        
        Creates normalized versions of text columns and adds
        phonetic encodings for name matching.
        
        Args:
            df: Input DataFrame with customer data.
            
        Returns:
            Preprocessed DataFrame with additional features.
        """
        processed = df.copy()

        # If a full `address` column is missing, derive it from structured parts.
        if "address" not in processed.columns:
            has_parts = all(c in processed.columns for c in ["strasse", "hausnummer", "plz", "stadt"])
            if has_parts:
                def _fmt(row: pd.Series) -> str:
                    street_part = " ".join(
                        p for p in [str(row.get("strasse", "") or "").strip(), str(row.get("hausnummer", "") or "").strip()]
                        if p
                    )
                    city_part = " ".join(
                        p for p in [str(row.get("plz", "") or "").strip(), str(row.get("stadt", "") or "").strip()]
                        if p
                    )
                    if street_part and city_part:
                        return f"{street_part}, {city_part}"
                    return street_part or city_part

                processed["address"] = processed.apply(_fmt, axis=1)
        
        # Normalize text columns
        for col in self.TEXT_COLUMNS:
            if col in processed.columns:
                processed[f"{col}_normalized"] = processed[col].apply(
                    self.normalize_text
                )
        
        # Add phonetic encodings for names
        if "surname" in processed.columns:
            processed["surname_soundex"] = processed["surname"].apply(
                self.get_soundex
            )
            processed["surname_metaphone"] = processed["surname"].apply(
                self.get_metaphone
            )
        
        if "first_name" in processed.columns:
            processed["first_name_soundex"] = processed["first_name"].apply(
                self.get_soundex
            )
            processed["first_name_metaphone"] = processed["first_name"].apply(
                self.get_metaphone
            )
        
        # Extract email components
        if "email" in processed.columns:
            email_parts = processed["email"].apply(
                lambda x: pd.Series(self.extract_email_components(x))
            )
            processed["email_local"] = email_parts[0]
            processed["email_domain"] = email_parts[1]
        
        self._processed_df = processed
        return processed
    
    def create_blocking_key(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create blocking keys to reduce comparison space.
        
        Blocking groups records that might be similar together,
        reducing O(n^2) comparisons to manageable subsets.
        
        Blocking strategies:
        - Same first letter of surname
        - Same soundex for surname
        - Same email domain
        - Same IBAN prefix (country code)
        
        Args:
            df: Preprocessed DataFrame.
            
        Returns:
            DataFrame with blocking key columns added.
        """
        blocked = df.copy()
        
        # Surname initial blocking
        if "surname_normalized" in blocked.columns:
            blocked["block_surname_init"] = blocked["surname_normalized"].str[:2]
        
        # Soundex blocking
        if "surname_soundex" in blocked.columns:
            blocked["block_surname_soundex"] = blocked["surname_soundex"]
        
        # Email domain blocking
        if "email_domain" in blocked.columns:
            blocked["block_email_domain"] = blocked["email_domain"]
        
        # IBAN country code blocking
        if "iban" in blocked.columns:
            blocked["block_iban_country"] = blocked["iban"].str[:2].str.upper()
        
        return blocked
    
    def get_feature_matrix(
        self, 
        df: pd.DataFrame, 
        columns: Optional[list[str]] = None
    ) -> np.ndarray:
        """
        Create a feature matrix from normalized text columns.
        
        This can be used for clustering algorithms that require
        numeric input.
        
        Args:
            df: Preprocessed DataFrame.
            columns: Columns to include (default: all normalized columns).
            
        Returns:
            NumPy array with encoded features.
        """
        if columns is None:
            columns = [c for c in df.columns if c.endswith("_normalized")]
        
        # For now, we'll return the DataFrame subset
        # Actual feature encoding would depend on the clustering approach
        return df[columns].values


if __name__ == "__main__":
    # Quick test
    test_data = pd.DataFrame({
        "surname": ["Mueller", "Mueller", "MUELLER", "Miller"],
        "first_name": ["Hans", "Hans", "Hans", "John"],
        "email": ["hans.mueller@gmail.com", "hans.mueller@gmail.com", 
                  "hans.muller@gmail.com", "john.miller@yahoo.com"],
        "address": ["Hauptstrasse 123", "Hauptstrasse 123", 
                   "Hauptstr. 123", "Main St 456"],
        "iban": ["DE89370400440532013000", "DE89370400440532013000",
                "DE89370400440532013001", "GB82WEST12345698765432"],
    })
    
    preprocessor = DataPreprocessor()
    processed = preprocessor.preprocess_dataframe(test_data)
    blocked = preprocessor.create_blocking_key(processed)
    
    print("Original vs Normalized:")
    print(blocked[["surname", "surname_normalized", "surname_soundex"]].to_string())
