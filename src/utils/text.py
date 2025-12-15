"""
Text processing utilities for fraud detection.

Provides string normalization and distance metrics used across
multiple modules for consistent text comparison.
"""

import re
import unicodedata
from typing import Optional

import jellyfish


def normalize_text(text: str) -> str:
    """
    Normalize text for consistent comparison.
    
    Performs the following transformations:
    - Converts to lowercase
    - Normalizes Unicode characters (NFKD normalization)
    - Removes diacritics/combining characters
    - Removes special characters (keeps only alphanumeric and spaces)
    - Collapses multiple whitespaces
    
    Args:
        text: Input string to normalize.
        
    Returns:
        Normalized string, or empty string if input is None/NaN.
        
    Example:
        >>> normalize_text("Müller-Schmidt")
        'muller schmidt'
    """
    import pandas as pd
    
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = ' '.join(text.split())
    
    return text


class StringDistanceMetrics:
    """
    Collection of string distance/similarity metrics for fraud detection.
    
    All methods return a distance value where:
    - 0.0 = identical strings
    - 1.0 = completely different strings
    
    These metrics are useful for detecting:
    - Typos and minor variations in names
    - Intentional obfuscation attempts
    - Related records with slight differences
    """
    
    @staticmethod
    def jaro_winkler_distance(s1: str, s2: str) -> float:
        """
        Jaro-Winkler distance between two strings.
        
        Particularly good for detecting typos and minor variations in names.
        Gives more weight to prefix matches, making it ideal for comparing
        names where the beginning is more likely to be correct.
        
        Args:
            s1: First string to compare.
            s2: Second string to compare.
            
        Returns:
            Distance value from 0.0 (identical) to 1.0 (completely different).
            
        Example:
            >>> StringDistanceMetrics.jaro_winkler_distance("MARTHA", "MARHTA")
            0.039  # Very similar (transposition)
        """
        if not s1 or not s2:
            return 1.0
        similarity = jellyfish.jaro_winkler_similarity(s1, s2)
        return 1.0 - similarity
    
    @staticmethod
    def levenshtein_distance_normalized(s1: str, s2: str) -> float:
        """
        Normalized Levenshtein distance (0-1 range).
        
        Calculates the minimum number of single-character edits (insertions,
        deletions, substitutions) needed to transform s1 into s2, normalized
        by the maximum string length.
        
        Args:
            s1: First string to compare.
            s2: Second string to compare.
            
        Returns:
            Normalized distance from 0.0 (identical) to 1.0 (completely different).
            
        Example:
            >>> StringDistanceMetrics.levenshtein_distance_normalized("kitten", "sitting")
            0.43  # 3 edits / 7 max length
        """
        if not s1 and not s2:
            return 0.0
        if not s1 or not s2:
            return 1.0
        
        distance = jellyfish.levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        return distance / max_len
    
    @staticmethod
    def damerau_levenshtein_normalized(s1: str, s2: str) -> float:
        """
        Normalized Damerau-Levenshtein distance.
        
        Like Levenshtein but also considers transpositions (swapping two
        adjacent characters) as a single edit. This makes it better for
        detecting typing errors where characters are accidentally swapped.
        
        Args:
            s1: First string to compare.
            s2: Second string to compare.
            
        Returns:
            Normalized distance from 0.0 (identical) to 1.0 (completely different).
            
        Example:
            >>> StringDistanceMetrics.damerau_levenshtein_normalized("CA", "AC")
            0.5  # 1 transposition / 2 max length
        """
        if not s1 and not s2:
            return 0.0
        if not s1 or not s2:
            return 1.0
        
        distance = jellyfish.damerau_levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        return distance / max_len
    
    @staticmethod
    def get_similarity(s1: str, s2: str, method: str = "jaro_winkler") -> float:
        """
        Get similarity score between two strings (inverse of distance).
        
        Convenience method that returns similarity instead of distance.
        
        Args:
            s1: First string to compare.
            s2: Second string to compare.
            method: Distance method to use ('jaro_winkler', 'levenshtein', 'damerau').
            
        Returns:
            Similarity value from 0.0 (completely different) to 1.0 (identical).
        """
        methods = {
            "jaro_winkler": StringDistanceMetrics.jaro_winkler_distance,
            "levenshtein": StringDistanceMetrics.levenshtein_distance_normalized,
            "damerau": StringDistanceMetrics.damerau_levenshtein_normalized,
        }
        distance_fn = methods.get(method, StringDistanceMetrics.jaro_winkler_distance)
        return 1.0 - distance_fn(s1, s2)
