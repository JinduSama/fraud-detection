"""
Address processing utilities for fraud detection.

Provides address normalization and comparison functions used
across multiple modules for consistent address handling.
"""

import re
from typing import Optional

import pandas as pd


def normalize_address(
    strasse: Optional[str] = None,
    hausnummer: Optional[str] = None,
    plz: Optional[str] = None,
    stadt: Optional[str] = None,
    address: Optional[str] = None,
) -> str:
    """
    Normalize address fields into a consistent format for comparison.
    
    Supports two modes:
    1. Structured fields: strasse, hausnummer, plz, stadt (preferred)
    2. Single address string (fallback)
    
    The normalization process:
    - Converts to lowercase
    - Strips whitespace
    - Joins components with pipe separator for grouping
    
    Args:
        strasse: Street name.
        hausnummer: House number.
        plz: Postal code.
        stadt: City name.
        address: Full address string (used if structured fields are empty).
        
    Returns:
        Normalized address string suitable for comparison/grouping.
        
    Example:
        >>> normalize_address(strasse="Hauptstraße", hausnummer="42", plz="12345", stadt="Berlin")
        'hauptstraße|42|12345|berlin'
        >>> normalize_address(address="Hauptstraße 42, 12345 Berlin")
        'hauptstraße 42, 12345 berlin'
    """
    def safe_str(val: Optional[str]) -> str:
        if pd.isna(val) or val is None:
            return ""
        return str(val).strip()
    
    strasse_s = safe_str(strasse)
    hausnummer_s = safe_str(hausnummer)
    plz_s = safe_str(plz)
    stadt_s = safe_str(stadt)
    
    # Prefer structured address if any component is present
    if any([strasse_s, hausnummer_s, plz_s, stadt_s]):
        return (
            f"{strasse_s.lower()}|"
            f"{hausnummer_s.lower()}|"
            f"{plz_s.lower()}|"
            f"{stadt_s.lower()}"
        )
    
    # Fallback to full address string
    address_s = safe_str(address)
    return address_s.lower()


def format_address(
    strasse: Optional[str] = None,
    hausnummer: Optional[str] = None,
    plz: Optional[str] = None,
    stadt: Optional[str] = None,
) -> str:
    """
    Format structured address fields into a human-readable string.
    
    Combines address components into a standard format:
    "Strasse Hausnummer, PLZ Stadt"
    
    Args:
        strasse: Street name.
        hausnummer: House number.
        plz: Postal code.
        stadt: City name.
        
    Returns:
        Formatted address string for display.
        
    Example:
        >>> format_address("Hauptstraße", "42", "12345", "Berlin")
        'Hauptstraße 42, 12345 Berlin'
    """
    def safe_str(val: Optional[str]) -> str:
        if pd.isna(val) or val is None:
            return ""
        return str(val).strip()
    
    street_part = " ".join(
        p for p in [safe_str(strasse), safe_str(hausnummer)] if p
    )
    city_part = " ".join(
        p for p in [safe_str(plz), safe_str(stadt)] if p
    )
    
    if street_part and city_part:
        return f"{street_part}, {city_part}"
    return street_part or city_part


def normalize_address_from_row(row: pd.Series) -> str:
    """
    Extract and normalize address from a DataFrame row.
    
    Attempts to use structured address fields first, then falls back
    to the 'address' column if structured fields are not available.
    
    Args:
        row: DataFrame row containing address data.
        
    Returns:
        Normalized address string for comparison.
    """
    return normalize_address(
        strasse=row.get("strasse"),
        hausnummer=row.get("hausnummer"),
        plz=row.get("plz"),
        stadt=row.get("stadt"),
        address=row.get("address"),
    )


def get_address_text(row: pd.Series) -> str:
    """
    Get formatted address text from a DataFrame row.
    
    Prefers structured address fields; falls back to legacy 'address' column.
    
    Args:
        row: DataFrame row containing address data.
        
    Returns:
        Human-readable address string.
    """
    def safe_str(val) -> str:
        if pd.isna(val):
            return ""
        return str(val).strip()
    
    strasse = safe_str(row.get("strasse", ""))
    hausnummer = safe_str(row.get("hausnummer", ""))
    plz = safe_str(row.get("plz", ""))
    stadt = safe_str(row.get("stadt", ""))
    
    if any([strasse, hausnummer, plz, stadt]):
        return format_address(strasse, hausnummer, plz, stadt)
    
    return safe_str(row.get("address", ""))
