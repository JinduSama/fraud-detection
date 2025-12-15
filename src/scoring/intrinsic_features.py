"""Intrinsic feature extraction for real-time fraud scoring.

These features can be computed from a single record WITHOUT comparing to other records,
enabling real-time scoring with <10ms latency.

Features include:
- IBAN validation and country matching
- Email pattern analysis (entropy, disposable domains, numeric content)
- Name quality indicators (entropy, digits, keyboard patterns)
- Address validation (postal code format)
- Cross-field consistency checks
"""

import math
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# Common disposable email domains
DISPOSABLE_EMAIL_DOMAINS = frozenset([
    "tempmail.com", "throwaway.email", "guerrillamail.com", "10minutemail.com",
    "mailinator.com", "trashmail.com", "fakeinbox.com", "tempinbox.com",
    "dispostable.com", "maildrop.cc", "yopmail.com", "getnada.com",
    "temp-mail.org", "emailondeck.com", "mohmal.com", "sharklasers.com",
    "spam4.me", "grr.la", "burnermail.io", "tempail.com",
])

# Keyboard patterns that suggest lazy/fake input
KEYBOARD_PATTERNS = [
    "qwerty", "qwertz", "asdf", "zxcv", "1234", "abcd", "aaaa", "1111",
    "qazwsx", "password", "12345", "123456", "test", "fake", "xxx",
]

# Valid postal code patterns by country (subset)
POSTAL_CODE_PATTERNS = {
    "DE": r"^\d{5}$",  # Germany: 5 digits
    "AT": r"^\d{4}$",  # Austria: 4 digits
    "CH": r"^\d{4}$",  # Switzerland: 4 digits
    "NL": r"^\d{4}\s?[A-Z]{2}$",  # Netherlands: 4 digits + 2 letters
    "BE": r"^\d{4}$",  # Belgium: 4 digits
    "FR": r"^\d{5}$",  # France: 5 digits
    "GB": r"^[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}$",  # UK: complex
    "US": r"^\d{5}(-\d{4})?$",  # US: 5 or 9 digits
}


@dataclass
class IntrinsicFeatures:
    """Container for intrinsic features extracted from a single record."""
    
    # IBAN features
    iban_valid: bool
    iban_country: str
    iban_country_matches_address: bool
    
    # Email features
    email_entropy: float
    email_is_disposable: bool
    email_numeric_ratio: float
    email_length: int
    email_domain: str
    email_has_plus_addressing: bool
    
    # Name features
    surname_entropy: float
    firstname_entropy: float
    name_has_digits: bool
    name_keyboard_pattern_score: float
    name_length_ratio: float  # surname / firstname
    
    # Address features
    postal_code_valid: bool
    address_entropy: float
    address_has_po_box: bool
    
    # Cross-field features
    email_contains_name: bool
    field_completeness: float  # ratio of non-empty fields
    
    # Aggregate scores
    total_flags: int  # count of suspicious indicators
    
    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame compatibility."""
        return {
            "iban_valid": self.iban_valid,
            "iban_country": self.iban_country,
            "iban_country_matches_address": self.iban_country_matches_address,
            "email_entropy": self.email_entropy,
            "email_is_disposable": self.email_is_disposable,
            "email_numeric_ratio": self.email_numeric_ratio,
            "email_length": self.email_length,
            "email_domain": self.email_domain,
            "email_has_plus_addressing": self.email_has_plus_addressing,
            "surname_entropy": self.surname_entropy,
            "firstname_entropy": self.firstname_entropy,
            "name_has_digits": self.name_has_digits,
            "name_keyboard_pattern_score": self.name_keyboard_pattern_score,
            "name_length_ratio": self.name_length_ratio,
            "postal_code_valid": self.postal_code_valid,
            "address_entropy": self.address_entropy,
            "address_has_po_box": self.address_has_po_box,
            "email_contains_name": self.email_contains_name,
            "field_completeness": self.field_completeness,
            "total_flags": self.total_flags,
        }
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            float(self.iban_valid),
            float(self.iban_country_matches_address),
            self.email_entropy,
            float(self.email_is_disposable),
            self.email_numeric_ratio,
            self.email_length,
            float(self.email_has_plus_addressing),
            self.surname_entropy,
            self.firstname_entropy,
            float(self.name_has_digits),
            self.name_keyboard_pattern_score,
            self.name_length_ratio,
            float(self.postal_code_valid),
            self.address_entropy,
            float(self.address_has_po_box),
            float(self.email_contains_name),
            self.field_completeness,
            self.total_flags,
        ])
    
    @staticmethod
    def feature_names() -> list[str]:
        """Return ordered list of feature names for array output."""
        return [
            "iban_valid",
            "iban_country_matches_address",
            "email_entropy",
            "email_is_disposable",
            "email_numeric_ratio",
            "email_length",
            "email_has_plus_addressing",
            "surname_entropy",
            "firstname_entropy",
            "name_has_digits",
            "name_keyboard_pattern_score",
            "name_length_ratio",
            "postal_code_valid",
            "address_entropy",
            "address_has_po_box",
            "email_contains_name",
            "field_completeness",
            "total_flags",
        ]


class IntrinsicFeatureExtractor:
    """Extract features from a single record without comparing to other records.
    
    This enables real-time scoring with minimal latency since no database
    lookups or similarity computations are required.
    
    Example:
        extractor = IntrinsicFeatureExtractor()
        features = extractor.extract(record)
        print(f"IBAN valid: {features.iban_valid}")
        print(f"Suspicious flags: {features.total_flags}")
    """
    
    def __init__(
        self,
        disposable_domains: Optional[set[str]] = None,
        keyboard_patterns: Optional[list[str]] = None,
        postal_patterns: Optional[dict[str, str]] = None,
    ):
        """Initialize the feature extractor.
        
        Args:
            disposable_domains: Set of known disposable email domains.
            keyboard_patterns: List of keyboard pattern strings to detect.
            postal_patterns: Dict of country code to postal code regex.
        """
        self.disposable_domains = disposable_domains or DISPOSABLE_EMAIL_DOMAINS
        self.keyboard_patterns = keyboard_patterns or KEYBOARD_PATTERNS
        self.postal_patterns = postal_patterns or POSTAL_CODE_PATTERNS
    
    def extract(self, record: dict | pd.Series) -> IntrinsicFeatures:
        """Extract intrinsic features from a single record.
        
        Args:
            record: Dictionary or Series with customer data fields.
            
        Returns:
            IntrinsicFeatures dataclass with all computed features.
        """
        if isinstance(record, pd.Series):
            record = record.to_dict()
        
        # Extract fields with defaults
        iban = str(record.get("iban", "")).strip().upper()
        email = str(record.get("email", "")).strip().lower()
        surname = str(record.get("surname", "")).strip()
        first_name = str(record.get("first_name", "")).strip()
        postal_code = str(record.get("postal_code", "")).strip()
        address = self._get_address(record)
        
        # Compute features
        iban_valid = self._validate_iban(iban)
        iban_country = iban[:2] if len(iban) >= 2 else ""
        address_country = self._infer_country_from_address(record)
        iban_country_matches = iban_country == address_country if address_country else True
        
        email_local, email_domain = self._parse_email(email)
        email_entropy = self._compute_entropy(email_local)
        email_is_disposable = email_domain in self.disposable_domains
        email_numeric_ratio = self._numeric_ratio(email_local)
        email_has_plus = "+" in email_local
        
        surname_entropy = self._compute_entropy(surname.lower())
        firstname_entropy = self._compute_entropy(first_name.lower())
        name_has_digits = bool(re.search(r"\d", surname + first_name))
        keyboard_score = self._keyboard_pattern_score(surname + first_name)
        name_length_ratio = len(surname) / max(len(first_name), 1)
        
        postal_valid = self._validate_postal_code(postal_code, iban_country)
        address_entropy = self._compute_entropy(address.lower())
        has_po_box = bool(re.search(r"\b(po|postfach|postbus)\s*(box)?\s*\d*\b", address.lower()))
        
        email_contains_name = self._email_contains_name(email_local, surname, first_name)
        field_completeness = self._compute_field_completeness(record)
        
        # Count flags
        flags = sum([
            not iban_valid,
            not iban_country_matches,
            email_is_disposable,
            email_numeric_ratio > 0.5,
            name_has_digits,
            keyboard_score > 0.3,
            not postal_valid and postal_code != "",
            has_po_box,
            email_entropy > 4.0,  # Very high entropy
            surname_entropy > 3.5,
        ])
        
        return IntrinsicFeatures(
            iban_valid=iban_valid,
            iban_country=iban_country,
            iban_country_matches_address=iban_country_matches,
            email_entropy=email_entropy,
            email_is_disposable=email_is_disposable,
            email_numeric_ratio=email_numeric_ratio,
            email_length=len(email),
            email_domain=email_domain,
            email_has_plus_addressing=email_has_plus,
            surname_entropy=surname_entropy,
            firstname_entropy=firstname_entropy,
            name_has_digits=name_has_digits,
            name_keyboard_pattern_score=keyboard_score,
            name_length_ratio=name_length_ratio,
            postal_code_valid=postal_valid,
            address_entropy=address_entropy,
            address_has_po_box=has_po_box,
            email_contains_name=email_contains_name,
            field_completeness=field_completeness,
            total_flags=flags,
        )
    
    def extract_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract intrinsic features for all records in a DataFrame.
        
        Args:
            df: DataFrame with customer records.
            
        Returns:
            DataFrame with one row per record, columns for each feature.
        """
        features_list = []
        for _, row in df.iterrows():
            features = self.extract(row)
            features_list.append(features.to_dict())
        
        return pd.DataFrame(features_list)
    
    def extract_array(self, record: dict | pd.Series) -> np.ndarray:
        """Extract features as numpy array for model input.
        
        Args:
            record: Dictionary or Series with customer data.
            
        Returns:
            1D numpy array of feature values.
        """
        return self.extract(record).to_array()
    
    def _validate_iban(self, iban: str) -> bool:
        """Validate IBAN using ISO 13616 mod-97 checksum."""
        if not iban or len(iban) < 4:
            return False
        
        # Remove spaces and convert to uppercase
        iban = iban.replace(" ", "").upper()
        
        # Check basic format: 2 letters + 2 digits + up to 30 alphanumeric
        if not re.match(r"^[A-Z]{2}\d{2}[A-Z0-9]{1,30}$", iban):
            return False
        
        # Rearrange: move first 4 chars to end
        rearranged = iban[4:] + iban[:4]
        
        # Convert letters to numbers (A=10, B=11, ..., Z=35)
        numeric = ""
        for char in rearranged:
            if char.isdigit():
                numeric += char
            else:
                numeric += str(ord(char) - ord("A") + 10)
        
        # Check mod 97
        try:
            return int(numeric) % 97 == 1
        except ValueError:
            return False
    
    def _parse_email(self, email: str) -> tuple[str, str]:
        """Parse email into local part and domain."""
        if "@" not in email:
            return email, ""
        parts = email.split("@")
        return parts[0], parts[1] if len(parts) > 1 else ""
    
    def _compute_entropy(self, text: str) -> float:
        """Compute Shannon entropy of a string."""
        if not text:
            return 0.0
        
        # Count character frequencies
        freq = {}
        for char in text:
            freq[char] = freq.get(char, 0) + 1
        
        # Compute entropy
        length = len(text)
        entropy = 0.0
        for count in freq.values():
            p = count / length
            entropy -= p * math.log2(p)
        
        return entropy
    
    def _numeric_ratio(self, text: str) -> float:
        """Compute ratio of numeric characters in text."""
        if not text:
            return 0.0
        digits = sum(1 for c in text if c.isdigit())
        return digits / len(text)
    
    def _keyboard_pattern_score(self, text: str) -> float:
        """Score how much the text resembles keyboard patterns."""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        max_match = 0.0
        
        for pattern in self.keyboard_patterns:
            if pattern in text_lower:
                match_ratio = len(pattern) / len(text_lower)
                max_match = max(max_match, match_ratio)
        
        return max_match
    
    def _validate_postal_code(self, postal_code: str, country: str) -> bool:
        """Validate postal code format for given country."""
        if not postal_code:
            return True  # Empty is not invalid, just missing
        
        pattern = self.postal_patterns.get(country)
        if not pattern:
            return True  # Unknown country, assume valid
        
        return bool(re.match(pattern, postal_code.upper()))
    
    def _get_address(self, record: dict) -> str:
        """Get full address string from record."""
        # Try structured fields first
        parts = []
        for field in ["street", "house_number", "postal_code", "city"]:
            value = record.get(field, "")
            if value:
                parts.append(str(value))
        
        if parts:
            return " ".join(parts)
        
        # Fall back to address field
        return str(record.get("address", ""))
    
    def _infer_country_from_address(self, record: dict) -> str:
        """Infer country code from address or nationality."""
        # Check explicit nationality
        nationality = record.get("nationality", "")
        if nationality and len(nationality) == 2:
            return nationality.upper()
        
        # Try to infer from postal code format or city
        postal = str(record.get("postal_code", ""))
        
        # German 5-digit postal codes
        if re.match(r"^\d{5}$", postal):
            return "DE"
        
        # Austrian/Swiss 4-digit
        if re.match(r"^\d{4}$", postal):
            return ""  # Could be AT, CH, BE, etc.
        
        return ""
    
    def _email_contains_name(self, email_local: str, surname: str, first_name: str) -> bool:
        """Check if email local part contains parts of the name."""
        email_lower = email_local.lower()
        
        # Check for surname (at least 3 chars)
        if len(surname) >= 3 and surname.lower()[:3] in email_lower:
            return True
        
        # Check for first name (at least 3 chars)
        if len(first_name) >= 3 and first_name.lower()[:3] in email_lower:
            return True
        
        return False
    
    def _compute_field_completeness(self, record: dict) -> float:
        """Compute ratio of non-empty important fields."""
        important_fields = [
            "surname", "first_name", "email", "iban",
            "street", "postal_code", "city", "date_of_birth",
        ]
        
        filled = 0
        total = len(important_fields)
        
        for field in important_fields:
            value = record.get(field, "")
            if value and str(value).strip():
                filled += 1
        
        return filled / total
