"""Tests for the real-time scoring package."""

import pytest
import pandas as pd
import numpy as np

from src.scoring.intrinsic_features import (
    IntrinsicFeatureExtractor,
    IntrinsicFeatures,
    DISPOSABLE_EMAIL_DOMAINS,
)


class TestIntrinsicFeatureExtractor:
    """Tests for IntrinsicFeatureExtractor."""
    
    @pytest.fixture
    def extractor(self):
        """Create a feature extractor."""
        return IntrinsicFeatureExtractor()
    
    @pytest.fixture
    def valid_record(self):
        """Create a valid customer record."""
        return {
            "customer_id": "CUST-001",
            "surname": "Mueller",
            "first_name": "Hans",
            "email": "hans.mueller@gmail.com",
            "iban": "DE89370400440532013000",
            "street": "Hauptstrasse",
            "house_number": "42",
            "postal_code": "10115",
            "city": "Berlin",
            "date_of_birth": "1985-06-15",
        }
    
    @pytest.fixture
    def suspicious_record(self):
        """Create a suspicious customer record."""
        return {
            "customer_id": "CUST-002",
            "surname": "Xyz123",
            "first_name": "Test",
            "email": "x9y8z7w6@tempmail.com",
            "iban": "XX00000000000000000000",  # Invalid IBAN
            "street": "Fake Street",
            "postal_code": "00000",
            "city": "Nowhere",
        }
    
    def test_extract_valid_record(self, extractor, valid_record):
        """Test feature extraction from valid record."""
        features = extractor.extract(valid_record)
        
        assert isinstance(features, IntrinsicFeatures)
        assert features.iban_valid is True
        assert features.iban_country == "DE"
        assert features.email_is_disposable is False
        assert features.name_has_digits is False
        assert features.total_flags <= 2  # Should have few flags
    
    def test_extract_suspicious_record(self, extractor, suspicious_record):
        """Test feature extraction from suspicious record."""
        features = extractor.extract(suspicious_record)
        
        assert features.iban_valid is False
        assert features.email_is_disposable is True
        assert features.name_has_digits is True
        assert features.total_flags >= 3  # Should have several flags
    
    def test_iban_validation_valid(self, extractor):
        """Test IBAN validation with valid IBANs."""
        # German IBAN
        assert extractor._validate_iban("DE89370400440532013000") is True
        # British IBAN
        assert extractor._validate_iban("GB82WEST12345698765432") is True
        # With spaces (should be handled)
        assert extractor._validate_iban("DE89 3704 0044 0532 0130 00") is True
    
    def test_iban_validation_invalid(self, extractor):
        """Test IBAN validation with invalid IBANs."""
        assert extractor._validate_iban("XX00000000000000000000") is False
        assert extractor._validate_iban("DE00000000000000000000") is False
        assert extractor._validate_iban("invalid") is False
        assert extractor._validate_iban("") is False
    
    def test_email_entropy(self, extractor):
        """Test email entropy calculation."""
        # Low entropy (repetitive)
        low_entropy = extractor._compute_entropy("aaa")
        # High entropy (random)
        high_entropy = extractor._compute_entropy("xyz123abc")
        
        assert low_entropy < high_entropy
    
    def test_disposable_email_detection(self, extractor):
        """Test disposable email domain detection."""
        record_disposable = {
            "email": "test@tempmail.com",
            "surname": "Test",
            "first_name": "User",
            "iban": "DE89370400440532013000",
        }
        record_normal = {
            "email": "test@gmail.com",
            "surname": "Test",
            "first_name": "User",
            "iban": "DE89370400440532013000",
        }
        
        features_disposable = extractor.extract(record_disposable)
        features_normal = extractor.extract(record_normal)
        
        assert features_disposable.email_is_disposable is True
        assert features_normal.email_is_disposable is False
    
    def test_keyboard_pattern_detection(self, extractor):
        """Test keyboard pattern detection."""
        score_qwerty = extractor._keyboard_pattern_score("qwerty")
        score_normal = extractor._keyboard_pattern_score("mueller")
        
        assert score_qwerty > 0.5
        assert score_normal == 0.0
    
    def test_postal_code_validation(self, extractor):
        """Test postal code validation."""
        # German 5-digit
        assert extractor._validate_postal_code("10115", "DE") is True
        assert extractor._validate_postal_code("1234", "DE") is False
        
        # Austrian 4-digit
        assert extractor._validate_postal_code("1010", "AT") is True
        
        # Unknown country (should pass)
        assert extractor._validate_postal_code("12345", "ZZ") is True
    
    def test_to_array(self, extractor, valid_record):
        """Test conversion to numpy array."""
        features = extractor.extract(valid_record)
        array = features.to_array()
        
        assert isinstance(array, np.ndarray)
        assert len(array) == len(IntrinsicFeatures.feature_names())
    
    def test_to_dict(self, extractor, valid_record):
        """Test conversion to dictionary."""
        features = extractor.extract(valid_record)
        d = features.to_dict()
        
        assert isinstance(d, dict)
        assert "iban_valid" in d
        assert "email_entropy" in d
        assert "total_flags" in d
    
    def test_extract_batch(self, extractor, valid_record, suspicious_record):
        """Test batch extraction."""
        df = pd.DataFrame([valid_record, suspicious_record])
        result = extractor.extract_batch(df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "iban_valid" in result.columns
        assert "total_flags" in result.columns
    
    def test_empty_fields(self, extractor):
        """Test handling of empty/missing fields."""
        record = {
            "surname": "",
            "first_name": None,
            "email": "",
            "iban": "",
        }
        
        features = extractor.extract(record)
        
        # Should not raise, should handle gracefully
        assert features.iban_valid is False
        assert features.email_entropy == 0.0
    
    def test_email_contains_name(self, extractor):
        """Test email-name consistency check."""
        record_consistent = {
            "email": "hans.mueller@gmail.com",
            "surname": "Mueller",
            "first_name": "Hans",
            "iban": "DE89370400440532013000",
        }
        record_inconsistent = {
            "email": "xyz123@gmail.com",
            "surname": "Mueller",
            "first_name": "Hans",
            "iban": "DE89370400440532013000",
        }
        
        features_consistent = extractor.extract(record_consistent)
        features_inconsistent = extractor.extract(record_inconsistent)
        
        assert features_consistent.email_contains_name is True
        assert features_inconsistent.email_contains_name is False


class TestIntrinsicFeaturesDataclass:
    """Tests for IntrinsicFeatures dataclass."""
    
    def test_feature_names_count(self):
        """Test that feature names match array length."""
        names = IntrinsicFeatures.feature_names()
        
        # Create a dummy features object
        features = IntrinsicFeatures(
            iban_valid=True,
            iban_country="DE",
            iban_country_matches_address=True,
            email_entropy=2.5,
            email_is_disposable=False,
            email_numeric_ratio=0.1,
            email_length=20,
            email_domain="gmail.com",
            email_has_plus_addressing=False,
            surname_entropy=2.0,
            firstname_entropy=1.5,
            name_has_digits=False,
            name_keyboard_pattern_score=0.0,
            name_length_ratio=1.2,
            postal_code_valid=True,
            address_entropy=3.0,
            address_has_po_box=False,
            email_contains_name=True,
            field_completeness=0.9,
            total_flags=0,
        )
        
        array = features.to_array()
        assert len(array) == len(names)
