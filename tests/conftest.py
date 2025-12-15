"""Pytest configuration and fixtures for fraud detection tests."""

import pandas as pd
import pytest


@pytest.fixture
def sample_customer_data() -> pd.DataFrame:
    """Create sample customer data for testing.
    
    Returns:
        DataFrame with 6 customer records including:
        - 3 similar records (potential fraud cluster)
        - 1 obvious outlier
        - 2 normal records
    """
    return pd.DataFrame({
        "customer_id": ["C001", "C002", "C003", "C004", "C005", "C006"],
        "surname": ["Mueller", "Muller", "Smith", "RandomXYZ", "Mueller", "Williams"],
        "first_name": ["Hans", "Hans", "John", "Test123", "Hans", "Bob"],
        "strasse": ["Main St", "Main St", "Oak Ave", "Fake", "Main St", "Elm St"],
        "hausnummer": ["1", "1", "5", "123", "1", "20"],
        "plz": ["10115", "10115", "20095", "00000", "10115", "99999"],
        "stadt": ["Berlin", "Berlin", "Hamburg", "Nowhere", "Berlin", "London"],
        "address": [
            "Main St 1, 10115 Berlin",
            "Main St 1, 10115 Berlin",
            "Oak Ave 5, 20095 Hamburg",
            "Fake 123, 00000 Nowhere",
            "Main St 1, 10115 Berlin",
            "Elm St 20, 99999 London",
        ],
        "email": [
            "hans@test.com",
            "hans@test.com",
            "john@test.com",
            "x1y2@fake.net",
            "h@test.com",
            "bob@outlook.com",
        ],
        "iban": ["DE123", "DE123", "DE456", "XX000", "DE123", "GB999"],
        "date_of_birth": [
            "1990-01-01",
            "1990-01-01",
            "1985-06-15",
            "2000-01-01",
            "1990-01-01",
            "1988-12-25",
        ],
        "nationality": ["German", "German", "British", "Unknown", "German", "British"],
    })


@pytest.fixture
def sample_fraud_data() -> pd.DataFrame:
    """Create sample data with known fraud labels.
    
    Returns:
        DataFrame with customer records and is_fraud ground truth.
    """
    df = pd.DataFrame({
        "customer_id": ["C001", "C002", "C003", "C004", "C005"],
        "surname": ["Mueller", "Muller", "Schmidt", "RandomXYZ", "Mueller"],
        "first_name": ["Hans", "Hans", "Klaus", "Test123", "Hans"],
        "email": ["hans@test.com", "hans@test.com", "klaus@mail.de", "x@fake.net", "h@test.com"],
        "iban": ["DE123456", "DE123456", "DE654321", "XX000000", "DE123456"],
        "address": ["Street 1", "Street 1", "Avenue 5", "Fake 123", "Street 1"],
        "is_fraud": [True, True, False, True, True],
        "fraud_type": ["shared_iban", "shared_iban", "", "synthetic_identity", "shared_iban"],
    })
    return df


@pytest.fixture
def empty_dataframe() -> pd.DataFrame:
    """Create an empty DataFrame for edge case testing."""
    return pd.DataFrame()


@pytest.fixture
def minimal_dataframe() -> pd.DataFrame:
    """Create minimal DataFrame with just required columns."""
    return pd.DataFrame({
        "customer_id": ["C001"],
        "surname": ["Test"],
        "first_name": ["User"],
    })
