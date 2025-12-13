"""
Synthetic Customer Data Generator.

Generates legitimate customer records using Faker library with realistic
distributions for telecommunications customer data.

TASK-002: Create src/data/generator.py to generate legitimate customer records.
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd
from faker import Faker


@dataclass
class CustomerRecord:
    """Represents a single customer record with PII fields."""
    
    customer_id: str
    surname: str
    first_name: str
    address: str
    iban: str
    email: str
    date_of_birth: date
    nationality: str
    is_fraud: bool = False
    fraud_type: Optional[str] = None


class CustomerDataGenerator:
    """
    Generates synthetic customer data for fraud detection testing.
    
    Uses Faker library to create realistic PII data including names,
    addresses, banking information, and contact details.
    """
    
    # Supported locales with their nationality codes
    SUPPORTED_LOCALES = {
        "de_DE": "German",
        "en_US": "American",
        "en_GB": "British",
        "fr_FR": "French",
        "es_ES": "Spanish",
        "it_IT": "Italian",
        "nl_NL": "Dutch",
    }
    
    def __init__(self, seed: Optional[int] = None, locale: str = "de_DE"):
        """
        Initialize the data generator.
        
        Args:
            seed: Random seed for reproducibility.
            locale: Faker locale for region-specific data generation.
        """
        self.seed = seed
        self.locale = locale
        self.faker = Faker(locale)
        
        if seed is not None:
            Faker.seed(seed)
            
        self._customer_counter = 0
        
    def _generate_customer_id(self) -> str:
        """Generate a unique customer ID."""
        self._customer_counter += 1
        return f"CUST-{self._customer_counter:08d}"
    
    def _generate_iban(self) -> str:
        """Generate a realistic IBAN."""
        return self.faker.iban()
    
    def _generate_email(self, first_name: str, surname: str) -> str:
        """
        Generate an email address based on the customer name.
        
        Creates realistic email patterns like firstname.lastname@domain.com
        """
        domains = [
            "gmail.com", "yahoo.com", "outlook.com", "web.de",
            "gmx.de", "hotmail.com", "protonmail.com", "icloud.com"
        ]
        
        # Various email patterns
        patterns = [
            f"{first_name.lower()}.{surname.lower()}",
            f"{first_name.lower()}{surname.lower()}",
            f"{first_name[0].lower()}.{surname.lower()}",
            f"{first_name.lower()}.{surname.lower()}{self.faker.random_int(1, 99)}",
            f"{surname.lower()}.{first_name.lower()}",
        ]
        
        pattern = self.faker.random_element(patterns)
        domain = self.faker.random_element(domains)
        
        # Clean special characters from email
        clean_pattern = "".join(c for c in pattern if c.isalnum() or c == ".")
        
        return f"{clean_pattern}@{domain}"
    
    def generate_single_record(self) -> CustomerRecord:
        """
        Generate a single legitimate customer record.
        
        Returns:
            CustomerRecord with all PII fields populated.
        """
        first_name = self.faker.first_name()
        surname = self.faker.last_name()
        
        # Generate date of birth (18-80 years old)
        dob = self.faker.date_of_birth(minimum_age=18, maximum_age=80)
        
        # Get nationality based on locale
        nationality = self.SUPPORTED_LOCALES.get(self.locale, "Unknown")
        
        return CustomerRecord(
            customer_id=self._generate_customer_id(),
            surname=surname,
            first_name=first_name,
            address=self.faker.address().replace("\n", ", "),
            iban=self._generate_iban(),
            email=self._generate_email(first_name, surname),
            date_of_birth=dob,
            nationality=nationality,
            is_fraud=False,
            fraud_type=None,
        )
    
    def generate_records(self, count: int) -> list[CustomerRecord]:
        """
        Generate multiple legitimate customer records.
        
        Args:
            count: Number of records to generate.
            
        Returns:
            List of CustomerRecord objects.
        """
        return [self.generate_single_record() for _ in range(count)]
    
    def to_dataframe(self, records: list[CustomerRecord]) -> pd.DataFrame:
        """
        Convert customer records to a pandas DataFrame.
        
        Args:
            records: List of CustomerRecord objects.
            
        Returns:
            DataFrame with all customer data.
        """
        data = []
        for record in records:
            data.append({
                "customer_id": record.customer_id,
                "surname": record.surname,
                "first_name": record.first_name,
                "address": record.address,
                "iban": record.iban,
                "email": record.email,
                "date_of_birth": record.date_of_birth.isoformat(),
                "nationality": record.nationality,
                "is_fraud": record.is_fraud,
                "fraud_type": record.fraud_type,
            })
        
        return pd.DataFrame(data)


if __name__ == "__main__":
    # Quick test of the generator
    generator = CustomerDataGenerator(seed=42)
    records = generator.generate_records(5)
    df = generator.to_dataframe(records)
    print(df.to_string())
