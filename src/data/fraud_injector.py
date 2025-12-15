"""Fraud Pattern Injector.

Injects specific fraud patterns into synthetic customer data to create
realistic anomalies for testing fraud detection algorithms.
"""

import random
import re
import string
from dataclasses import replace
from typing import Optional

from faker import Faker

from .generator import CustomerRecord
from .fraud_patterns import FraudType


class FraudInjector:
    """
    Injects fraud patterns into legitimate customer data.
    
    Creates various types of fraudulent records that mimic real-world
    fraud patterns like identity theft, synthetic identities, and
    subscription fraud.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the fraud injector.
        
        Args:
            seed: Random seed for reproducibility.
        """
        self.seed = seed
        self.faker = Faker()
        
        if seed is not None:
            random.seed(seed)
            Faker.seed(seed)
            
        self._fraud_counter = 0
    
    def _generate_fraud_id(self) -> str:
        """Generate a unique fraud customer ID."""
        self._fraud_counter += 1
        return f"FRAUD-{self._fraud_counter:08d}"
    
    def _introduce_typo(self, text: str) -> str:
        """
        Introduce a realistic typo into text.
        
        Common typo patterns:
        - Character swap (adjacent keys)
        - Missing character
        - Double character
        - Wrong character
        """
        if len(text) < 3:
            return text
            
        typo_type = random.choice(["swap", "missing", "double", "wrong"])
        pos = random.randint(1, len(text) - 2)
        
        if typo_type == "swap" and pos < len(text) - 1:
            # Swap adjacent characters
            chars = list(text)
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
            return "".join(chars)
            
        elif typo_type == "missing":
            # Remove a character
            return text[:pos] + text[pos + 1:]
            
        elif typo_type == "double":
            # Double a character
            return text[:pos] + text[pos] + text[pos:]
            
        else:  # wrong
            # Replace with adjacent keyboard character
            keyboard_neighbors = {
                'a': 'sq', 'b': 'vn', 'c': 'xv', 'd': 'sf', 'e': 'wr',
                'f': 'dg', 'g': 'fh', 'h': 'gj', 'i': 'uo', 'j': 'hk',
                'k': 'jl', 'l': 'k', 'm': 'n', 'n': 'bm', 'o': 'ip',
                'p': 'o', 'q': 'w', 'r': 'et', 's': 'ad', 't': 'ry',
                'u': 'yi', 'v': 'cb', 'w': 'qe', 'x': 'zc', 'y': 'tu',
                'z': 'x'
            }
            char = text[pos].lower()
            if char in keyboard_neighbors:
                new_char = random.choice(keyboard_neighbors[char])
                if text[pos].isupper():
                    new_char = new_char.upper()
                return text[:pos] + new_char + text[pos + 1:]
        
        return text
    
    def _modify_address_slightly(self, address: str) -> str:
        """
        Make slight modifications to an address.
        
        Patterns:
        - Abbreviate street type (Street -> St.)
        - Add/remove apartment numbers
        - Slight typos
        """
        modifications = [
            ("Street", "St."),
            ("Avenue", "Ave."),
            ("Boulevard", "Blvd."),
            ("Road", "Rd."),
            ("Strasse", "Str."),
            ("strasse", "str."),
        ]
        
        modified = address
        for original, replacement in modifications:
            if original in modified:
                if random.random() > 0.5:
                    modified = modified.replace(original, replacement)
                break
        
        # Sometimes add apartment number
        if random.random() > 0.7:
            apt_num = random.randint(1, 50)
            modified = f"Apt {apt_num}, {modified}"
        
        return modified

    @staticmethod
    def _format_address(strasse: str, hausnummer: str, plz: str, stadt: str) -> str:
        street_part = " ".join(p for p in [strasse.strip(), hausnummer.strip()] if p)
        city_part = " ".join(p for p in [plz.strip(), stadt.strip()] if p)
        if street_part and city_part:
            return f"{street_part}, {city_part}"
        return street_part or city_part

    def _generate_address_parts(self) -> tuple[str, str, str, str]:
        strasse = str(self.faker.street_name())
        hausnummer = str(self.faker.building_number())
        plz = str(self.faker.postcode())
        stadt = str(self.faker.city())
        return strasse, hausnummer, plz, stadt

    def _modify_address_parts_slightly(
        self,
        strasse: str,
        hausnummer: str,
        plz: str,
        stadt: str,
    ) -> tuple[str, str, str, str]:
        """Slightly modify structured address parts (typos/abbreviations/small number tweaks)."""
        new_strasse = strasse
        new_hausnummer = hausnummer
        new_plz = plz
        new_stadt = stadt

        # Street abbreviation/variants (German-leaning)
        replacements = [
            ("straße", "str."),
            ("strasse", "str."),
            ("Straße", "Str."),
            ("Strasse", "Str."),
            ("Street", "St."),
            ("Avenue", "Ave."),
            ("Road", "Rd."),
        ]
        if new_strasse:
            for original, replacement in replacements:
                if original in new_strasse and random.random() > 0.5:
                    new_strasse = new_strasse.replace(original, replacement)
                    break

        # Occasional small typo in street name
        if new_strasse and random.random() > 0.85:
            new_strasse = self._introduce_typo(new_strasse)

        # House number: sometimes +1/-1 if it looks numeric-ish
        if new_hausnummer and random.random() > 0.8:
            m = re.match(r"^(\d{1,5})([a-zA-Z]?)$", str(new_hausnummer).strip())
            if m:
                base = int(m.group(1))
                suffix = m.group(2)
                delta = random.choice([-1, 1])
                new_base = max(1, base + delta)
                new_hausnummer = f"{new_base}{suffix}"
            else:
                # fallback: append a plausible suffix
                if str(new_hausnummer).isdigit():
                    new_hausnummer = f"{new_hausnummer}a"

        return new_strasse, new_hausnummer, new_plz, new_stadt
    
    def create_near_duplicate(
        self, 
        base_record: CustomerRecord
    ) -> CustomerRecord:
        """
        Create a near-duplicate fraud record.
        
        Same address but different name - indicates possible identity theft
        or multiple fraudulent accounts at one location.
        
        Args:
            base_record: The legitimate record to base fraud on.
            
        Returns:
            Fraudulent CustomerRecord with same address, different identity.
        """
        return CustomerRecord(
            customer_id=self._generate_fraud_id(),
            surname=self.faker.last_name(),  # Different surname
            first_name=self.faker.first_name(),  # Different first name
            strasse=base_record.strasse,
            hausnummer=base_record.hausnummer,
            plz=base_record.plz,
            stadt=base_record.stadt,
            address=base_record.address,  # SAME address
            iban=self.faker.iban(),  # Different IBAN
            email=self.faker.email(),
            date_of_birth=self.faker.date_of_birth(minimum_age=18, maximum_age=80),
            nationality=base_record.nationality,
            is_fraud=True,
            fraud_type=FraudType.NEAR_DUPLICATE.value,
        )
    
    def create_typo_variant(
        self, 
        base_record: CustomerRecord
    ) -> CustomerRecord:
        """
        Create a typo variant fraud record.
        
        Slight variations in name/email that might indicate someone trying
        to create multiple accounts with the same identity.
        
        Args:
            base_record: The legitimate record to base fraud on.
            
        Returns:
            Fraudulent CustomerRecord with typos in identifying fields.
        """
        # Introduce typos in name
        new_surname = self._introduce_typo(base_record.surname)
        new_first_name = self._introduce_typo(base_record.first_name)
        
        # Modify email slightly
        email_parts = base_record.email.split("@")
        if len(email_parts) == 2:
            new_email = f"{self._introduce_typo(email_parts[0])}@{email_parts[1]}"
        else:
            new_email = self.faker.email()
        
        # Modify structured address fields slightly (and keep `address` consistent)
        new_strasse, new_hausnummer, new_plz, new_stadt = self._modify_address_parts_slightly(
            base_record.strasse,
            base_record.hausnummer,
            base_record.plz,
            base_record.stadt,
        )
        new_address = self._format_address(new_strasse, new_hausnummer, new_plz, new_stadt)

        return CustomerRecord(
            customer_id=self._generate_fraud_id(),
            surname=new_surname,
            first_name=new_first_name,
            strasse=new_strasse,
            hausnummer=new_hausnummer,
            plz=new_plz,
            stadt=new_stadt,
            address=new_address,
            iban=self.faker.iban(),  # Different IBAN
            email=new_email,
            date_of_birth=base_record.date_of_birth,  # Same DOB
            nationality=base_record.nationality,
            is_fraud=True,
            fraud_type=FraudType.TYPO_VARIANT.value,
        )
    
    def create_shared_iban(
        self, 
        base_record: CustomerRecord
    ) -> CustomerRecord:
        """
        Create a record sharing the same IBAN.
        
        Multiple accounts with the same banking info is a strong fraud indicator.
        
        Args:
            base_record: The legitimate record to base fraud on.
            
        Returns:
            Fraudulent CustomerRecord with same IBAN, different identity.
        """
        strasse, hausnummer, plz, stadt = self._generate_address_parts()
        address = self._format_address(strasse, hausnummer, plz, stadt)

        return CustomerRecord(
            customer_id=self._generate_fraud_id(),
            surname=self.faker.last_name(),
            first_name=self.faker.first_name(),
            strasse=strasse,
            hausnummer=hausnummer,
            plz=plz,
            stadt=stadt,
            address=address,
            iban=base_record.iban,  # SAME IBAN
            email=self.faker.email(),
            date_of_birth=self.faker.date_of_birth(minimum_age=18, maximum_age=80),
            nationality=base_record.nationality,
            is_fraud=True,
            fraud_type=FraudType.SHARED_IBAN.value,
        )
    
    def create_synthetic_identity(
        self, 
        base_record: CustomerRecord
    ) -> CustomerRecord:
        """
        Create a synthetic identity fraud record.
        
        Mix of real data (from base) and fake data to create a new identity.
        
        Args:
            base_record: The legitimate record to partially use.
            
        Returns:
            Fraudulent CustomerRecord with mixed real/fake data.
        """
        # Use real surname but fake first name (or vice versa)
        use_real_surname = random.random() > 0.5
        
        strasse, hausnummer, plz, stadt = self._generate_address_parts()
        address = self._format_address(strasse, hausnummer, plz, stadt)

        return CustomerRecord(
            customer_id=self._generate_fraud_id(),
            surname=base_record.surname if use_real_surname else self.faker.last_name(),
            first_name=self.faker.first_name() if use_real_surname else base_record.first_name,
            strasse=strasse,
            hausnummer=hausnummer,
            plz=plz,
            stadt=stadt,
            address=address,
            iban=self.faker.iban(),
            email=self.faker.email(),
            date_of_birth=base_record.date_of_birth,  # Use real DOB
            nationality=base_record.nationality,
            is_fraud=True,
            fraud_type=FraudType.SYNTHETIC_IDENTITY.value,
        )
    
    def inject_fraud_patterns(
        self,
        legitimate_records: list[CustomerRecord],
        fraud_ratio: float = 0.1,
        fraud_types: Optional[list[FraudType]] = None
    ) -> list[CustomerRecord]:
        """
        Inject fraud patterns based on legitimate records.
        
        Args:
            legitimate_records: List of legitimate CustomerRecords.
            fraud_ratio: Ratio of fraudulent records to create (0.1 = 10%).
            fraud_types: Types of fraud to inject. If None, uses all types.
            
        Returns:
            List of fraudulent CustomerRecords.
        """
        if fraud_types is None:
            fraud_types = list(FraudType)
        
        num_fraud = int(len(legitimate_records) * fraud_ratio)
        fraudulent_records = []
        
        # Select random legitimate records to base fraud on
        base_records = random.sample(
            legitimate_records, 
            min(num_fraud, len(legitimate_records))
        )
        
        fraud_creators = {
            FraudType.NEAR_DUPLICATE: self.create_near_duplicate,
            FraudType.TYPO_VARIANT: self.create_typo_variant,
            FraudType.SHARED_IBAN: self.create_shared_iban,
            FraudType.SYNTHETIC_IDENTITY: self.create_synthetic_identity,
        }
        
        for base_record in base_records:
            fraud_type = random.choice(fraud_types)
            creator = fraud_creators.get(fraud_type)
            
            if creator:
                fraud_record = creator(base_record)
                fraudulent_records.append(fraud_record)
        
        return fraudulent_records


if __name__ == "__main__":
    from .generator import CustomerDataGenerator
    
    # Test fraud injection
    gen = CustomerDataGenerator(seed=42)
    legit = gen.generate_records(10)
    
    injector = FraudInjector(seed=42)
    fraud = injector.inject_fraud_patterns(legit, fraud_ratio=0.3)
    
    print(f"Generated {len(legit)} legitimate and {len(fraud)} fraudulent records")
    for f in fraud:
        print(f"  - {f.fraud_type}: {f.first_name} {f.surname}")
