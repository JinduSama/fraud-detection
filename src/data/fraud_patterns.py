"""
Extended Fraud Pattern Library.

Comprehensive collection of fraud patterns for testing and
training fraud detection systems.
"""

import random
import string
from dataclasses import replace
from datetime import date, timedelta
from enum import Enum
from typing import Optional

from faker import Faker

from .generator import CustomerRecord


class FraudType(Enum):
    """Extended types of fraud patterns."""
    
    # Original patterns
    NEAR_DUPLICATE = "near_duplicate"
    TYPO_VARIANT = "typo_variant"
    SHARED_IBAN = "shared_iban"
    SYNTHETIC_IDENTITY = "synthetic_identity"
    ADDRESS_MANIPULATION = "address_manipulation"
    
    # New patterns
    DEVICE_SHARING = "device_sharing"
    VELOCITY_FRAUD = "velocity_fraud"
    RING_FRAUD = "ring_fraud"
    DATA_HARVESTING = "data_harvesting"
    BIRTHDAY_PARADOX = "birthday_paradox"


class ExtendedFraudInjector:
    """
    Extended fraud pattern injector with additional fraud types.
    
    Supports all original fraud patterns plus:
    - Device sharing (multiple accounts from same device)
    - Velocity fraud (many accounts in short time)
    - Ring fraud (circular references)
    - Data harvesting (sequential patterns)
    - Birthday paradox (suspicious DOB distributions)
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the extended fraud injector.
        
        Args:
            seed: Random seed for reproducibility.
        """
        self.seed = seed
        self.faker = Faker()
        
        if seed is not None:
            random.seed(seed)
            Faker.seed(seed)
        
        self._fraud_counter = 0
        self._device_fingerprints: list[str] = []
        self._ring_base_records: list[CustomerRecord] = []
    
    def _generate_fraud_id(self) -> str:
        """Generate a unique fraud customer ID."""
        self._fraud_counter += 1
        return f"FRAUD-{self._fraud_counter:08d}"
    
    def _generate_device_fingerprint(self) -> str:
        """Generate a device fingerprint."""
        return ''.join(random.choices(string.hexdigits.lower(), k=32))
    
    def _introduce_typo(self, text: str) -> str:
        """Introduce a realistic typo into text."""
        if len(text) < 3:
            return text
        
        typo_type = random.choice(["swap", "missing", "double", "wrong"])
        pos = random.randint(1, len(text) - 2)
        
        if typo_type == "swap" and pos < len(text) - 1:
            chars = list(text)
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
            return "".join(chars)
        elif typo_type == "missing":
            return text[:pos] + text[pos + 1:]
        elif typo_type == "double":
            return text[:pos] + text[pos] + text[pos:]
        else:
            keyboard_neighbors = {
                'a': 'sq', 'b': 'vn', 'c': 'xv', 'd': 'sf', 'e': 'wr',
                'f': 'dg', 'g': 'fh', 'h': 'gj', 'i': 'uo', 'j': 'hk',
                'k': 'jl', 'l': 'k', 'm': 'n', 'n': 'bm', 'o': 'ip',
                'p': 'o', 'q': 'w', 'r': 'et', 's': 'ad', 't': 'ry',
                'u': 'yi', 'v': 'cb', 'w': 'qe', 'x': 'zc', 'y': 'tu', 'z': 'x'
            }
            char = text[pos].lower()
            if char in keyboard_neighbors:
                new_char = random.choice(keyboard_neighbors[char])
                if text[pos].isupper():
                    new_char = new_char.upper()
                return text[:pos] + new_char + text[pos + 1:]
        return text
    
    def _modify_address_slightly(self, address: str) -> str:
        """Make slight modifications to an address."""
        modifications = [
            ("Street", "St."), ("Avenue", "Ave."), ("Boulevard", "Blvd."),
            ("Road", "Rd."), ("Strasse", "Str."), ("strasse", "str."),
        ]
        
        modified = address
        for original, replacement in modifications:
            if original in modified:
                if random.random() > 0.5:
                    modified = modified.replace(original, replacement)
                break
        
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
        # Keep it simple: sometimes abbreviate/typo street, sometimes tweak house number suffix.
        new_strasse = strasse
        new_hausnummer = hausnummer
        new_plz = plz
        new_stadt = stadt

        replacements = [
            ("straße", "str."),
            ("strasse", "str."),
            ("Straße", "Str."),
            ("Strasse", "Str."),
            ("Street", "St."),
            ("Avenue", "Ave."),
            ("Road", "Rd."),
        ]
        if new_strasse and random.random() > 0.6:
            for original, replacement in replacements:
                if original in new_strasse:
                    new_strasse = new_strasse.replace(original, replacement)
                    break

        if new_strasse and random.random() > 0.85:
            new_strasse = self._introduce_typo(new_strasse)

        if new_hausnummer and random.random() > 0.8:
            # Add a plausible suffix if it doesn't already have one
            s = str(new_hausnummer).strip()
            if s.isdigit():
                new_hausnummer = f"{s}{random.choice(['a', 'b', ''])}".strip()

        return new_strasse, new_hausnummer, new_plz, new_stadt
    
    # Original fraud patterns
    def create_near_duplicate(self, base_record: CustomerRecord) -> CustomerRecord:
        """Create a near-duplicate fraud record (same address, different name)."""
        return CustomerRecord(
            customer_id=self._generate_fraud_id(),
            surname=self.faker.last_name(),
            first_name=self.faker.first_name(),
            strasse=base_record.strasse,
            hausnummer=base_record.hausnummer,
            plz=base_record.plz,
            stadt=base_record.stadt,
            address=base_record.address,
            iban=self.faker.iban(),
            email=self.faker.email(),
            date_of_birth=self.faker.date_of_birth(minimum_age=18, maximum_age=80),
            nationality=base_record.nationality,
            is_fraud=True,
            fraud_type=FraudType.NEAR_DUPLICATE.value,
        )
    
    def create_typo_variant(self, base_record: CustomerRecord) -> CustomerRecord:
        """Create a typo variant fraud record."""
        new_surname = self._introduce_typo(base_record.surname)
        new_first_name = self._introduce_typo(base_record.first_name)
        
        email_parts = base_record.email.split("@")
        if len(email_parts) == 2:
            new_email = f"{self._introduce_typo(email_parts[0])}@{email_parts[1]}"
        else:
            new_email = self.faker.email()
        
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
            iban=self.faker.iban(),
            email=new_email,
            date_of_birth=base_record.date_of_birth,
            nationality=base_record.nationality,
            is_fraud=True,
            fraud_type=FraudType.TYPO_VARIANT.value,
        )
    
    def create_shared_iban(self, base_record: CustomerRecord) -> CustomerRecord:
        """Create a record sharing the same IBAN."""
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
            iban=base_record.iban,
            email=self.faker.email(),
            date_of_birth=self.faker.date_of_birth(minimum_age=18, maximum_age=80),
            nationality=base_record.nationality,
            is_fraud=True,
            fraud_type=FraudType.SHARED_IBAN.value,
        )
    
    def create_synthetic_identity(self, base_record: CustomerRecord) -> CustomerRecord:
        """Create a synthetic identity fraud record."""
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
            date_of_birth=base_record.date_of_birth,
            nationality=base_record.nationality,
            is_fraud=True,
            fraud_type=FraudType.SYNTHETIC_IDENTITY.value,
        )
    
    # New fraud patterns
    def create_device_sharing_group(
        self, 
        base_record: CustomerRecord, 
        group_size: int = 3
    ) -> list[CustomerRecord]:
        """
        Create multiple accounts from same 'device fingerprint'.
        
        Simulates fraudsters creating multiple accounts from the same device.
        """
        fingerprint = self._generate_device_fingerprint()
        records = []
        
        # Use similar email patterns
        email_domain = random.choice(["gmail.com", "yahoo.com", "outlook.com"])
        
        for i in range(group_size):
            first_name = self.faker.first_name()
            surname = self.faker.last_name()
            
            # Create similar email pattern
            email = f"{first_name.lower()}{random.randint(1, 99)}@{email_domain}"

            strasse, hausnummer, plz, stadt = self._generate_address_parts()
            address = self._format_address(strasse, hausnummer, plz, stadt)
            
            record = CustomerRecord(
                customer_id=self._generate_fraud_id(),
                surname=surname,
                first_name=first_name,
                strasse=strasse,
                hausnummer=hausnummer,
                plz=plz,
                stadt=stadt,
                address=address,
                iban=self.faker.iban(),
                email=email,
                date_of_birth=self.faker.date_of_birth(minimum_age=18, maximum_age=40),
                nationality=base_record.nationality,
                is_fraud=True,
                fraud_type=FraudType.DEVICE_SHARING.value,
            )
            records.append(record)
        
        return records
    
    def create_velocity_fraud_group(
        self, 
        base_record: CustomerRecord, 
        group_size: int = 5
    ) -> list[CustomerRecord]:
        """
        Create accounts simulating velocity fraud.
        
        Multiple accounts created in rapid succession with similar patterns.
        """
        records = []
        base_email_domain = random.choice(["gmail.com", "tempmail.com", "protonmail.com"])
        
        # Use sequential or similar patterns
        for i in range(group_size):
            first_name = self.faker.first_name()
            
            # Sequential email pattern
            email = f"user{random.randint(1000, 9999)}_{i}@{base_email_domain}"
            
            # Similar address area
            # Keep same city/PLZ area as base where possible
            strasse = str(self.faker.street_name())
            hausnummer = str(random.randint(1, 100))
            plz = getattr(base_record, "plz", "") or str(self.faker.postcode())
            stadt = getattr(base_record, "stadt", "") or str(self.faker.city())
            address = self._format_address(strasse, hausnummer, plz, stadt)
            
            record = CustomerRecord(
                customer_id=self._generate_fraud_id(),
                surname=self.faker.last_name(),
                first_name=first_name,
                strasse=strasse,
                hausnummer=hausnummer,
                plz=plz,
                stadt=stadt,
                address=address,
                iban=self.faker.iban(),
                email=email,
                date_of_birth=self.faker.date_of_birth(minimum_age=18, maximum_age=35),
                nationality=base_record.nationality,
                is_fraud=True,
                fraud_type=FraudType.VELOCITY_FRAUD.value,
            )
            records.append(record)
        
        return records
    
    def create_ring_fraud_group(
        self, 
        base_record: CustomerRecord, 
        ring_size: int = 4
    ) -> list[CustomerRecord]:
        """
        Create a fraud ring with circular references.
        
        Records reference each other through shared attributes.
        """
        records = []
        shared_address_parts = base_record.address.split(",")[0] if "," in base_record.address else base_record.address
        
        # Create ring with shared attributes
        ibans = [self.faker.iban() for _ in range(ring_size)]
        
        for i in range(ring_size):
            # Share IBAN with next member (circular)
            shared_iban = ibans[(i + 1) % ring_size]
            
            record = CustomerRecord(
                customer_id=self._generate_fraud_id(),
                surname=self.faker.last_name(),
                first_name=self.faker.first_name(),
                strasse=base_record.strasse,
                hausnummer=f"{base_record.hausnummer}{i + 1}" if str(base_record.hausnummer).strip() else str(i + 1),
                plz=base_record.plz,
                stadt=base_record.stadt,
                address=self._format_address(
                    base_record.strasse,
                    f"{base_record.hausnummer}{i + 1}" if str(base_record.hausnummer).strip() else str(i + 1),
                    base_record.plz,
                    base_record.stadt,
                ),
                iban=shared_iban,
                email=self.faker.email(),
                date_of_birth=self.faker.date_of_birth(minimum_age=25, maximum_age=45),
                nationality=base_record.nationality,
                is_fraud=True,
                fraud_type=FraudType.RING_FRAUD.value,
            )
            records.append(record)
        
        return records
    
    def create_data_harvesting_group(
        self, 
        base_record: CustomerRecord, 
        group_size: int = 4
    ) -> list[CustomerRecord]:
        """
        Create records with sequential/patterned data.
        
        Simulates data harvesting with systematic patterns.
        """
        records = []
        base_num = random.randint(1000, 9000)
        
        for i in range(group_size):
            # Sequential email
            email = f"account{base_num + i}@{random.choice(['gmail.com', 'yahoo.com'])}"
            
            # Sequential IBAN-like pattern
            iban_base = base_record.iban[:4] if len(base_record.iban) >= 4 else "DE00"
            iban_suffix = str(base_num + i).zfill(16)
            sequential_iban = f"{iban_base}{iban_suffix}"

            strasse, hausnummer, plz, stadt = self._generate_address_parts()
            address = self._format_address(strasse, hausnummer, plz, stadt)
            
            record = CustomerRecord(
                customer_id=self._generate_fraud_id(),
                surname=self.faker.last_name(),
                first_name=self.faker.first_name(),
                strasse=strasse,
                hausnummer=hausnummer,
                plz=plz,
                stadt=stadt,
                address=address,
                iban=sequential_iban,
                email=email,
                date_of_birth=self.faker.date_of_birth(minimum_age=18, maximum_age=60),
                nationality=base_record.nationality,
                is_fraud=True,
                fraud_type=FraudType.DATA_HARVESTING.value,
            )
            records.append(record)
        
        return records
    
    def create_birthday_paradox_group(
        self, 
        base_record: CustomerRecord, 
        group_size: int = 5
    ) -> list[CustomerRecord]:
        """
        Create records with suspicious DOB patterns.
        
        Unrealistic distributions like all Jan 1, 2000.
        """
        records = []
        suspicious_dates = [
            date(2000, 1, 1),
            date(1990, 1, 1),
            date(1980, 1, 1),
            date(1985, 6, 15),
            date(2000, 12, 31),
        ]
        
        for i in range(group_size):
            suspicious_dob = random.choice(suspicious_dates)

            strasse, hausnummer, plz, stadt = self._generate_address_parts()
            address = self._format_address(strasse, hausnummer, plz, stadt)
            
            record = CustomerRecord(
                customer_id=self._generate_fraud_id(),
                surname=self.faker.last_name(),
                first_name=self.faker.first_name(),
                strasse=strasse,
                hausnummer=hausnummer,
                plz=plz,
                stadt=stadt,
                address=address,
                iban=self.faker.iban(),
                email=self.faker.email(),
                date_of_birth=suspicious_dob,
                nationality=base_record.nationality,
                is_fraud=True,
                fraud_type=FraudType.BIRTHDAY_PARADOX.value,
            )
            records.append(record)
        
        return records
    
    def inject_fraud_patterns(
        self,
        legitimate_records: list[CustomerRecord],
        fraud_ratio: float = 0.15,
        fraud_types: Optional[list[FraudType]] = None,
        include_group_patterns: bool = True
    ) -> list[CustomerRecord]:
        """
        Inject fraud patterns into dataset.
        
        Args:
            legitimate_records: List of legitimate records.
            fraud_ratio: Ratio of fraudulent records to create.
            fraud_types: Types of fraud to inject (default: all).
            include_group_patterns: Whether to include group-based fraud.
            
        Returns:
            List of fraudulent records.
        """
        if fraud_types is None:
            fraud_types = list(FraudType)
        
        num_fraud = int(len(legitimate_records) * fraud_ratio)
        fraudulent_records = []
        
        # Simple fraud creators (single record)
        simple_creators = {
            FraudType.NEAR_DUPLICATE: self.create_near_duplicate,
            FraudType.TYPO_VARIANT: self.create_typo_variant,
            FraudType.SHARED_IBAN: self.create_shared_iban,
            FraudType.SYNTHETIC_IDENTITY: self.create_synthetic_identity,
        }
        
        # Group fraud creators
        group_creators = {
            FraudType.DEVICE_SHARING: self.create_device_sharing_group,
            FraudType.VELOCITY_FRAUD: self.create_velocity_fraud_group,
            FraudType.RING_FRAUD: self.create_ring_fraud_group,
            FraudType.DATA_HARVESTING: self.create_data_harvesting_group,
            FraudType.BIRTHDAY_PARADOX: self.create_birthday_paradox_group,
        }
        
        base_records = random.sample(
            legitimate_records,
            min(num_fraud, len(legitimate_records))
        )
        
        for base_record in base_records:
            if len(fraudulent_records) >= num_fraud:
                break
            
            fraud_type = random.choice(fraud_types)
            
            if fraud_type in simple_creators:
                fraud_record = simple_creators[fraud_type](base_record)
                fraudulent_records.append(fraud_record)
            elif fraud_type in group_creators and include_group_patterns:
                group_size = random.randint(2, 4)
                group_records = group_creators[fraud_type](base_record, group_size)
                fraudulent_records.extend(group_records[:num_fraud - len(fraudulent_records)])
        
        return fraudulent_records


if __name__ == "__main__":
    from .generator import CustomerDataGenerator
    
    # Test extended fraud injection
    gen = CustomerDataGenerator(seed=42)
    legit = gen.generate_records(20)
    
    injector = ExtendedFraudInjector(seed=42)
    fraud = injector.inject_fraud_patterns(legit, fraud_ratio=0.4, include_group_patterns=True)
    
    print(f"Generated {len(legit)} legitimate and {len(fraud)} fraudulent records")
    
    # Count by fraud type
    fraud_type_counts = {}
    for f in fraud:
        ft = f.fraud_type
        fraud_type_counts[ft] = fraud_type_counts.get(ft, 0) + 1
    
    print("\nFraud by type:")
    for ft, count in fraud_type_counts.items():
        print(f"  - {ft}: {count}")
