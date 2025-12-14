"""
Feature Engineering Module.

Comprehensive feature extraction for fraud detection including
string analysis, phonetic encoding, cross-field analysis, and entropy metrics.
"""

import math
import re
import unicodedata
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import jellyfish


class FeatureExtractor:
    """
    Comprehensive feature extractor for fraud detection.
    
    Extracts multiple categories of features:
    - String length features
    - Complexity features (digits, special chars, case)
    - Email analysis features
    - Phonetic encodings
    - Cross-field features
    - Entropy features
    - Temporal features (from DOB)
    - Network features (shared attributes)
    """
    
    def __init__(self, compute_network_features: bool = True):
        """
        Initialize the feature extractor.
        
        Args:
            compute_network_features: Whether to compute network features
                                     (requires full dataset context).
        """
        self.compute_network_features = compute_network_features
        self._feature_names: list[str] = []
        
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for comparison."""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if not unicodedata.combining(c))
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = ' '.join(text.split())
        
        return text
    
    @staticmethod
    def calculate_entropy(text: str) -> float:
        """
        Calculate Shannon entropy of a string.
        
        Higher entropy indicates more random/unusual text.
        
        Args:
            text: Input string.
            
        Returns:
            Entropy value (higher = more random).
        """
        if not text:
            return 0.0
        
        # Count character frequencies
        counter = Counter(text.lower())
        length = len(text)
        
        # Calculate entropy
        entropy = 0.0
        for count in counter.values():
            prob = count / length
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    @staticmethod
    def extract_email_components(email: str) -> tuple[str, str]:
        """Extract local part and domain from email."""
        if pd.isna(email) or '@' not in str(email):
            return "", ""
        
        parts = str(email).lower().split('@')
        if len(parts) == 2:
            return parts[0], parts[1]
        return "", ""
    
    @staticmethod
    def _safe_str(val) -> str:
        return "" if pd.isna(val) else str(val)

    @staticmethod
    def _longest_run(pattern: str, text: str) -> int:
        if not text:
            return 0
        runs = [len(m.group(0)) for m in re.finditer(pattern, text)]
        return max(runs) if runs else 0

    @staticmethod
    def _normalize_iban(iban: str) -> str:
        """Normalize IBAN by stripping whitespace and uppercasing."""
        if pd.isna(iban):
            return ""
        return re.sub(r"\s+", "", str(iban)).upper()

    @staticmethod
    def format_address(strasse: str, hausnummer: str, plz: str, stadt: str) -> str:
        """Format a simple address string from structured fields."""
        street_part = " ".join(p for p in [str(strasse or "").strip(), str(hausnummer or "").strip()] if p)
        city_part = " ".join(p for p in [str(plz or "").strip(), str(stadt or "").strip()] if p)
        if street_part and city_part:
            return f"{street_part}, {city_part}"
        return street_part or city_part

    def _get_address_text(self, row: pd.Series) -> str:
        """Prefer structured address fields; fall back to legacy `address`."""
        strasse = self._safe_str(row.get("strasse", "")).strip()
        hausnummer = self._safe_str(row.get("hausnummer", "")).strip()
        plz = self._safe_str(row.get("plz", "")).strip()
        stadt = self._safe_str(row.get("stadt", "")).strip()
        if any([strasse, hausnummer, plz, stadt]):
            return self.format_address(strasse, hausnummer, plz, stadt)
        return self._safe_str(row.get("address", "")).strip()

    @staticmethod
    def _is_all_same(text: str) -> int:
        if not text:
            return 0
        return int(len(set(text)) == 1)

    @staticmethod
    def _is_simple_sequence(text: str) -> int:
        """Detects very simple ascending/descending digit sequences (weak heuristic)."""
        if not text or not text.isdigit():
            return 0
        asc = "0123456789"
        desc = asc[::-1]
        return int(text in asc or text in desc)

    @staticmethod
    def validate_iban(iban: str) -> int:
        """
        ISO 13616 (IBAN) mod-97 check. Returns 1 if valid else 0.
        """
        if pd.isna(iban):
            return 0
        iban = FeatureExtractor._normalize_iban(iban)
        if not iban or not re.fullmatch(r"[A-Z0-9]+", iban):
            return 0
        if len(iban) < 15 or len(iban) > 34:
            return 0

        rearranged = iban[4:] + iban[:4]
        digits = []
        for ch in rearranged:
            if ch.isalpha():
                digits.append(str(ord(ch) - 55))  # A=10 ... Z=35
            else:
                digits.append(ch)
        digit_str = "".join(digits)

        remainder = 0
        for ch in digit_str:
            remainder = (remainder * 10 + (ord(ch) - 48)) % 97
        return int(remainder == 1)
    
    def _extract_string_length_features(self, row: pd.Series) -> dict:
        """Extract string length features."""
        features = {}

        for col in ["surname", "first_name", "address", "strasse", "hausnummer", "plz", "stadt"]:
            if col in row.index:
                val = str(row[col]) if pd.notna(row[col]) else ""
                features[f'len_{col}'] = len(val)
        
        return features
    
    def _extract_complexity_features(self, row: pd.Series) -> dict:
        """Extract complexity features (digits, special chars, case, repeats, vowels)."""
        features = {}

        for col in ["surname", "first_name", "email", "address", "strasse", "hausnummer", "plz", "stadt"]:
            if col in row.index:
                val = str(row[col]) if pd.notna(row[col]) else ""
                
                features[f'{col}_digit_count'] = sum(c.isdigit() for c in val)
                features[f'{col}_special_count'] = sum(not c.isalnum() and c != ' ' for c in val)
                features[f'{col}_upper_ratio'] = sum(c.isupper() for c in val) / max(len(val), 1)

                # Casing features
                features[f'{col}_is_lower'] = int(val.islower())
                features[f'{col}_is_upper'] = int(val.isupper())
                features[f'{col}_is_title'] = int(val.istitle())
                
                # New features
                # Max consecutive repeated characters
                if val:
                    # Find all repeated sequences
                    repeats = [len(match.group(0)) for match in re.finditer(r'(.)\1+', val.lower())]
                    features[f'{col}_max_repeats'] = max(repeats) if repeats else 1
                else:
                    features[f'{col}_max_repeats'] = 0
                
                # Vowel ratio
                vowels = set('aeiou')
                vowel_count = sum(1 for c in val.lower() if c in vowels)
                features[f'{col}_vowel_ratio'] = vowel_count / max(len(val), 1)
        
        return features
    
    def _extract_email_features(self, row: pd.Series) -> dict:
        """Extract email-specific features."""
        features = {
            "email_domain_length": 0,
            "email_local_length": 0,
            "email_has_numbers_local": 0,
            "email_num_dots_local": 0,
            "email_consonant_ratio": 0.0,

            "email_domain_num_parts": 0,
            "email_tld_length": 0,
            "email_local_has_plus": 0,
            "email_local_has_underscore": 0,
            "email_domain_has_digits": 0,
            "email_local_longest_digit_run": 0,
            "email_domain_has_dot": 0,
        }
        
        if 'email' not in row.index or pd.isna(row['email']):
            return features
        
        local, domain = self.extract_email_components(row['email'])
        
        features['email_domain_length'] = len(domain)
        features['email_local_length'] = len(local)
        features['email_has_numbers_local'] = int(any(c.isdigit() for c in local))
        features['email_num_dots_local'] = local.count('.')

        consonants = set('bcdfghjklmnpqrstvwxyz')
        consonant_count = sum(1 for c in local.lower() if c in consonants)
        features['email_consonant_ratio'] = consonant_count / max(len(local), 1)
        
        # New email features
        features["email_domain_num_parts"] = 0 if not domain else len([p for p in domain.split(".") if p])
        tld = domain.rsplit(".", 1)[-1] if domain and "." in domain else ""
        features["email_tld_length"] = len(tld)
        features["email_local_has_plus"] = int("+" in local)
        features["email_local_has_underscore"] = int("_" in local)
        features["email_domain_has_digits"] = int(any(c.isdigit() for c in domain))
        features["email_local_longest_digit_run"] = self._longest_run(r"\d+", local)
        features["email_domain_has_dot"] = int("." in domain)
        
        return features
    
    def _extract_iban_features(self, row: pd.Series) -> dict:
        """Extract IBAN specific features."""
        features = {
            'iban_length': 0,
            'iban_digits_count': 0,
            'iban_letters_count': 0,
            "iban_is_valid": 0,
            "iban_has_spaces": 0,

            # General IBAN structure components
            "iban_check_digits": -1,
            "iban_country_is_de": 0,

            # Germany-specific structure + heuristics
            "iban_de_structure_ok": 0,
            "iban_de_blz_all_same": 0,
            "iban_de_account_all_same": 0,
            "iban_de_blz_is_sequence": 0,
            "iban_de_account_is_sequence": 0,
            "iban_de_account_leading_zeros": 0,
            "iban_de_blz_entropy": 0.0,
            "iban_de_account_entropy": 0.0,
        }
        iban = str(row.get('iban', '')) if pd.notna(row.get('iban')) else ""
        if iban:
            features['iban_length'] = len(iban)
            features['iban_digits_count'] = sum(c.isdigit() for c in iban)
            features['iban_letters_count'] = sum(c.isalpha() for c in iban)
            features["iban_has_spaces"] = int(bool(re.search(r"\s", iban)))

            iban_norm = self._normalize_iban(iban)
            features["iban_is_valid"] = self.validate_iban(iban_norm)

            # Extract country + check digits if present
            if len(iban_norm) >= 4:
                country = iban_norm[:2]
                features["iban_country_is_de"] = int(country == "DE")
                try:
                    features["iban_check_digits"] = int(iban_norm[2:4])
                except Exception:
                    features["iban_check_digits"] = -1

            # Germany-specific checks: DE + 22 chars total and digits after DE
            if features["iban_country_is_de"]:
                # DE IBAN: 2 letters + 2 check digits + 8 BLZ + 10 account = 22
                features["iban_de_structure_ok"] = int(bool(re.fullmatch(r"DE\d{20}", iban_norm)))
                if len(iban_norm) == 22 and iban_norm.startswith("DE") and iban_norm[2:].isdigit():
                    blz = iban_norm[4:12]
                    account = iban_norm[12:22]

                    features["iban_de_blz_all_same"] = self._is_all_same(blz)
                    features["iban_de_account_all_same"] = self._is_all_same(account)

                    features["iban_de_blz_is_sequence"] = self._is_simple_sequence(blz)
                    features["iban_de_account_is_sequence"] = self._is_simple_sequence(account)

                    features["iban_de_account_leading_zeros"] = len(account) - len(account.lstrip("0"))

                    # Entropy as weak randomness/plausibility signal
                    features["iban_de_blz_entropy"] = self.calculate_entropy(blz)
                    features["iban_de_account_entropy"] = self.calculate_entropy(account)
        return features
    
    def _extract_phonetic_features(self, row: pd.Series) -> dict:
        """Extract phonetic encoding features."""
        features = {}
        
        for col in ['surname', 'first_name']:
            if col in row.index:
                val = str(row[col]) if pd.notna(row[col]) else ""
                clean = self.normalize_text(val)
                words = clean.split()
                first_word = words[0] if words else ""
                
                try:
                    features[f'{col}_soundex'] = jellyfish.soundex(first_word) if first_word else ""
                    features[f'{col}_metaphone'] = jellyfish.metaphone(first_word) if first_word else ""
                    features[f'{col}_nysiis'] = jellyfish.nysiis(first_word) if first_word else ""
                except Exception:
                    features[f'{col}_soundex'] = ""
                    features[f'{col}_metaphone'] = ""
                    features[f'{col}_nysiis'] = ""
        
        return features
    
    def _extract_cross_field_features(self, row: pd.Series) -> dict:
        """Extract cross-field relationship features."""
        features = {
            "name_in_email": 0,
            "surname_email_similarity": 0.0,
            "address_contains_name": 0,
            "full_name_email_similarity": 0.0,
            "dob_year_in_email": 0,

            # New, more targeted cross-field features
            "first_name_in_email": 0,
            "surname_in_email": 0,
            "first_name_email_similarity": 0.0,
            "name_tokens_email_overlap_ratio": 0.0,
            "email_local_startswith_initial_surname": 0,
            "first_name_in_address": 0,
            "full_name_in_address": 0,
        }

        surname_raw = self._safe_str(row.get("surname", ""))
        first_name_raw = self._safe_str(row.get("first_name", ""))
        email_raw = self._safe_str(row.get("email", ""))
        address_raw = self._get_address_text(row)
        dob = row.get("date_of_birth")

        local_raw, _ = self.extract_email_components(email_raw)

        surname_norm = self.normalize_text(surname_raw)
        first_norm = self.normalize_text(first_name_raw)
        address_norm = self.normalize_text(address_raw)

        local_norm = self.normalize_text(local_raw)  # strips . _ - + etc

        # Email-based cross checks
        if local_norm:
            if surname_norm:
                features["surname_in_email"] = int(surname_norm in local_norm)
            if first_norm:
                features["first_name_in_email"] = int(first_norm in local_norm)

            features["name_in_email"] = int(features["surname_in_email"] or features["first_name_in_email"])

            if surname_norm:
                features["surname_email_similarity"] = jellyfish.jaro_winkler_similarity(surname_norm, local_norm)

            if first_norm:
                features["first_name_email_similarity"] = jellyfish.jaro_winkler_similarity(first_norm, local_norm)

            if first_norm and surname_norm:
                full_name = f"{first_norm}{surname_norm}"
                features["full_name_email_similarity"] = jellyfish.jaro_winkler_similarity(full_name, local_norm)

                # Token overlap: name tokens vs email-local tokens
                name_tokens = {t for t in f"{first_norm} {surname_norm}".split() if len(t) >= 2}
                local_tokens = {t for t in re.split(r"[^a-z0-9]+", local_raw.lower()) if len(t) >= 2}
                features["name_tokens_email_overlap_ratio"] = len(name_tokens & local_tokens) / max(len(name_tokens), 1)

                # Common pattern: "hmueller", "jsmith" etc.
                local_alnum = re.sub(r"[^a-z0-9]", "", local_raw.lower())
                surname_alnum = re.sub(r"[^a-z0-9]", "", surname_norm)
                first_initial = first_norm[0] if first_norm else ""
                features["email_local_startswith_initial_surname"] = int(
                    bool(first_initial and surname_alnum and local_alnum.startswith(first_initial + surname_alnum))
                )

        # Address-based cross checks
        if address_norm:
            if surname_norm:
                features["address_contains_name"] = int(surname_norm in address_norm)
            if first_norm:
                features["first_name_in_address"] = int(first_norm in address_norm)
            if first_norm and surname_norm:
                features["full_name_in_address"] = int(f"{first_norm} {surname_norm}" in address_norm)

        # DOB year in email-local
        if pd.notna(dob) and local_raw:
            try:
                if isinstance(dob, str):
                    dob = pd.to_datetime(dob)
                year = str(dob.year)
                short_year = year[-2:]
                features["dob_year_in_email"] = int(year in local_raw or short_year in local_raw)
            except Exception:
                features["dob_year_in_email"] = 0

        return features
    
    def _extract_behavioral_features(self, row: pd.Series) -> dict:
        """Extract behavioral/consistency features."""
        features = {
            'iban_country_matches_nationality': 0,
            'email_domain_common': 0,
            'address_is_pobox': 0,
        }
        
        # Check IBAN country vs nationality
        iban = str(row.get('iban', '')) if pd.notna(row.get('iban')) else ""
        nationality = str(row.get('nationality', '')).lower() if pd.notna(row.get('nationality')) else ""
        
        iban_country_map = {
            'DE': 'german', 'GB': 'british', 'FR': 'french', 
            'ES': 'spanish', 'IT': 'italian', 'NL': 'dutch', 'US': 'american'
        }
        
        if len(iban) >= 2:
            iban_country = iban[:2].upper()
            expected_nationality = iban_country_map.get(iban_country, '')
            features['iban_country_matches_nationality'] = int(expected_nationality in nationality)
        
        # Check if email domain is common
        _, domain = self.extract_email_components(str(row.get('email', '')))
        common_domains = {'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 
                         'web.de', 'gmx.de', 'protonmail.com', 'icloud.com'}
        features['email_domain_common'] = int(domain in common_domains)
        
        # Check for PO Box / Packstation in address (German context)
        address = self._get_address_text(row).lower()
        # Includes standard PO Box terms and German specific delivery points
        pobox_terms = [
            'postfach',      # PO Box
            'packstation',   # DHL automated locker
            'postfiliale',   # Post office branch
            'paketshop',     # Parcel shop
            'paketstation',  # Parcel station
            'box'            # Generic fallback
        ]
        
        if any(term in address for term in pobox_terms):
             features['address_is_pobox'] = 1
        
        return features
    
    def _extract_entropy_features(self, row: pd.Series) -> dict:
        """Extract entropy (randomness) features."""
        features = {}
        
        for col in ['email', 'surname', 'first_name']:
            if col in row.index:
                val = str(row[col]) if pd.notna(row[col]) else ""
                if col == 'email':
                    local, _ = self.extract_email_components(val)
                    features[f'{col}_entropy'] = self.calculate_entropy(local)
                else:
                    features[f'{col}_entropy'] = self.calculate_entropy(val)
        
        return features
    
    def _extract_temporal_features(self, row: pd.Series) -> dict:
        """Extract temporal features from date of birth."""
        features = {
            'dob_is_weekend': 0,
            'dob_is_round_number': 0,
            'dob_is_jan_first': 0,
            'age_years': -1,
        }
        
        dob = row.get('date_of_birth')
        if pd.isna(dob):
            return features
        
        # Handle both date objects and strings
        if isinstance(dob, str):
            try:
                dob = pd.to_datetime(dob)
            except Exception:
                return features
        
        try:
            features['dob_is_weekend'] = int(dob.weekday() >= 5)
            features['dob_is_round_number'] = int(dob.year % 10 == 0)
            features['dob_is_jan_first'] = int(dob.month == 1 and dob.day == 1)
            
            # Calculate age (approximate)
            now = pd.Timestamp.now()
            features['age_years'] = now.year - dob.year - ((now.month, now.day) < (dob.month, dob.day))
        except Exception:
            pass
        
        return features
    
    def _extract_network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract network features (shared attributes across records).
        
        This requires the full dataset to compute.
        """
        network_features = pd.DataFrame(index=df.index)
        
        # Shared IBAN count
        if 'iban' in df.columns:
            iban_counts = df['iban'].value_counts()
            network_features['shared_iban_count'] = df['iban'].map(iban_counts).fillna(1).astype(int)
        
        # Shared address counts
        has_structured = all(c in df.columns for c in ["strasse", "hausnummer", "plz", "stadt"])
        if has_structured:
            # Normalize street/city for grouping robustness
            street_norm = df["strasse"].fillna("").astype(str).apply(self.normalize_text)
            city_norm = df["stadt"].fillna("").astype(str).apply(self.normalize_text)
            house = df["hausnummer"].fillna("").astype(str).str.strip()
            plz = df["plz"].fillna("").astype(str).str.strip()

            network_features["shared_strasse_count"] = street_norm.map(street_norm.value_counts()).fillna(1).astype(int)
            network_features["shared_stadt_count"] = city_norm.map(city_norm.value_counts()).fillna(1).astype(int)
            network_features["shared_plz_count"] = plz.map(plz.value_counts()).fillna(1).astype(int)
            network_features["shared_hausnummer_count"] = house.map(house.value_counts()).fillna(1).astype(int)

            full_key = (street_norm + "|" + house + "|" + plz + "|" + city_norm)
            network_features["shared_full_address_count"] = full_key.map(full_key.value_counts()).fillna(1).astype(int)
        elif "address" in df.columns:
            addr_counts = df["address"].value_counts()
            network_features["shared_address_count"] = df["address"].map(addr_counts).fillna(1).astype(int)
        
        # Shared email domain count
        if 'email' in df.columns:
            df_temp = df['email'].apply(lambda x: self.extract_email_components(str(x))[1])
            domain_counts = df_temp.value_counts()
            network_features['shared_email_domain_count'] = df_temp.map(domain_counts).fillna(1).astype(int)
        
        return network_features
    
    def _extract_string_stat_features(self, row: pd.Series) -> dict:
        features = {}
        targets = {
            "surname": self._safe_str(row.get("surname", "")),
            "first_name": self._safe_str(row.get("first_name", "")),
            "address": self._get_address_text(row),
            "strasse": self._safe_str(row.get("strasse", "")),
            "hausnummer": self._safe_str(row.get("hausnummer", "")),
            "plz": self._safe_str(row.get("plz", "")),
            "stadt": self._safe_str(row.get("stadt", "")),
        }

        email = self._safe_str(row.get("email", ""))
        local, _ = self.extract_email_components(email)
        targets["email_local"] = local

        for key, raw in targets.items():
            text = raw.strip()
            norm = self.normalize_text(text)

            features[f"{key}_token_count"] = 0 if not norm else len(norm.split())
            tokens = norm.split() if norm else []
            features[f"{key}_unique_token_ratio"] = (len(set(tokens)) / max(len(tokens), 1)) if tokens else 0.0

            if not text:
                features[f"{key}_unique_char_ratio"] = 0.0
                features[f"{key}_digit_ratio"] = 0.0
                features[f"{key}_alpha_ratio"] = 0.0
                features[f"{key}_whitespace_ratio"] = 0.0
                continue

            length = max(len(text), 1)
            features[f"{key}_unique_char_ratio"] = len(set(text.lower())) / length
            features[f"{key}_digit_ratio"] = sum(c.isdigit() for c in text) / length
            features[f"{key}_alpha_ratio"] = sum(c.isalpha() for c in text) / length
            features[f"{key}_whitespace_ratio"] = sum(c.isspace() for c in text) / length

        return features

    def _extract_address_features(self, row: pd.Series) -> dict:
        features = {
            "address_has_house_number": 0,
            "address_house_number_len": 0,
            "address_has_postcode": 0,
            "address_postcode_len": 0,
            "address_has_street_keyword": 0,
        }

        # Prefer structured parts
        strasse = self._safe_str(row.get("strasse", "")).strip()
        hausnummer = self._safe_str(row.get("hausnummer", "")).strip()
        plz = self._safe_str(row.get("plz", "")).strip()

        if any([strasse, hausnummer, plz]):
            if hausnummer:
                features["address_has_house_number"] = 1
                features["address_house_number_len"] = len(hausnummer)
            if plz:
                features["address_has_postcode"] = 1
                features["address_postcode_len"] = len(plz)

            street_keywords = {
                "strasse", "straße", "street", "st", "road", "rd", "avenue", "ave",
                "allee", "platz", "gasse", "weg"
            }
            norm = self.normalize_text(strasse)
            tokens = set(norm.split()) if norm else set()
            features["address_has_street_keyword"] = int(any(k in tokens for k in street_keywords))
            return features

        # Fallback: parse legacy address string
        address = self._safe_str(row.get("address", "")).strip()
        if not address:
            return features

        addr_lower = address.lower()
        m_house = re.search(r"\b\d{1,5}[a-zA-Z]?\b", addr_lower)
        if m_house:
            features["address_has_house_number"] = 1
            features["address_house_number_len"] = len(m_house.group(0))

        m_zip = re.search(r"\b\d{5}\b", addr_lower)
        if m_zip:
            features["address_has_postcode"] = 1
            features["address_postcode_len"] = len(m_zip.group(0))

        street_keywords = {
            "strasse", "straße", "street", "st", "road", "rd", "avenue", "ave",
            "allee", "platz", "gasse", "weg"
        }
        norm = self.normalize_text(address)
        tokens = set(norm.split()) if norm else set()
        features["address_has_street_keyword"] = int(any(k in tokens for k in street_keywords))

        return features
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all features from the dataset.
        
        Args:
            df: Input DataFrame with customer records.
            
        Returns:
            DataFrame with extracted features.
        """
        feature_dicts = []
        
        for idx, row in df.iterrows():
            features = {}
            
            # Extract all feature categories
            features.update(self._extract_string_length_features(row))
            features.update(self._extract_complexity_features(row))
            features.update(self._extract_email_features(row))
            features.update(self._extract_iban_features(row))
            features.update(self._extract_cross_field_features(row))
            features.update(self._extract_behavioral_features(row))
            features.update(self._extract_entropy_features(row))
            features.update(self._extract_temporal_features(row))
            features.update(self._extract_string_stat_features(row))
            features.update(self._extract_address_features(row))
            
            feature_dicts.append(features)
        
        feature_df = pd.DataFrame(feature_dicts, index=df.index)
        
        # Add phonetic features (stored as strings for blocking)
        phonetic_dicts = []
        for idx, row in df.iterrows():
            phonetic_dicts.append(self._extract_phonetic_features(row))
        phonetic_df = pd.DataFrame(phonetic_dicts, index=df.index)
        
        # Add network features if enabled
        if self.compute_network_features:
            network_df = self._extract_network_features(df)
            feature_df = pd.concat([feature_df, network_df], axis=1)
        
        # Store feature names (only numeric features)
        self._feature_names = [col for col in feature_df.columns 
                               if feature_df[col].dtype in [np.float64, np.int64, np.int32, np.float32]]
        
        # Add phonetic strings separately (not for ML but for blocking)
        for col in phonetic_df.columns:
            feature_df[col] = phonetic_df[col]
        
        return feature_df
    
    def get_numeric_features(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get only numeric features suitable for ML models.
        
        Args:
            feature_df: DataFrame from extract_features().
            
        Returns:
            DataFrame with only numeric columns.
        """
        numeric_cols = [col for col in feature_df.columns 
                       if feature_df[col].dtype in [np.float64, np.int64, np.int32, np.float32]]
        return feature_df[numeric_cols]
    
    @property
    def feature_names(self) -> list[str]:
        """Get list of numeric feature names."""
        return self._feature_names


if __name__ == "__main__":
    # Test feature extraction
    test_data = pd.DataFrame({
        "customer_id": ["C001", "C002", "C003"],
        "surname": ["Mueller", "Smith", "RandomXYZ123"],
        "first_name": ["Hans", "John", "Test"],
        "strasse": ["Main St", "Oak Ave", "Unknown Rd"],
        "hausnummer": ["1", "5", "123"],
        "plz": ["10115", "20095", "00000"],
        "stadt": ["Berlin", "Hamburg", "Nowhere"],
        "address": ["Main St 1, 10115 Berlin", "Oak Ave 5, 20095 Hamburg", "Unknown Rd 123, 00000 Nowhere"],
        "email": ["hans.mueller@gmail.com", "john@yahoo.com", "x1y2z3@fake.net"],
        "iban": ["DE89370400440532013000", "GB82WEST12345698765432", "DE11111111111111111111"],
        "date_of_birth": ["1990-01-01", "1985-06-15", "2000-01-01"],
        "nationality": ["German", "British", "German"],
    })
    
    extractor = FeatureExtractor()
    features = extractor.extract_features(test_data)
    
    print("Extracted features:")
    print(features.columns.tolist())
    print("\nNumeric features shape:", extractor.get_numeric_features(features).shape)
