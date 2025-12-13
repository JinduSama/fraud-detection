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
    
    def _extract_string_length_features(self, row: pd.Series) -> dict:
        """Extract string length features."""
        features = {}
        
        for col in ['surname', 'first_name', 'address']:
            if col in row.index:
                val = str(row[col]) if pd.notna(row[col]) else ""
                features[f'len_{col}'] = len(val)
        
        return features
    
    def _extract_complexity_features(self, row: pd.Series) -> dict:
        """Extract complexity features (digits, special chars, case, repeats, vowels)."""
        features = {}
        
        for col in ['surname', 'first_name', 'email', 'address']:
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
            'email_domain_length': 0,
            'email_local_length': 0,
            'email_has_numbers_local': 0,
            'email_num_dots_local': 0,
            'email_consonant_ratio': 0.0,
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
        
        return features
    
    def _extract_iban_features(self, row: pd.Series) -> dict:
        """Extract IBAN specific features."""
        features = {
            'iban_length': 0,
            'iban_digits_count': 0,
            'iban_letters_count': 0,
        }
        iban = str(row.get('iban', '')) if pd.notna(row.get('iban')) else ""
        if iban:
            features['iban_length'] = len(iban)
            features['iban_digits_count'] = sum(c.isdigit() for c in iban)
            features['iban_letters_count'] = sum(c.isalpha() for c in iban)
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
            'name_in_email': 0,
            'surname_email_similarity': 0.0,
            'address_contains_name': 0,
            'full_name_email_similarity': 0.0,
            'dob_year_in_email': 0,
        }
        
        surname = str(row.get('surname', '')).lower() if pd.notna(row.get('surname')) else ""
        first_name = str(row.get('first_name', '')).lower() if pd.notna(row.get('first_name')) else ""
        email = str(row.get('email', '')).lower() if pd.notna(row.get('email')) else ""
        address = str(row.get('address', '')).lower() if pd.notna(row.get('address')) else ""
        
        local, _ = self.extract_email_components(email)
        
        # Check if name appears in email
        if surname and local:
            features['name_in_email'] = int(surname in local or first_name in local)
            features['surname_email_similarity'] = jellyfish.jaro_winkler_similarity(surname, local)
            
            # Full name similarity
            full_name = f"{first_name}{surname}"
            features['full_name_email_similarity'] = jellyfish.jaro_winkler_similarity(full_name, local)
        
        # Check if name appears in address
        if surname and address:
            features['address_contains_name'] = int(surname in address)

        # Check if DOB year appears in email local part
        dob = row.get('date_of_birth')
        if pd.notna(dob) and local:
            try:
                if isinstance(dob, str):
                    dob = pd.to_datetime(dob)
                year = str(dob.year)
                short_year = year[-2:]
                features['dob_year_in_email'] = int(year in local or short_year in local)
            except Exception:
                features['dob_year_in_email'] = 0
        
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
        address = str(row.get('address', '')).lower() if pd.notna(row.get('address')) else ""
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
        
        # Shared address count
        if 'address' in df.columns:
            addr_counts = df['address'].value_counts()
            network_features['shared_address_count'] = df['address'].map(addr_counts).fillna(1).astype(int)
        
        # Shared email domain count
        if 'email' in df.columns:
            df_temp = df['email'].apply(lambda x: self.extract_email_components(str(x))[1])
            domain_counts = df_temp.value_counts()
            network_features['shared_email_domain_count'] = df_temp.map(domain_counts).fillna(1).astype(int)
        
        return network_features
    
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
        "address": ["Main St 1", "Oak Ave 5", "123 Unknown Rd"],
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
