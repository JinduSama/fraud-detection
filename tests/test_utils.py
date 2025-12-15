"""Unit tests for shared text utilities."""

import pytest

from src.utils.text import StringDistanceMetrics, normalize_text


class TestNormalizeText:
    """Tests for text normalization function."""

    def test_lowercase_conversion(self):
        """Test text is converted to lowercase."""
        assert normalize_text("HELLO") == "hello"
        assert normalize_text("MiXeD") == "mixed"

    def test_unicode_normalization(self):
        """Test Unicode characters are normalized."""
        assert normalize_text("Müller") == "muller"
        assert normalize_text("café") == "cafe"
        assert normalize_text("naïve") == "naive"

    def test_special_chars_removed(self):
        """Test special characters are removed (not replaced with spaces)."""
        assert normalize_text("hello-world") == "helloworld"
        assert normalize_text("test@email.com") == "testemailcom"
        assert normalize_text("name (alias)") == "name alias"

    def test_whitespace_collapsed(self):
        """Test multiple whitespaces are collapsed."""
        assert normalize_text("hello   world") == "hello world"
        assert normalize_text("  leading  trailing  ") == "leading trailing"

    def test_empty_input(self):
        """Test empty string input."""
        assert normalize_text("") == ""

    def test_none_input(self):
        """Test None input returns empty string."""
        assert normalize_text(None) == ""

    def test_non_string_input(self):
        """Test non-string input returns empty string."""
        assert normalize_text(123) == ""
        # Note: Arrays/lists are not supported by pd.isna check, 
        # so we only test scalar non-string values


class TestStringDistanceMetrics:
    """Tests for string distance metric functions."""

    def test_jaro_winkler_identical(self):
        """Test identical strings have distance 0."""
        assert StringDistanceMetrics.jaro_winkler_distance("hello", "hello") == 0.0

    def test_jaro_winkler_completely_different(self):
        """Test completely different strings have high distance."""
        distance = StringDistanceMetrics.jaro_winkler_distance("abc", "xyz")
        assert distance > 0.5

    def test_jaro_winkler_typo(self):
        """Test small typo has low distance."""
        distance = StringDistanceMetrics.jaro_winkler_distance("MARTHA", "MARHTA")
        assert distance < 0.1  # Transposition should be low distance

    def test_jaro_winkler_empty_string(self):
        """Test empty string returns distance 1.0."""
        assert StringDistanceMetrics.jaro_winkler_distance("", "hello") == 1.0
        assert StringDistanceMetrics.jaro_winkler_distance("hello", "") == 1.0
        assert StringDistanceMetrics.jaro_winkler_distance("", "") == 1.0

    def test_levenshtein_identical(self):
        """Test identical strings have distance 0."""
        assert StringDistanceMetrics.levenshtein_distance_normalized("test", "test") == 0.0

    def test_levenshtein_one_edit(self):
        """Test single character edit."""
        # "kitten" -> "sitten" = 1 edit, max_len = 6, distance = 1/6 ≈ 0.167
        distance = StringDistanceMetrics.levenshtein_distance_normalized("kitten", "sitten")
        assert 0.15 < distance < 0.2

    def test_levenshtein_empty_string(self):
        """Test empty string handling."""
        assert StringDistanceMetrics.levenshtein_distance_normalized("", "hello") == 1.0
        assert StringDistanceMetrics.levenshtein_distance_normalized("", "") == 0.0

    def test_damerau_levenshtein_transposition(self):
        """Test transposition is counted as single edit."""
        # "CA" -> "AC" = 1 transposition, max_len = 2, distance = 0.5
        distance = StringDistanceMetrics.damerau_levenshtein_normalized("CA", "AC")
        assert distance == 0.5

    def test_damerau_levenshtein_vs_levenshtein(self):
        """Test Damerau-Levenshtein handles transpositions better."""
        # For transpositions, Damerau should give lower distance
        s1, s2 = "abcd", "abdc"  # Transposition of c and d
        damerau = StringDistanceMetrics.damerau_levenshtein_normalized(s1, s2)
        levenshtein = StringDistanceMetrics.levenshtein_distance_normalized(s1, s2)
        assert damerau <= levenshtein

    def test_get_similarity(self):
        """Test similarity helper method."""
        sim = StringDistanceMetrics.get_similarity("hello", "hello")
        assert sim == 1.0

        sim = StringDistanceMetrics.get_similarity("abc", "xyz")
        assert sim < 0.5


class TestAddressUtilities:
    """Tests for address normalization utilities."""

    def test_normalize_address_structured(self):
        """Test structured address normalization."""
        from src.utils.address import normalize_address

        result = normalize_address(
            strasse="Hauptstraße",
            hausnummer="42",
            plz="12345",
            stadt="Berlin",
        )
        assert result == "hauptstraße|42|12345|berlin"

    def test_normalize_address_fallback(self):
        """Test fallback to full address string."""
        from src.utils.address import normalize_address

        result = normalize_address(address="123 Main Street, City")
        assert result == "123 main street, city"

    def test_format_address(self):
        """Test address formatting."""
        from src.utils.address import format_address

        result = format_address("Hauptstraße", "42", "12345", "Berlin")
        assert result == "Hauptstraße 42, 12345 Berlin"

    def test_format_address_partial(self):
        """Test partial address formatting."""
        from src.utils.address import format_address

        result = format_address("Hauptstraße", "42", None, None)
        assert result == "Hauptstraße 42"

        result = format_address(None, None, "12345", "Berlin")
        assert result == "12345 Berlin"
