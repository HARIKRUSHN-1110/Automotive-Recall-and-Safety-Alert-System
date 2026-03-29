"""
test_preprocessing.py - Unit tests for the text preprocessing pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.data_processing.text_processing import TextPreprocessor

@pytest.fixture
def processor():
    """Create a TextPreprocessor instance for testing."""
    return TextPreprocessor()

class TestCleanText:
    """Tests for the clean_text() function."""

    def test_lowercase(self, processor):
        """Text should be lowercased."""
        result = processor.clean_text("ENGINE STALLED")
        assert result == result.lower()

    def test_removes_urls(self, processor):
        """URLs should be stripped out."""
        text   = "See www.nhtsa.gov for details"
        result = processor.clean_text(text)
        assert "www" not in result
        assert "nhtsa" not in result

    def test_removes_vin(self, processor):
        """17-char VIN numbers should be removed."""
        text   = "VIN 1HGBH41JXMN109186 had issue"
        result = processor.clean_text(text)
        assert "1HGBH41JXMN109186" not in result

    def test_removes_dates(self, processor):
        """Date strings should be removed."""
        text   = "Filed on 03/15/2023 about engine"
        result = processor.clean_text(text)
        assert "03/15/2023" not in result

    def test_removes_punctuation(self, processor):
        """Punctuation should be replaced with spaces."""
        text   = "engine stalled!!!"
        result = processor.clean_text(text)
        assert "!" not in result

    def test_handles_none(self, processor):
        """None input should return empty string, not crash."""
        result = processor.clean_text(None)
        assert result == ""

    def test_handles_empty_string(self, processor):
        """Empty string should return empty string."""
        result = processor.clean_text("")
        assert result == ""

    def test_collapses_whitespace(self, processor):
        """Multiple spaces should collapse to single space."""
        result = processor.clean_text("engine   stalled   highway")
        assert "  " not in result


class TestTokenize:
    """Tests for the tokenize() function."""

    def test_splits_into_words(self, processor):
        """Should return a list of individual words."""
        result = processor.tokenize("engine stalled highway")
        assert isinstance(result, list)
        assert "engine" in result
        assert "stalled" in result

    def test_removes_non_alpha(self, processor):
        """Non-alphabetic tokens should be filtered out."""
        result = processor.tokenize("engine 123 stalled")
        assert "123" not in result

    def test_empty_input(self, processor):
        """Empty string should return empty list."""
        result = processor.tokenize("")
        assert result == []


class TestRemoveStopwords:
    """Tests for the remove_stopwords() function."""

    def test_removes_common_words(self, processor):
        """Common English stopwords should be removed."""
        tokens = ["the", "engine", "was", "stalled", "on", "highway"]
        result = processor.remove_stopwords(tokens)
        assert "the" not in result
        assert "was" not in result
        assert "on" not in result

    def test_keeps_signal_words(self, processor):
        """Meaningful words should be kept."""
        tokens = ["engine", "stalled", "highway", "fire"]
        result = processor.remove_stopwords(tokens)
        assert "engine" in result
        assert "stalled" in result

    def test_removes_short_tokens(self, processor):
        """Tokens shorter than 3 chars should be removed."""
        tokens = ["it", "to", "be", "engine"]
        result = processor.remove_stopwords(tokens)
        assert "it" not in result
        assert "engine" in result

    def test_removes_automotive_noise(self, processor):
        """Domain-specific noise words should be removed."""
        tokens = ["dealer", "contacted", "engine", "stalled"]
        result = processor.remove_stopwords(tokens)
        assert "dealer" not in result
        assert "engine" in result


class TestFullPipeline:
    """Tests for the full process() pipeline."""

    def test_returns_string(self, processor):
        """process() should always return a string."""
        result = processor.process("THE ENGINE STALLED on highway!!!")
        assert isinstance(result, str)

    def test_meaningful_output(self, processor):
        """A real complaint should produce meaningful tokens."""
        complaint = (
            "THE ENGINE STALLED ON THE HIGHWAY AT 65MPH. "
            "Very dangerous situation. Called dealer 03/15/2023."
        )
        result = processor.process(complaint)
        words = result.split()
        assert len(words) >= 2
        assert "engine" in result or "stalled" in result or "highway" in result

    def test_handles_none_gracefully(self, processor):
        """None input should not crash the pipeline."""
        result = processor.process(None)
        assert isinstance(result, str)

    def test_is_usable_threshold(self, processor):
        """Short cleaned text should fail the usability check."""
        assert processor.is_usable("ok good") is False
        assert processor.is_usable("engine stalled highway dangerous fire") is True