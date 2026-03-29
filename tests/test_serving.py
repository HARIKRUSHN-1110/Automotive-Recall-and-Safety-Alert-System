"""
test_serving.py

Unit tests for the model serving module.
Tests risk label thresholds, input validation, and feature building logic — without requiring actual model files to be present.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.models.serve import (
    ModelServer,
    PredictionInput,
    PredictionResult,
    InvalidInputError,
)


@pytest.fixture
def server():
    """Create a ModelServer instance (not loaded — no model files needed)."""
    return ModelServer()


class TestRiskLabel:
    """Tests for the _risk_label() scoring thresholds."""

    def test_low_risk_below_40(self, server):
        assert server._risk_label(0)  == "Low"
        assert server._risk_label(20) == "Low"
        assert server._risk_label(39) == "Low"

    def test_medium_risk_40_to_69(self, server):
        assert server._risk_label(40) == "Medium"
        assert server._risk_label(55) == "Medium"
        assert server._risk_label(69) == "Medium"

    def test_high_risk_70_and_above(self, server):
        assert server._risk_label(70)  == "High"
        assert server._risk_label(85)  == "High"
        assert server._risk_label(100) == "High"


class TestInputValidation:
    """Tests for the _validate() input guard."""

    def test_valid_input_passes(self, server):
        """Valid input should return a PredictionInput without raising."""
        result = server._validate(
            make      = "BMW",
            model     = "3 Series",
            year      = 2020,
            summary   = "engine stalled on highway",
            component = "ENGINE",
            crash     = False,
            fire      = False,
        )
        assert isinstance(result, PredictionInput)
        assert result.make == "BMW"

    def test_empty_make_raises(self, server):
        """Empty make should raise InvalidInputError."""
        with pytest.raises(InvalidInputError):
            server._validate("", "Camry", 2020, "engine failed", "ENGINE", False, False)

    def test_none_make_raises(self, server):
        """None make should raise InvalidInputError."""
        with pytest.raises(InvalidInputError):
            server._validate(None, "Camry", 2020, "engine failed", "ENGINE", False, False)

    def test_year_too_old_raises(self, server):
        """Year before 2000 should raise InvalidInputError."""
        with pytest.raises(InvalidInputError):
            server._validate("BMW", "3 Series", 1800, "engine failed", "ENGINE", False, False)

    def test_year_too_future_raises(self, server):
        """Year beyond 2026 should raise InvalidInputError."""
        with pytest.raises(InvalidInputError):
            server._validate("BMW", "3 Series", 2099, "engine failed", "ENGINE", False, False)

    def test_empty_summary_raises(self, server):
        """Empty summary should raise InvalidInputError."""
        with pytest.raises(InvalidInputError):
            server._validate("BMW", "3 Series", 2020, "", "ENGINE", False, False)

    def test_none_summary_raises(self, server):
        """None summary should raise InvalidInputError."""
        with pytest.raises(InvalidInputError):
            server._validate("BMW", "3 Series", 2020, None, "ENGINE", False, False)

    def test_make_uppercased(self, server):
        """Make should be uppercased regardless of input case."""
        result = server._validate(
            "bmw", "3 Series", 2020, "engine stalled", "ENGINE", False, False
        )
        assert result.make == "BMW"

    def test_unknown_component_default(self, server):
        """None component should default to UNKNOWN."""
        result = server._validate(
            "BMW", "3 Series", 2020, "engine stalled", None, False, False
        )
        assert result.component == "UNKNOWN"


class TestPredictionResult:
    """Tests for the PredictionResult dataclass."""

    def test_risk_color_low(self):
        result = PredictionResult(
            risk_score=20, probability=0.2, prediction=0,
            risk_label="Low", threshold=0.338, elapsed_ms=10.0
        )
        assert result.risk_color == "green"

    def test_risk_color_medium(self):
        result = PredictionResult(
            risk_score=55, probability=0.55, prediction=1,
            risk_label="Medium", threshold=0.338, elapsed_ms=10.0
        )
        assert result.risk_color == "orange"

    def test_risk_color_high(self):
        result = PredictionResult(
            risk_score=80, probability=0.8, prediction=1,
            risk_label="High", threshold=0.338, elapsed_ms=10.0
        )
        assert result.risk_color == "red"


class TestModelServerConfig:
    """Tests for ModelServer configuration."""

    def test_default_threshold(self, server):
        """Default threshold should be 0.338."""
        assert server.threshold == 0.338

    def test_not_loaded_initially(self, server):
        """Model should not be loaded before load_model() is called."""
        assert server.is_loaded() is False

    def test_custom_threshold(self):
        """Custom threshold should be accepted."""
        custom_server = ModelServer(threshold=0.5)
        assert custom_server.threshold == 0.5

    def test_repr_shows_status(self, server):
        """__repr__ should show not loaded status."""
        assert "not loaded" in repr(server)