"""
test_database.py - Integration tests for the DatabaseManager.

These tests require a real database connection.
They are skipped in CI (no DB available) but
run locally and against Supabase in staging.

Run locally:
    pytest tests/test_database.py -v

Skipped in CI via:
    pytest -k "not test_database"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from dotenv import load_dotenv
load_dotenv()

from src.data_ingestion.database import DatabaseManager
from src.data_ingestion.nhtsa_client import Complaint, Recall

# Mark all tests in this file as database tests
# CI skips them with: pytest -k "not test_database"
pytestmark = pytest.mark.database


@pytest.fixture(scope="module")
def db():
    """Create a DatabaseManager connected to the real database."""
    return DatabaseManager()


class TestDatabaseConnection:
    """Tests for database connectivity."""

    def test_database_connects(self, db):
        """Database should connect without raising an exception."""
        count = db.count_complaints()
        assert count is not None

    def test_complaints_exist(self, db):
        """Database should have complaints from ingestion."""
        count = db.count_complaints()
        assert count > 0, f"Expected complaints in DB, got {count}"

    def test_recalls_exist(self, db):
        """Database should have recalls from ingestion."""
        count = db.count_recalls()
        assert count > 0, f"Expected recalls in DB, got {count}"

    def test_get_stats_returns_all_fields(self, db):
        """get_stats() should return all expected keys."""
        stats = db.get_stats()
        assert "total_complaints" in stats
        assert "total_recalls"    in stats
        assert "manufacturers"    in stats
        assert "year_range"       in stats

    def test_manufacturers_not_empty(self, db):
        """Should have at least one manufacturer."""
        stats = db.get_stats()
        assert len(stats["manufacturers"]) > 0

    def test_get_complaints_returns_list(self, db):
        """get_complaints() should return a list for known vehicle."""
        rows = db.get_complaints("TOYOTA", "Camry", 2020)
        assert isinstance(rows, list)

    def test_complaint_row_has_required_fields(self, db):
        """Each complaint row should have the critical ML fields."""
        rows = db.get_complaints("TOYOTA", "Camry", 2020, limit=1)
        if rows:
            row = rows[0]
            assert "odi_number"  in row
            assert "summary"     in row
            assert "make"        in row
            assert "model_year"  in row
            assert "component"   in row

    def test_complaint_exists_check(self, db):
        """complaint_exists() should work for known and unknown ODIs."""
        # An ODI that definitely doesn't exist
        assert db.complaint_exists("FAKE-ODI-999999") is False

    def test_recall_exists_check(self, db):
        """recall_exists() should work for known and unknown campaigns."""
        assert db.recall_exists("FAKE-CAMPAIGN-999") is False