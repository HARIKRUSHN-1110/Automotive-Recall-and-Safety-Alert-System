"""
test_nhtsa_client.py

Unit tests for the NHTSA API client.
Uses mocking so no real API calls are made — tests run fast and don't depend on network access.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock
from src.data_ingestion.nhtsa_client import (
    NHTSAClient,
    Complaint,
    Recall,
    FetchResult,
    NHTSATimeoutError,
    NHTSAConnectionError,
)
import requests

@pytest.fixture
def client():
    """Create a NHTSAClient with fast retry settings for testing."""
    return NHTSAClient(max_retries=1, backoff_min=0, backoff_max=0)


class TestFetchComplaints:
    """Tests for fetch_complaints()."""

    def test_success_returns_fetch_result(self, client):
        """Successful API call should return FetchResult with success=True."""
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "results": [
                {
                    "odiNumber":          "12345",
                    "make":               "TOYOTA",
                    "model":              "Camry",
                    "modelYear":          2020,
                    "components":         "ENGINE",
                    "summary":            "Engine stalled on highway",
                    "crash":              False,
                    "fire":               False,
                    "numberOfInjuries":   0,
                    "numberOfDeaths":     0,
                    "dateComplaintFiled": "01/01/2023",
                }
            ]
        }

        with patch.object(client.session, "get", return_value=mock_response):
            result = client.fetch_complaints("TOYOTA", "Camry", 2020)

        assert result.success is True
        assert result.count == 1
        assert isinstance(result.data[0], Complaint)
        assert result.data[0].odi_number == "12345"

    def test_empty_results_returns_success(self, client):
        """API returning empty results should still be success=True."""
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"results": []}

        with patch.object(client.session, "get", return_value=mock_response):
            result = client.fetch_complaints("FERRARI", "Roma", 2022)

        assert result.success is True
        assert result.count == 0
        assert result.data == []

    def test_400_returns_empty_not_error(self, client):
        """
        400 response should return empty results gracefully.
        NHTSA returns 400 for unknown vehicles — not a real error.
        """
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 400

        with patch.object(client.session, "get", return_value=mock_response):
            result = client.fetch_complaints("BMW", "3 Series", 2025)

        assert result.success is True
        assert result.count == 0

    def test_api_failure_returns_error_result(self, client):
        """Network failure should return FetchResult with success=False."""
        with patch.object(
            client.session, "get",
            side_effect=requests.ConnectionError("Network down")
        ):
            result = client.fetch_complaints("TOYOTA", "Camry", 2020)

        assert result.success is False
        assert result.error is not None


class TestParseComplaint:
    """Tests for the complaint parser."""

    def test_parses_all_fields(self, client):
        """All fields should be parsed correctly from raw dict."""
        raw = {
            "odiNumber":          "99999",
            "make":               "FORD",
            "model":              "F-150",
            "modelYear":          2019,
            "components":         "ENGINE AND ENGINE COOLING",
            "summary":            "Engine stalled without warning",
            "crash":              True,
            "fire":               False,
            "numberOfInjuries":   1,
            "numberOfDeaths":     0,
            "dateComplaintFiled": "06/01/2023",
            "dateOfIncident":     "05/15/2023",
            "speed":              65,
        }
        complaint = client._parse_complaint(raw, "FORD", "F-150", 2019)

        assert complaint.odi_number == "99999"
        assert complaint.make       == "FORD"
        assert complaint.crash      is True
        assert complaint.injuries   == 1
        assert complaint.vehicle_speed == 65

    def test_handles_none_fields_gracefully(self, client):
        """Missing fields should use safe defaults without crashing."""
        raw = {"odiNumber": "11111"}
        complaint = client._parse_complaint(raw, "BMW", "X5", 2020)

        assert complaint.odi_number == "11111"
        assert complaint.injuries   == 0
        assert complaint.crash      is False


class TestParseRecall:
    """Tests for the recall parser."""

    def test_parses_all_fields(self, client):
        """All recall fields should be parsed from raw dict."""
        raw = {
            "NHTSACampaignNumber": "21V123000",
            "Manufacturer":        "Toyota Motor Corp",
            "Make":                "TOYOTA",
            "Model":               "Camry",
            "ModelYear":           2020,
            "Component":           "AIR BAGS",
            "Summary":             "Airbag may not deploy",
            "Consequence":         "Injury risk",
            "Remedy":              "Replace inflator",
            "Notes":               "Contact dealer",
            "ReportReceivedDate":  "01/15/2022",
            "parkIt":              False,
        }
        recall = client._parse_recall(raw, "TOYOTA", "Camry", 2020)

        assert recall.campaign_number == "21V123000"
        assert recall.component       == "AIR BAGS"
        assert recall.park_it         is False

    def test_handles_none_consequence(self, client):
        """None fields like Consequence should not crash the parser."""
        raw = {
            "NHTSACampaignNumber": "22V000001",
            "Manufacturer":        "Honda",
            "Make":                "HONDA",
            "Model":               "Civic",
            "ModelYear":           2021,
            "Component":           "ENGINE",
            "Summary":             "Engine issue",
            "Consequence":         None,
            "Remedy":              None,
            "Notes":               None,
            "ReportReceivedDate":  "03/01/2023",
            "parkIt":              False,
        }
        recall = client._parse_recall(raw, "HONDA", "Civic", 2021)

        assert recall.consequence == ""
        assert recall.remedy      == ""
        assert recall.notes       == ""


class TestClientConfig:
    """Tests for NHTSAClient configuration."""

    def test_default_base_url(self, client):
        assert "nhtsa.gov" in client.base_url

    def test_custom_timeout(self):
        custom = NHTSAClient(timeout=5)
        assert custom.timeout == 5

    def test_repr(self, client):
        assert "NHTSAClient" in repr(client)

    def test_context_manager(self):
        """Client should work as a context manager."""
        with NHTSAClient() as c:
            assert c is not None