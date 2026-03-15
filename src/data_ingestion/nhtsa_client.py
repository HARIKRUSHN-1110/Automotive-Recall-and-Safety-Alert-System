"""
nhtsa_client.py
---------------
NHTSA (National Highway Traffic Safety Administration) API client.

Responsibilities:
  - Fetch vehicle complaints from the NHTSA complaints API
  - Fetch vehicle recalls from the NHTSA recalls API
  - Handle retries, timeouts, and HTTP errors gracefully
  - Log every request and failure for observability

NHTSA API Docs: https://api.nhtsa.gov/
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


# Constants

NHTSA_BASE_URL = "https://api.nhtsa.gov"
COMPLAINTS_ENDPOINT = "/complaints/complaintsByVehicle"
RECALLS_ENDPOINT = "/recalls/recallsByVehicle"

DEFAULT_TIMEOUT = 30        # seconds per request
DEFAULT_MAX_RETRIES = 3     # total retry attempts
DEFAULT_BACKOFF_MIN = 2     # seconds — minimum wait between retries
DEFAULT_BACKOFF_MAX = 10    # seconds — maximum wait between retries

# Data Structures

@dataclass
class Complaint:
    """
    Represents a single vehicle complaint filed with NHTSA.
    These are the raw signals our ML model learns from.
    """
    odi_number: str                    # Unique NHTSA complaint ID
    make: str                          # e.g. "BMW"
    model: str                         # e.g. "3 Series"
    model_year: int                    # e.g. 2020
    component: str                     # e.g. "ENGINE AND ENGINE COOLING"
    summary: str                       # Full complaint text (ML input)
    crash: bool                        # Was there a crash?
    fire: bool                         # Was there a fire?
    injuries: int                      # Number of injuries reported
    deaths: int                        # Number of deaths reported
    date_complained: Optional[str]     # When complaint was filed
    date_of_incident: Optional[str]    # When incident occurred
    vehicle_speed: Optional[int]       # Speed at time of incident (mph)


@dataclass
class Recall:
    """
    Represents an official NHTSA recall campaign.
    These become our training labels (what we want to predict).
    """
    campaign_number: str               # e.g. "21V123000"
    manufacturer: str                  # e.g. "BMW OF NORTH AMERICA, LLC"
    make: str                          # e.g. "BMW"
    model: str                         # e.g. "3 Series"
    model_year: int                    # e.g. 2020
    component: str                     # What part failed
    summary: str                       # Recall description
    consequence: str                   # What can go wrong
    remedy: str                        # How it's being fixed
    recall_date: Optional[str]         # When recall was issued
    notes: str = ""                        # Hotline numbers, additional info from NHTSA
    park_it: bool = False                     # True = NHTSA says don't drive this vehicle


@dataclass
class FetchResult:
    """
    Wrapper returned by every fetch method.
    Bundles the data with metadata about the request itself.
    """
    success: bool
    data: list[Any] = field(default_factory=list)
    count: int = 0
    error: Optional[str] = None
    make: str = ""
    model: str = ""
    model_year: Optional[int] = None
    elapsed_seconds: float = 0.0

    def __repr__(self) -> str:
        if self.success:
            return (
                f"FetchResult(success=True, count={self.count}, "
                f"vehicle={self.make} {self.model} {self.model_year or ''}, "
                f"elapsed={self.elapsed_seconds:.2f}s)"
            )
        return f"FetchResult(success=False, error='{self.error}')"


# Custom Exceptions

class NHTSAClientError(Exception):
    """Base exception for all NHTSA client errors."""
    pass


class NHTSAAPIError(NHTSAClientError):
    """Raised when the NHTSA API returns an HTTP error (4xx, 5xx)."""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {message}")

class NHTSAClientRequestError(NHTSAAPIError):
    """
    Raised on 4xx errors (bad request, not found, etc.).
    NOT retried — the request is wrong, retrying won't help.
    e.g. invalid model name, malformed params.
    """
    pass

class NHTSAServerError(NHTSAAPIError):
    """
    Raised on 5xx errors (server unavailable, internal error).
    IS retried — the server might recover after a short wait.
    """
    pass

class NHTSATimeoutError(NHTSAClientError):
    """Raised when a request times out after all retries are exhausted."""
    pass


class NHTSAConnectionError(NHTSAClientError):
    """Raised when a network connection error occurs."""
    pass


# Client

class NHTSAClient:
    """
    HTTP client for the NHTSA public API.

    Usage:
        client = NHTSAClient()

        # Fetch complaints for a specific vehicle
        result = client.fetch_complaints(make="BMW", model="3 Series", year=2020)
        if result.success:
            for complaint in result.data:
                print(complaint.summary)

        # Fetch recalls for a specific vehicle
        result = client.fetch_recalls(make="BMW", model="3 Series", year=2020)
    """

    def __init__(
        self,
        base_url: str = NHTSA_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_min: int = DEFAULT_BACKOFF_MIN,
        backoff_max: int = DEFAULT_BACKOFF_MAX,
    ):
        """
        Initialise the NHTSA client.

        Args:
            base_url:    Root URL of the NHTSA API.
            timeout:     Seconds to wait before declaring a request timed out.
            max_retries: How many times to retry a failed request.
            backoff_min: Minimum seconds to wait between retries.
            backoff_max: Maximum seconds to wait between retries.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_min = backoff_min
        self.backoff_max = backoff_max

        # Build a requests.Session with connection-level retries.
        # This handles low-level TCP failures (connection refused, DNS errors).
        # Tenacity (below) handles higher-level retries (timeouts, 5xx errors).
        self.session = self._build_session()

        logger.info(
            "NHTSAClient initialised — base_url=%s, timeout=%ss, max_retries=%d",
            self.base_url, self.timeout, self.max_retries,
        )

    # Public Methods 

    def fetch_complaints(
        self,
        make: str,
        model: str,
        year: int,
    ) -> FetchResult:
        """
        Fetch all complaints filed with NHTSA for a specific vehicle.

        Args:
            make:  Vehicle manufacturer, e.g. "BMW" or "TOYOTA"
            model: Vehicle model name, e.g. "3 Series" or "Camry"
            year:  Model year as an integer, e.g. 2020

        Returns:
            FetchResult containing a list of Complaint objects,
            or an error message if the request failed.

        Example:
            result = client.fetch_complaints("BMW", "3 Series", 2020)
            # result.data → [Complaint(...), Complaint(...), ...]
        """
        start = time.perf_counter()
        url = f"{self.base_url}{COMPLAINTS_ENDPOINT}"
        params = {
            "make": make.strip().upper(),
            "model": model.strip(),
            "modelYear": str(year),
        }

        logger.info("Fetching complaints — %s %s %d", make, model, year)

        try:
            raw = self._make_request(url, params)
            complaints = [
                self._parse_complaint(item, make, model, year)
                for item in raw.get("results", [])
            ]
            elapsed = time.perf_counter() - start
            logger.info(
                "Fetched %d complaints for %s %s %d (%.2fs)",
                len(complaints), make, model, year, elapsed,
            )
            return FetchResult(
                success=True,
                data=complaints,
                count=len(complaints),
                make=make,
                model=model,
                model_year=year,
                elapsed_seconds=elapsed,
            )

        except Exception as exc:
            elapsed = time.perf_counter() - start
            error_msg = str(exc)
            logger.error(
                "Failed to fetch complaints for %s %s %d: %s",
                make, model, year, error_msg,
            )
            return FetchResult(
                success=False,
                error=error_msg,
                make=make,
                model=model,
                model_year=year,
                elapsed_seconds=elapsed,
            )

    def fetch_recalls(
        self,
        make: str,
        model: str,
        year: int,
    ) -> FetchResult:
        """
        Fetch all official recall campaigns from NHTSA for a specific vehicle.

        Args:
            make:  Vehicle manufacturer, e.g. "BMW" or "TOYOTA"
            model: Vehicle model name, e.g. "3 Series" or "Camry"
            year:  Model year as an integer, e.g. 2020

        Returns:
            FetchResult containing a list of Recall objects,
            or an error message if the request failed.

        Example:
            result = client.fetch_recalls("BMW", "3 Series", 2020)
            # result.data → [Recall(...), Recall(...), ...]
        """
        start = time.perf_counter()
        url = f"{self.base_url}{RECALLS_ENDPOINT}"
        params = {
            "make": make.strip().upper(),
            "model": model.strip(),
            "modelYear": str(year),
        }

        logger.info("Fetching recalls — %s %s %d", make, model, year)

        try:
            raw = self._make_request(url, params)
            recalls = [
                self._parse_recall(item, make, model, year)
                for item in raw.get("results", [])
            ]
            elapsed = time.perf_counter() - start
            logger.info(
                "Fetched %d recalls for %s %s %d (%.2fs)",
                len(recalls), make, model, year, elapsed,
            )
            return FetchResult(
                success=True,
                data=recalls,
                count=len(recalls),
                make=make,
                model=model,
                model_year=year,
                elapsed_seconds=elapsed,
            )

        except Exception as exc:
            elapsed = time.perf_counter() - start
            error_msg = str(exc)
            logger.error(
                "Failed to fetch recalls for %s %s %d: %s",
                make, model, year, error_msg,
            )
            return FetchResult(
                success=False,
                error=error_msg,
                make=make,
                model=model,
                model_year=year,
                elapsed_seconds=elapsed,
            )

    def health_check(self) -> bool:
        """
        Verify the NHTSA API is reachable.
        Uses a lightweight known-good request (BMW 3 Series complaints).

        Returns:
            True if the API responds successfully, False otherwise.
        """
        logger.info("Running NHTSA API health check...")
        result = self.fetch_complaints("BMW", "3 Series", 2020)
        if result.success:
            logger.info("Health check passed — API is reachable")
        else:
            logger.warning("Health check failed — %s", result.error)
        return result.success

    # ── Private Methods ────────────────────────────────────────

    def _build_session(self) -> requests.Session:
        """
        Build a requests.Session with connection-level retry handling.

        Why use a Session?
        - Reuses the underlying TCP connection across requests (faster)
        - Automatically retries on connection-level failures (DNS, refused)
        - Sets consistent headers for all requests
        """
        session = requests.Session()

        # urllib3-level retry — handles connection errors and resets.
        # This is separate from tenacity, which handles application-level retries.
        urllib3_retry = Retry(
            total=2,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=urllib3_retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        session.headers.update({
            "Accept": "application/json",
            "User-Agent": "AutomotiveRecallSystem/1.0 (research project)",
        })
        return session

    def _make_request(self, url: str, params: dict) -> dict:
        """
        Execute a GET request with tenacity-powered retry logic.

        Retry strategy:
          - Retries on: requests.Timeout, requests.ConnectionError, NHTSAAPIError (5xx)
          - Does NOT retry on: 4xx errors (bad request, not found) — these won't improve
          - Waits: exponential backoff (2s → 4s → 8s ... up to backoff_max)
          - Stops: after max_retries attempts

        Args:
            url:    Full endpoint URL to request.
            params: Query parameters dict (make, model, modelYear).

        Returns:
            Parsed JSON response as a dict.

        Raises:
            NHTSAAPIError:        On non-retryable HTTP errors.
            NHTSATimeoutError:    When all retry attempts time out.
            NHTSAConnectionError: When all retry attempts fail due to network errors.
        """

        # defining the retry decorator dynamically so it can use self.* config.
        @retry(
            retry=retry_if_exception_type(
                (requests.Timeout, requests.ConnectionError, NHTSAAPIError)
            ),
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(
                multiplier=1,
                min=self.backoff_min,
                max=self.backoff_max,
            ),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=False,
        )
        def _execute() -> dict:
            logger.debug("GET %s — params: %s", url, params)
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout,
                )
            except requests.Timeout as exc:
                logger.warning("Request timed out: %s", exc)
                raise

            except requests.ConnectionError as exc:
                logger.warning("Connection error: %s", exc)
                raise

            if not response.ok:
                if response.status_code == 400:
                    logger.debug("400 - no data for this vehicle: %s", params)
                    return {"results": []}
                
                if 401 <= response.status_code < 500:
                    # Other 4xx — genuine bad request, no retry
                    raise NHTSAClientRequestError(
                        response.status_code,
                        f"Client error for URL {url} — params {params}",
                    )
                raise NHTSAAPIError(
                    response.status_code,
                    f"Server error for URL {url}",
                )

            return response.json()

        try:
            return _execute()

        except RetryError as exc:
            cause = exc.last_attempt.exception()
            if isinstance(cause, requests.Timeout):
                raise NHTSATimeoutError(
                    f"Request timed out after {self.max_retries} attempts: {url}"
                ) from exc
            if isinstance(cause, requests.ConnectionError):
                raise NHTSAConnectionError(
                    f"Connection failed after {self.max_retries} attempts: {url}"
                ) from exc
            raise NHTSAClientError(
                f"Request failed after {self.max_retries} attempts: {cause}"
            ) from exc

        except NHTSAAPIError:
            raise

    #  Parsers 

    def _parse_complaint(
        self,
        raw: dict,
        make: str,
        model: str,
        year: int,
    ) -> Complaint:
        """
        Convert a raw NHTSA complaint dict into a typed Complaint object.

        Why a separate parser?
        - Centralises all field-name mapping in one place.
        - Provides safe defaults so a missing field never crashes the pipeline.
        - If NHTSA changes their field names, we fix it here only.

        NHTSA complaint fields reference:
          odiNumber, make, model, modelYear, components, summary,
          crash, fire, numberOfInjuries, numberOfDeaths,
          dateComplaintFiled, dateOfIncident, speed
        """
        return Complaint(
            odi_number=str(raw.get("odiNumber", "")),
            make=raw.get("make", make).upper(),
            model=raw.get("model", model),
            model_year=int(raw.get("modelYear", year)),
            component=raw.get("components", "UNKNOWN"),
            summary=raw.get("summary", "").strip(),
            crash=bool(raw.get("crash", False)),
            fire=bool(raw.get("fire", False)),
            injuries=int(raw.get("numberOfInjuries", 0) or 0),
            deaths=int(raw.get("numberOfDeaths", 0) or 0),
            date_complained=raw.get("dateComplaintFiled"),
            date_of_incident=raw.get("dateOfIncident"),
            vehicle_speed=raw.get("speed"),
        )

    def _parse_recall(
        self,
        raw: dict,
        make: str,
        model: str,
        year: int,
    ) -> Recall:
        """
        Convert a raw NHTSA recall dict into a typed Recall object.

        NHTSA recall fields reference:
          NHTSACampaignNumber, Manufacturer, Make, Model, ModelYear,
          Component, Summary, Consequence, Remedy, ReportReceivedDate
        """
        return Recall(
            campaign_number=str(raw.get("NHTSACampaignNumber", "")),
            manufacturer=(raw.get("Manufacturer") or "").upper(),
            make=(raw.get("Make") or make).upper(),
            model=raw.get("Model") or model,
            model_year=int(raw.get("ModelYear") or year),
            component=raw.get("Component") or "UNKNOWN",
            summary=(raw.get("Summary") or "").strip(),
            consequence=(raw.get("Consequence") or "").strip(),
            remedy=(raw.get("Remedy") or "").strip(),
            recall_date=raw.get("ReportReceivedDate"),
            notes=(raw.get("Notes") or "").strip(),
            park_it=bool(raw.get("parkIt", False)),
        )

    # Dunder methods

    def __repr__(self) -> str:
        return (
            f"NHTSAClient(base_url='{self.base_url}', "
            f"timeout={self.timeout}s, max_retries={self.max_retries})"
        )

    def __enter__(self):
        """Support usage as a context manager: `with NHTSAClient() as client:`"""
        return self

    def __exit__(self, *args):
        """Close the underlying HTTP session when used as a context manager."""
        self.session.close()
        logger.debug("NHTSAClient session closed")