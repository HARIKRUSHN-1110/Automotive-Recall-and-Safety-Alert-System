"""
data_ingestion.py
-----------------
Main data pipeline — fetches complaints and recalls from NHTSA
and stores them in the SQLite database.

This is the orchestrator. It coordinates NHTSAClient (fetch)
and DatabaseManager (store) across all manufacturers, models,
and years defined in the config.

Main entry point:
    from src.data_ingestion.data_ingestion import DataIngestionPipeline

    pipeline = DataIngestionPipeline()
    pipeline.ingest_all_data()
"""

import logging
import time
from dataclasses import dataclass, field
from dotenv import load_dotenv
load_dotenv()
from src.data_ingestion.database import DatabaseManager
from src.data_ingestion.nhtsa_client import NHTSAClient
import os
logger = logging.getLogger(__name__)

# Vehicle Catalogue
# Top manufacturers + their most complained-about models
# Source: NHTSA complaint volume analysis

VEHICLE_CATALOGUE = {
    "TOYOTA":       ["Camry", "Corolla", "RAV4", "Highlander", "Tacoma"],
    "HONDA":        ["Civic", "Accord", "CR-V", "Pilot", "Odyssey"],
    "FORD":         ["F-150", "Explorer", "Escape", "Fusion", "Mustang"],
    "CHEVROLET":    ["Silverado", "Equinox", "Malibu", "Traverse", "Tahoe"],
    "BMW":          ["3 Series", "5 Series", "X3", "X5", "7 Series"],
    "MERCEDES-BENZ":["C-Class", "E-Class", "GLE", "GLC", "S-Class"],
    "VOLKSWAGEN":   ["Jetta", "Passat", "Tiguan", "Atlas", "Golf"],
    "AUDI":         ["A4", "A6", "Q5", "Q7", "A3"],
    "NISSAN":       ["Altima", "Rogue", "Sentra", "Pathfinder", "Murano"],
    "HYUNDAI":      ["Elantra", "Sonata", "Tucson", "Santa Fe", "Kona"],
    "KIA":          ["Optima", "Sorento", "Sportage", "Soul", "Telluride"],
    "SUBARU":       ["Outback", "Forester", "Impreza", "Crosstrek", "Legacy"],
    "MAZDA":        ["Mazda3", "Mazda6", "CX-5", "CX-9", "MX-5 Miata"],
    "JEEP":         ["Grand Cherokee", "Wrangler", "Cherokee", "Compass", "Renegade"],
    "RAM":          ["1500", "2500", "3500", "ProMaster"],
    "GMC":          ["Sierra", "Terrain", "Acadia", "Yukon"],
    "DODGE":        ["Charger", "Challenger", "Durango", "Journey"],
    "TESLA":        ["Model 3", "Model S", "Model X", "Model Y"],
    "VOLVO":        ["XC90", "XC60", "S60", "V60"],
    "PORSCHE":      ["Cayenne", "Macan", "Panamera", "911"],
}

# Years to collect data for
YEAR_START = 2015
YEAR_END   = 2025

# Polite delay between API calls (seconds)
# Prevents hammering the NHTSA server
REQUEST_DELAY = 0.5
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH    = os.path.join(ROOT, "data", "automotive_recall.db")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{DB_PATH}"   # f-string so DB_PATH gets substituted
)

# Progress Tracking

@dataclass
class IngestionStats:
    """Tracks counts and timing across the full ingestion run."""
    total_api_calls:       int = 0
    successful_api_calls:  int = 0
    failed_api_calls:      int = 0
    complaints_inserted:   int = 0
    complaints_skipped:    int = 0
    recalls_inserted:      int = 0
    recalls_skipped:       int = 0
    start_time:            float = field(default_factory=time.time)

    @property
    def elapsed_minutes(self) -> float:
        return (time.time() - self.start_time) / 60

    @property
    def total_complaints(self) -> int:
        return self.complaints_inserted + self.complaints_skipped

    @property
    def total_recalls(self) -> int:
        return self.recalls_inserted + self.recalls_skipped

    def summary(self) -> str:
        return (
            f"Ingestion complete in {self.elapsed_minutes:.1f} min\n"
            f"  API calls    : {self.successful_api_calls} ok, "
            f"{self.failed_api_calls} failed / {self.total_api_calls} total\n"
            f"  Complaints   : {self.complaints_inserted} inserted, "
            f"{self.complaints_skipped} skipped\n"
            f"  Recalls      : {self.recalls_inserted} inserted, "
            f"{self.recalls_skipped} skipped"
        )

# Pipeline

class DataIngestionPipeline:
    """
    Orchestrates the full NHTSA data ingestion.

    Usage:
        pipeline = DataIngestionPipeline()
        pipeline.ingest_all_data()

        # Or ingest a single vehicle (useful for testing)
        pipeline.ingest_vehicle("TOYOTA", "Camry", 2020)
    """

    def __init__(
        self,
        db_path:       str = DATABASE_URL,
        request_delay: float = REQUEST_DELAY,
    ):
        self.db      = DatabaseManager(db_url=DATABASE_URL)
        self.client  = NHTSAClient()
        self.delay   = request_delay
        self.stats   = IngestionStats()

        # Ensure DB tables exist before trying to write anything
        self.db.init_db()
        logger.info("DataIngestionPipeline ready")

    # Main Entry Point

    def ingest_all_data(
        self,
        catalogue:  dict = None,
        year_start: int  = YEAR_START,
        year_end:   int  = YEAR_END,
    ) -> IngestionStats:
        """
        Ingest complaints and recalls for every vehicle in the catalogue.

        Loops through every manufacturer → model → year combination,
        fetches from NHTSA, and stores in the database.
        Skips vehicles already fully ingested (incremental update support).

        Args:
            catalogue:  Dict of {make: [models]}. Defaults to VEHICLE_CATALOGUE.
            year_start: First model year to fetch (inclusive).
            year_end:   Last model year to fetch (inclusive).

        Returns:
            IngestionStats object with full counts and timing.
        """
        catalogue  = catalogue or VEHICLE_CATALOGUE
        self.stats = IngestionStats()

        # Build the full list of (make, model, year) combinations
        tasks = [
            (make, model, year)
            for make, models in catalogue.items()
            for model in models
            for year in range(year_start, year_end + 1)
        ]
        total_tasks = len(tasks)

        logger.info(
            "Starting ingestion — %d manufacturers, %d tasks, years %d–%d",
            len(catalogue), total_tasks, year_start, year_end,
        )
        print(f"\n🚗  Starting data ingestion")
        print(f"    {len(catalogue)} manufacturers × models × "
              f"{year_end - year_start + 1} years = {total_tasks} vehicle/year combos\n")

        # Main loop
        for i, (make, model, year) in enumerate(tasks, start=1):
            prefix = f"[{i:4d}/{total_tasks}]  {make:<15} {model:<20} {year}"

            self.ingest_vehicle(make, model, year, log_prefix=prefix)

            # Polite delay between requests
            if i < total_tasks:
                time.sleep(self.delay)

        # Final summary
        print(f"\n{'=' * 55}")
        print(f"  ✅  {self.stats.summary()}")
        print(f"{'=' * 55}\n")
        logger.info(self.stats.summary())

        return self.stats

    # Single Vehicle 

    def ingest_vehicle(
        self,
        make: str,
        model: str,
        year: int,
        log_prefix: str = "",
    ) -> dict:
        """
        Fetch and store complaints + recalls for one vehicle/year.

        Args:
            make:       Vehicle manufacturer e.g. "TOYOTA"
            model:      Vehicle model e.g. "Camry"
            year:       Model year e.g. 2020
            log_prefix: Optional prefix for progress output

        Returns:
            Dict with counts for this vehicle:
            {"complaints_inserted": N, "recalls_inserted": N, "errors": N}
        """
        result = {"complaints_inserted": 0, "recalls_inserted": 0, "errors": 0}

        # Skip if already fetched and fully ingested (So no need to re-fetch complaints because of API)
        # If yes, skip the API call entirely — no point re-fetching

        existing = self.db.get_complaints(make, model, year, limit=1)
        if existing:
            self.stats.complaints_skipped += 1
            self.stats.recalls_skipped    += 1
            print(f"{log_prefix}  →  SKIP (already in DB)")
            return result
        
        # Fetch & store complaints
        complaints_result = self.client.fetch_complaints(make, model, year)
        self.stats.total_api_calls += 1

        if complaints_result.success:
            self.stats.successful_api_calls += 1
            if complaints_result.data:
                db_result = self.db.insert_complaints(complaints_result.data)
                self.stats.complaints_inserted += db_result["inserted"]
                self.stats.complaints_skipped  += db_result["skipped"]
                result["complaints_inserted"]   = db_result["inserted"]
        else:
            self.stats.failed_api_calls += 1
            result["errors"] += 1
            logger.warning(
                "Failed complaints fetch — %s %s %d: %s",
                make, model, year, complaints_result.error,
            )

        # Small gap between the two calls for the same vehicle
        time.sleep(0.2)

        # Fetch & store recalls 
        recalls_result = self.client.fetch_recalls(make, model, year)
        self.stats.total_api_calls += 1

        if recalls_result.success:
            self.stats.successful_api_calls += 1
            if recalls_result.data:
                db_result = self.db.insert_recalls(recalls_result.data)
                self.stats.recalls_inserted += db_result["inserted"]
                self.stats.recalls_skipped  += db_result["skipped"]
                result["recalls_inserted"]   = db_result["inserted"]
        else:
            self.stats.failed_api_calls += 1
            result["errors"] += 1
            logger.warning(
                "Failed recalls fetch — %s %s %d: %s",
                make, model, year, recalls_result.error,
            )

        # Progress line
        c = complaints_result.count if complaints_result.success else "ERR"
        r = recalls_result.count    if recalls_result.success    else "ERR"
        print(f"{log_prefix}  →  {str(c):>4} complaints  {str(r):>3} recalls")

        return result

    # Convenience: single manufacturer

    def ingest_manufacturer(
        self,
        make: str,
        year_start: int = YEAR_START,
        year_end:   int = YEAR_END,
    ) -> IngestionStats:
        """
        Ingest all models for a single manufacturer.
        Useful for targeted updates or testing.

        Example:
            pipeline.ingest_manufacturer("TOYOTA")
        """
        models = VEHICLE_CATALOGUE.get(make.upper())
        if not models:
            raise ValueError(
                f"Unknown manufacturer: {make}. "
                f"Available: {list(VEHICLE_CATALOGUE.keys())}"
            )
        catalogue = {make.upper(): models}
        return self.ingest_all_data(
            catalogue=catalogue,
            year_start=year_start,
            year_end=year_end,
        )