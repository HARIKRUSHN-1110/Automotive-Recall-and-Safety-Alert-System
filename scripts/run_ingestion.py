"""
run_ingestion.py
----------------
Trigger the full NHTSA data ingestion pipeline.

Run from your project root:
    python scripts/run_ingestion.py

Options (edit at the top of this file):
    FULL_RUN   = True   → all 20 manufacturers, 2015-2025  (~30 min)
    FULL_RUN   = False  → 3 manufacturers only             (~3 min, for testing)
"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_ingestion.data_ingestion import DataIngestionPipeline, VEHICLE_CATALOGUE

# CONFIGURE HERE

# Set to True for the full run (all 20 manufacturers, all years)
# Set to False for a quick test run (3 manufacturers only)
FULL_RUN = True

# Year range — adjust if you want more or fewer years
YEAR_START = 2015
YEAR_END   = 2025

# Set up basic logging so you see INFO messages in the terminal
logging.basicConfig(
    level=logging.WARNING,     # Only show warnings+ from libraries
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
# But show INFO from our own code
logging.getLogger("src").setLevel(logging.INFO)

def main():
    if FULL_RUN:
        catalogue = VEHICLE_CATALOGUE
        print("🚗  Mode: FULL RUN — all 20 manufacturers")
        print(f"    Expected time: ~30 minutes\n")
    else:
        # Smaller catalogue for a quick test
        catalogue = {
            "TOYOTA":    ["Camry", "RAV4"],
            "BMW":       ["3 Series", "X5"],
            "FORD":      ["F-150", "Explorer"],
        }
        print("🚗  Mode: TEST RUN — 3 manufacturers only")
        print("    Switch FULL_RUN = True for the complete dataset\n")

    pipeline = DataIngestionPipeline()
    stats    = pipeline.ingest_all_data(
        catalogue=catalogue,
        year_start=YEAR_START,
        year_end=YEAR_END,
    )

    print("\n📊  Final database counts:")
    db_stats = pipeline.db.get_stats()
    print(f"    Complaints : {db_stats['total_complaints']:,}")
    print(f"    Recalls    : {db_stats['total_recalls']:,}")
    print(f"    Makes      : {', '.join(db_stats['manufacturers']) or 'none'}")

    if not FULL_RUN:
        print("\n💡  IMPORTANT: to ingest the full and latest dataset:")
        print("    Set FULL_RUN = True in scripts/run_ingestion.py")
        print("    Then re-run: python scripts/run_ingestion.py")


if __name__ == "__main__":
    main()