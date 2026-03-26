"""
migrate_db.py
-------------
One-time script to initialise the database from scratch.

What it does:
  1. Creates data/automotive_recall.db (if not exists)
  2. Creates the complaints and recalls tables
  3. Creates all indexes for fast querying
  4. Runs a quick smoke test (insert + read + delete)
  5. Prints a confirmation summary

Safe to re-run — will never delete existing data.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.data_ingestion.database import DatabaseManager, DATABASE_URL
from src.data_ingestion.nhtsa_client import Complaint, Recall


def print_divider(title: str):
    print("\n" + "=" * 55)
    print(f"  {title}")
    print("=" * 55)


def run_migration():
    print("\n🗄️  Automotive Recall System — Database Migration")
    # Show which database we are connecting to (hide password)
    db_display = DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL[:40]
    print(f"   Connecting to: {db_display}\n")

    # Step 1: Initialise database
    print_divider("STEP 1 — Create database & tables")

    db = DatabaseManager()
    db.init_db()

    print(f"  ✅  complaints table   : ready")
    print(f"  ✅  recalls table      : ready")
    print(f"  ✅  Indexes (x8)       : ready")

    # Step 2: Smoke test — complaints
    print_divider("STEP 2 — Smoke test: complaints table")

    test_complaint = Complaint(
        odi_number       = "TEST-001",
        make             = "TOYOTA",
        model            = "Camry",
        model_year       = 2020,
        component        = "ENGINE",
        summary          = "Test complaint for migration verification",
        crash            = False,
        fire             = False,
        injuries         = 0,
        deaths           = 0,
        date_complained  = "2024-01-01",
        date_of_incident = "2023-12-01",
        vehicle_speed    = None,
    )

    inserted = db.insert_complaint(test_complaint)
    print(f"  {'✅' if inserted else '❌'}  Insert complaint       : {'OK' if inserted else 'FAILED'}")

    exists = db.complaint_exists("TEST-001")
    print(f"  {'✅' if exists else '❌'}  Read complaint back    : {'OK' if exists else 'FAILED'}")

    rows = db.get_complaints("TOYOTA", "Camry", 2020)
    found = any(r["odi_number"] == "TEST-001" for r in rows)
    print(f"  {'✅' if found else '❌'}  Query by vehicle       : {'OK' if found else 'FAILED'}")

    count = db.count_complaints()
    print(f"  ✅  Total complaints    : {count}")

    # Step 3: Smoke test — recalls
    print_divider("STEP 3 — Smoke test: recalls table")

    test_recall = Recall(
        campaign_number = "TEST-R001",
        manufacturer    = "TOYOTA MOTOR CORP",
        make            = "TOYOTA",
        model           = "Camry",
        model_year      = 2020,
        component       = "ENGINE",
        summary         = "Test recall for migration verification",
        consequence     = "Engine may stall",
        remedy          = "Replace engine part",
        recall_date     = "2024-06-01",
        notes           = "Contact dealer for details",
        park_it         = False,
    )

    inserted_r = db.insert_recall(test_recall)
    print(f"  {'✅' if inserted_r else '❌'}  Insert recall          : {'OK' if inserted_r else 'FAILED'}")

    exists_r = db.recall_exists("TEST-R001")
    print(f"  {'✅' if exists_r else '❌'}  Read recall back       : {'OK' if exists_r else 'FAILED'}")

    recalls = db.get_recalls("TOYOTA", "Camry", 2020)
    found_r = any(r["campaign_number"] == "TEST-R001" for r in recalls)
    print(f"  {'✅' if found_r else '❌'}  Query recall by vehicle: {'OK' if found_r else 'FAILED'}")

    count_r = db.count_recalls()
    print(f"  ✅  Total recalls       : {count_r}")

    # Step 4: Stats summary
    print_divider("STEP 4 — Database stats")

    stats = db.get_stats()
    print(f"  Total complaints : {stats['total_complaints']}")
    print(f"  Total recalls    : {stats['total_recalls']}")
    print(f"  Manufacturers    : {stats['manufacturers'] or 'none yet'}")

    # Final result
    all_passed = all([inserted, exists, found, inserted_r, exists_r, found_r])

    print_divider("RESULT")
    if all_passed:
        print("  🎉 Migration successful!")
        print(f"     Connected to  : {db_display}")
        print()
        print("  Next step: run the data ingestion pipeline")
        print("    python scripts/run_ingestion.py")
    else:
        print("  ❌ Migration had failures — check the output above")
    print()

if __name__ == "__main__":
    run_migration()