"""
verify_data.py
--------------
Data quality checks after ingestion.
Run this after run_ingestion.py finishes.

    python scripts/verify_data.py

Checks:
  1. Row counts (did we get enough data?)
  2. Null checks (are critical fields populated?)
  3. Manufacturer coverage (did all makes get ingested?)
  4. Year coverage (do we have data across all years?)
  5. Duplicate check (any repeated ODI numbers?)
  6. Sample data preview (spot-check a real complaint)
"""

import sys
import os
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_ingestion.database import DatabaseManager


def print_divider(title: str):
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")


def check_row_counts(db: DatabaseManager) -> bool:
    """Check we have a meaningful amount of data."""
    print_divider("CHECK 1 — Row counts")

    complaints = db.count_complaints()
    recalls    = db.count_recalls()

    # Thresholds — adjust based on your run scope
    MIN_COMPLAINTS = 100    # At least 100 for test run, 10000 for full
    MIN_RECALLS    = 10

    c_ok = complaints >= MIN_COMPLAINTS
    r_ok = recalls    >= MIN_RECALLS

    print(f"  Complaints : {complaints:>8,}   {'✅' if c_ok else '⚠️  low'}")
    print(f"  Recalls    : {recalls:>8,}   {'✅' if r_ok else '⚠️  low'}")

    if not c_ok:
        print(f"\n  ⚠️  Expected at least {MIN_COMPLAINTS} complaints.")
        print(f"     Did the ingestion run complete? Try running:")
        print(f"     python scripts/run_ingestion.py")

    return c_ok and r_ok


def check_null_fields(db: DatabaseManager) -> bool:
    """
    Check that critical fields are populated.
    The ML model needs summary — if most are null, training will fail.
    """
    print_divider("CHECK 2 — Null field rates")

    with db._get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM complaints").fetchone()[0]
        if total == 0:
            print("  ⚠️  No complaints in database — skipping null check")
            return False

        null_summary   = conn.execute(
            "SELECT COUNT(*) FROM complaints WHERE summary IS NULL OR summary = ''"
        ).fetchone()[0]

        null_component = conn.execute(
            "SELECT COUNT(*) FROM complaints WHERE component IS NULL OR component = ''"
        ).fetchone()[0]

        null_make = conn.execute(
            "SELECT COUNT(*) FROM complaints WHERE make IS NULL OR make = ''"
        ).fetchone()[0]

    summary_null_pct   = (null_summary   / total) * 100
    component_null_pct = (null_component / total) * 100
    make_null_pct      = (null_make      / total) * 100

    # Acceptable null rates
    summary_ok   = summary_null_pct   < 5.0   # summary is critical for ML
    component_ok = component_null_pct < 20.0  # component can sometimes be missing
    make_ok      = make_null_pct      < 1.0   # make should always be present

    print(f"  summary   null : {summary_null_pct:5.1f}%   {'✅' if summary_ok   else '❌  too high — ML needs this!'}")
    print(f"  component null : {component_null_pct:5.1f}%   {'✅' if component_ok else '⚠️  high'}")
    print(f"  make      null : {make_null_pct:5.1f}%   {'✅' if make_ok      else '❌  critical field missing'}")

    return summary_ok and make_ok


def check_manufacturer_coverage(db: DatabaseManager) -> bool:
    """Check which manufacturers actually made it into the database."""
    print_divider("CHECK 3 — Manufacturer coverage")

    with db._get_connection() as conn:
        rows = conn.execute(
            """
            SELECT make, COUNT(*) as cnt
            FROM complaints
            GROUP BY make
            ORDER BY cnt DESC
            """
        ).fetchall()

    if not rows:
        print("  ⚠️  No data found")
        return False

    print(f"  {'Make':<20} {'Complaints':>12}")
    print(f"  {'-'*20} {'-'*12}")
    for row in rows:
        print(f"  {row[0]:<20} {row[1]:>12,}")

    print(f"\n  Total makes in DB : {len(rows)}")
    return len(rows) > 0


def check_year_coverage(db: DatabaseManager) -> bool:
    """Check complaints are spread across years — not all from one year."""
    print_divider("CHECK 4 — Year coverage")

    with db._get_connection() as conn:
        rows = conn.execute(
            """
            SELECT model_year, COUNT(*) as cnt
            FROM complaints
            GROUP BY model_year
            ORDER BY model_year
            """
        ).fetchall()

    if not rows:
        print("  ⚠️  No data found")
        return False

    print(f"  {'Year':<8} {'Complaints':>12}")
    print(f"  {'-'*8} {'-'*12}")
    for row in rows:
        bar_len = min(int(row[1] / 10), 40)
        bar = "█" * bar_len
        print(f"  {row[0]:<8} {row[1]:>8,}   {bar}")

    return len(rows) >= 3   # At least 3 different years


def check_duplicates(db: DatabaseManager) -> bool:
    """
    Check for duplicate ODI numbers.
    There should be zero — our INSERT OR IGNORE prevents them.
    """
    print_divider("CHECK 5 — Duplicate check")

    with db._get_connection() as conn:
        dupes = conn.execute(
            """
            SELECT odi_number, COUNT(*) as cnt
            FROM complaints
            GROUP BY odi_number
            HAVING cnt > 1
            """
        ).fetchall()

    dupe_count = len(dupes)
    ok = dupe_count == 0

    if ok:
        print(f"  ✅  No duplicate ODI numbers found")
    else:
        print(f"  ❌  Found {dupe_count} duplicate ODI numbers!")
        for d in dupes[:5]:
            print(f"      ODI {d[0]} appears {d[1]} times")

    return ok


def check_sample_data(db: DatabaseManager):
    """Print a sample complaint so you can visually verify the data looks right."""
    print_divider("CHECK 6 — Sample data preview")

    with db._get_connection() as conn:
        row = conn.execute(
            """
            SELECT make, model, model_year, component, summary, crash, injuries
            FROM complaints
            WHERE summary IS NOT NULL AND summary != ''
            ORDER BY RANDOM()
            LIMIT 1
            """
        ).fetchone()

    if not row:
        print("  ⚠️  No complaints with summaries found")
        return

    summary = row[4]
    preview = summary[:200] + "..." if len(summary) > 200 else summary

    print(f"  Vehicle    : {row[0]} {row[1]} {row[2]}")
    print(f"  Component  : {row[3]}")
    print(f"  Crash?     : {'Yes ⚠️' if row[5] else 'No'}")
    print(f"  Injuries   : {row[6]}")
    print(f"  Summary    : {preview}")

# Main

if __name__ == "__main__":
    print("\n Automotive Recall System — Data Quality Verification")

    db = DatabaseManager()

    # Run all checks
    results = {
        "Row counts":            check_row_counts(db),
        "Null fields":           check_null_fields(db),
        "Manufacturer coverage": check_manufacturer_coverage(db),
        "Year coverage":         check_year_coverage(db),
        "Duplicates":            check_duplicates(db),
    }

    # Sample data (no pass/fail — just a visual spot check)
    check_sample_data(db)

    # Final summary
    print_divider("SUMMARY")
    all_passed = True
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  —  {check_name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("Data quality checks passed!")
    else:
        print("  ⚠️  Some checks failed.")
        print("     Review the output above and re-run ingestion if needed.")
    print()