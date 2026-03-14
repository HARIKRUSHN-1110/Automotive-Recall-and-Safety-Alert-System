"""
test_api_manual.py

A simple script to manually verify the NHTSA API client is working.

Run this from your project root folder:
    python scripts/test_api_manual.py

What it does:
  1. Connects to the real NHTSA API
  2. Fetches complaints for a known vehicle (Toyota Camry Series 2020)
  3. Fetches recalls for the same vehicle
  4. Prints the results in a readable format

This is NOT an automated test — just a quick sanity check.
"""

import sys
import os

# Make sure Python can find src/ folder
# This adds the project root to the path so can import from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_ingestion.nhtsa_client import NHTSAClient


# Test vehicle — Toyota Camry Series 2020 is known to have complaints

TEST_MAKE  = "TOYOTA"
TEST_MODEL = "Camry"
TEST_YEAR  = 2020


def print_divider(title: str):
    """Print a simple section header."""
    print("\n" + "=" * 55)
    print(f"  {title}")
    print("=" * 55)


def test_complaints():
    """Fetch and display complaints for the test vehicle."""
    print_divider(f"COMPLAINTS — {TEST_MAKE} {TEST_MODEL} {TEST_YEAR}")

    client = NHTSAClient()
    result = client.fetch_complaints(
        make=TEST_MAKE,
        model=TEST_MODEL,
        year=TEST_YEAR,
    )

    # Check if request succeeded 
    if not result.success:
        print(f"❌ FAILED: {result.error}")
        return False

    print(f"✅ Success! Found {result.count} complaints ({result.elapsed_seconds:.2f}s)")

    if result.count == 0:
        print("⚠️  No complaints found — try a different vehicle")
        return True

    # Show first 3 complaints 
    print(f"\nShowing first 3 of {result.count} complaints:\n")

    for i, complaint in enumerate(result.data[:3], start=1):
        print(f"  [{i}] ODI Number : {complaint.odi_number}")
        print(f"      Component  : {complaint.component}")
        print(f"      Crash?     : {'Yes ⚠️' if complaint.crash else 'No'}")
        print(f"      Injuries   : {complaint.injuries}")
        print(f"      Date Filed : {complaint.date_complained}")
        # Trim long summaries so the terminal doesn't flood
        summary = complaint.summary[:120] + "..." if len(complaint.summary) > 120 else complaint.summary
        print(f"      Summary    : {summary}")
        print()

    return True


def test_recalls():
    """Fetch and display recalls for the test vehicle."""
    print_divider(f"RECALLS — {TEST_MAKE} {TEST_MODEL} {TEST_YEAR}")

    client = NHTSAClient()
    result = client.fetch_recalls(
        make=TEST_MAKE,
        model=TEST_MODEL,
        year=TEST_YEAR,
    )

    # Check if request succeeded 
    if not result.success:
        print(f"❌ FAILED: {result.error}")
        return False

    print(f"✅ Success! Found {result.count} recalls ({result.elapsed_seconds:.2f}s)")

    if result.count == 0:
        print("ℹ️  No recalls found for this vehicle")
        return True

    # Show all recalls 
    print(f"\nShowing all {result.count} recalls:\n")

    for i, recall in enumerate(result.data, start=1):
        print(f"  [{i}] Campaign # : {recall.campaign_number}")
        print(f"      Component  : {recall.component}")
        print(f"      Date       : {recall.recall_date}")
        summary = recall.summary[:120] + "..." if len(recall.summary) > 120 else recall.summary
        print(f"      Summary    : {summary}")
        print()

    return True


def test_empty_vehicle():
    """
    Test with a vehicle that likely has no complaints.
    Confirms our code handles zero results gracefully (doesn't crash).
    """
    print_divider("EDGE CASE — Vehicle with no complaints")

    client = NHTSAClient()
    result = client.fetch_complaints(
        make="FERRARI",
        model="Roma",
        year=2022,
    )

    if result.success:
        print(f"✅ Handled gracefully — {result.count} complaints found (expected 0 or very few)")
    else:
        print(f"⚠️  Request failed: {result.error}")

    return True


def test_health_check():
    """Run the built-in health check to confirm API connectivity."""
    print_divider("HEALTH CHECK")

    client = NHTSAClient()
    is_healthy = client.health_check()

    if is_healthy:
        print("✅ NHTSA API is reachable and responding")
    else:
        print("❌ NHTSA API is NOT reachable — check your internet connection")

    return is_healthy


# Main

if __name__ == "__main__":
    print("\n🚗  NHTSA API Manual Test")
    print("   Testing connection to the real NHTSA API...")

    results = {
        "Health Check"        : test_health_check(),
        "Complaints Fetch"    : test_complaints(),
        "Recalls Fetch"       : test_recalls(),
        "Empty Vehicle Check" : test_empty_vehicle(),
    }

    # Final summary 
    print_divider("SUMMARY")
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  —  {test_name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("🎉 All tests passed! nhtsa_client.py is working correctly.")
    else:
        print("⚠️  Some tests failed. Check your internet connection")
        print("   or review the error messages above.")
    print()