"""
run_preprocessing.py
--------------------
Run the text preprocessing pipeline on all complaints.

    python scripts/run_preprocessing.py

What it does:
  1. Tests the pipeline on 5 random samples (visual check)
  2. Processes all 179k complaints
  3. Saves clean text + was_recalled label to complaints_processed table
  4. Prints quality stats
"""
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.text_processing import TextPreprocessor

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

if __name__ == "__main__":
    processor = TextPreprocessor()

    # Step 1 — visual sanity check before processing everything
    processor.test_on_samples(n=5)

    confirm = input("Does the preprocessing look correct? (y/n): ").strip().lower()
    if confirm != "y":
        print("Aborted. Adjust text_processing.py and re-run.")
        sys.exit(0)

    # Step 2 — process all complaints
    stats = processor.process_all_complaints()


'''Note: 

Recall rate = 11.3% — this is actually the most important number.

It means:   11.3% of complaints come from vehicles that eventually got recalled
            88.7% come from vehicles that were never recalled

'''