"""
text_processing.py
------------------

Pipeline:
    raw text  →  clean_text()  →  tokenize()  →  remove_stopwords()
                                                        ↓
                                              clean string saved to DB

Usage:
    from src.data_processing.text_processing import TextPreprocessor

    processor = TextPreprocessor()

    # Process a single complaint
    clean = processor.process("THE ENGINE STALLED!! VIN:1HGBH41 on 03/15/2023")
    # → "engine stalled"

    # Process entire database (saves to complaints_processed table)
    processor.process_all_complaints()
"""

import logging
import re
import sqlite3
import string
from datetime import datetime

import nltk

logger = logging.getLogger(__name__)

# NLTK Downloads

def _ensure_nltk_data():
    """Download required NLTK data files if not already present."""
    for resource, path in [
        ("tokenizers/punkt",     "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords",    "stopwords"),
    ]:
        try:
            nltk.data.find(resource)
        except LookupError:
            logger.info("Downloading NLTK resource: %s", path)
            nltk.download(path, quiet=True)

_ensure_nltk_data()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Constants

DB_PATH = "data/automotive_recall.db"

_BASE_STOPWORDS = set(stopwords.words("english"))

# Automotive-specific noise words — appear constantly but
# carry no signal about what the actual defect is
_AUTOMOTIVE_NOISE = {
    "vehicle", "car", "said", "also", "would", "could",
    "told", "called", "went", "got", "came",
    "dealer", "dealership", "contact", "contacted", "service",
    "stated", "informed",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "mph", "km", "mile", "miles", "approximately", "around",
}

STOPWORDS = _BASE_STOPWORDS | _AUTOMOTIVE_NOISE

# Minimum words after cleaning for a summary to be kept
MIN_WORDS_AFTER_CLEANING = 5

CREATE_PROCESSED_TABLE = """
CREATE TABLE IF NOT EXISTS complaints_processed (
    odi_number      TEXT PRIMARY KEY,
    make            TEXT NOT NULL,
    model           TEXT NOT NULL,
    model_year      INTEGER NOT NULL,
    component       TEXT,
    summary_clean   TEXT,
    crash           INTEGER DEFAULT 0,
    fire            INTEGER DEFAULT 0,
    injuries        INTEGER DEFAULT 0,
    deaths          INTEGER DEFAULT 0,
    was_recalled    INTEGER DEFAULT 0,
    processed_at    TEXT NOT NULL
);
"""

# TextPreprocessor

class TextPreprocessor:
    """
    Cleans and tokenizes NHTSA complaint summaries.

    Three-step pipeline:
        1. clean_text()        — standardise, remove noise
        2. tokenize()          — split into word list
        3. remove_stopwords()  — drop meaningless words

    Usage:
        p = TextPreprocessor()
        clean = p.process("THE ENGINE STALLED!! Called dealer 03/15/2023")
        # → "engine stalled"
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        logger.info("TextPreprocessor ready")

    # Core Three Functions

    def clean_text(self, text: str) -> str:
        """
        Step 1 — Standardise raw complaint text.

        Operations in order:
          1. Lowercase everything
          2. Remove URLs
          3. Remove VIN numbers (17 alphanumeric chars)
          4. Remove dates (03/15/2023 or 2023-03-15)
          5. Remove punctuation
          6. Remove standalone digits
          7. Collapse whitespace

        Args:
            text: Raw complaint summary from NHTSA.

        Returns:
            Cleaned lowercase string. Empty string if input is None.

        Example:
            clean_text("THE ENGINE STALLED!!! VIN:1HGBH41JXMN109186")
            → "the engine stalled vin"
        """
        if not text or not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        text = re.sub(r"\b[A-HJ-NPR-Z0-9]{17}\b", " ", text)
        text = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", " ", text)
        text = re.sub(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", " ", text)
        text = text.translate(
            str.maketrans(string.punctuation, " " * len(string.punctuation))
        )
        text = re.sub(r"\b\d+\b", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize(self, text: str) -> list:
        """
        Step 2 — Split cleaned text into individual word tokens.

        Uses NLTK's word_tokenize which correctly handles
        hyphenated words and edge cases simple split() misses.

        Args:
            text: Cleaned string from clean_text().

        Returns:
            List of lowercase alphabetic tokens.

        Example:
            tokenize("engine stalled highway")
            → ["engine", "stalled", "highway"]
        """
        if not text:
            return []

        tokens = word_tokenize(text)
        # Keep only alphabetic tokens — removes leftover punctuation
        return [t for t in tokens if t.isalpha()]

    def remove_stopwords(self, tokens: list) -> list:
        """
        Step 3 — Remove words that carry no predictive signal.

        Removes:
          - Standard English stopwords (the, a, was, I, etc.)
          - Automotive noise words (dealer, contact, etc.)
          - Tokens shorter than 3 characters

        Args:
            tokens: List of tokens from tokenize().

        Returns:
            Filtered list with stopwords and short tokens removed.

        Example:
            remove_stopwords(["the", "engine", "stalled", "on"])
            → ["engine", "stalled"]
        """
        return [
            t for t in tokens
            if t not in STOPWORDS and len(t) >= 3
        ]

    # Pipeline
    def process(self, text: str) -> str:
        """
        Run the full three-step pipeline on one complaint summary.

        clean_text → tokenize → remove_stopwords → join back to string

        Args:
            text: Raw complaint summary.

        Returns:
            Clean string of meaningful tokens joined by spaces.

        Example:
            process("THE ENGINE STALLED ON HIGHWAY!!! Called dealer.")
            → "engine stalled highway"
        """
        cleaned  = self.clean_text(text)
        tokens   = self.tokenize(cleaned)
        filtered = self.remove_stopwords(tokens)
        return " ".join(filtered)

    def is_usable(self, clean_text: str) -> bool:
        """
        Return True if clean text has enough words to be useful for ML.
        Summaries below MIN_WORDS_AFTER_CLEANING are dropped.
        """
        return len(clean_text.split()) >= MIN_WORDS_AFTER_CLEANING

    # Batch Processing

    def process_all_complaints(self) -> dict:
        """
        Process all complaints and save to complaints_processed table.

        Also adds the was_recalled label — 1 if this vehicle/year
        combination appears in the recalls table, 0 otherwise.
        This label is what the ML model learns to predict.

        Returns:
            Stats dict: {total, kept, dropped, recall_rate, elapsed}
        """
        start = datetime.utcnow()
        logger.info("Starting batch text preprocessing...")
        print("\n Text Preprocessing Pipeline")
        print(f"    Database: {self.db_path}\n")

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        # Load all raw complaints
        complaints = conn.execute("SELECT * FROM complaints").fetchall()
        total = len(complaints)
        print(f"  Loaded {total:,} complaints from database")

        # Build recalled vehicle lookup
        # Key: (make, model, model_year) → was this ever recalled?
        recalls = conn.execute(
            "SELECT DISTINCT make, model, model_year FROM recalls"
        ).fetchall()
        recalled_vehicles = {
            (r["make"].upper(), r["model"], int(r["model_year"]))
            for r in recalls
        }
        print(f"  Found {len(recalled_vehicles):,} recalled vehicle combos")

        # Create processed table
        conn.execute(CREATE_PROCESSED_TABLE)

        kept           = 0
        dropped        = 0
        rows_to_insert = []
        processed_at   = datetime.utcnow().isoformat()

        for i, row in enumerate(complaints, 1):
            if i % 10000 == 0:
                print(f"  Processing... {i:,}/{total:,} ({i/total*100:.0f}%)")

            clean = self.process(row["summary"])

            # Drop summaries too short to be useful for NLP
            if not self.is_usable(clean):
                dropped += 1
                continue

            # Add recall label — this is the ML target variable
            vehicle_key  = (
                (row["make"] or "").upper(),
                row["model"] or "",
                int(row["model_year"] or 0),
            )
            was_recalled = 1 if vehicle_key in recalled_vehicles else 0

            rows_to_insert.append({
                "odi_number":    row["odi_number"],
                "make":          (row["make"] or "").upper(),
                "model":         row["model"] or "",
                "model_year":    int(row["model_year"] or 0),
                "component":     row["component"] or "UNKNOWN",
                "summary_clean": clean,
                "crash":         int(row["crash"] or 0),
                "fire":          int(row["fire"] or 0),
                "injuries":      int(row["injuries"] or 0),
                "deaths":        int(row["deaths"] or 0),
                "was_recalled":  was_recalled,
                "processed_at":  processed_at,
            })
            kept += 1

        # Batch insert — one transaction for all rows (fast)
        print(f"  Saving {kept:,} rows to complaints_processed...")
        conn.executemany(
            """
            INSERT OR REPLACE INTO complaints_processed (
                odi_number, make, model, model_year, component,
                summary_clean, crash, fire, injuries, deaths,
                was_recalled, processed_at
            ) VALUES (
                :odi_number, :make, :model, :model_year, :component,
                :summary_clean, :crash, :fire, :injuries, :deaths,
                :was_recalled, :processed_at
            )
            """,
            rows_to_insert,
        )
        conn.commit()
        conn.close()

        elapsed      = (datetime.utcnow() - start).total_seconds()
        recall_count = sum(1 for r in rows_to_insert if r["was_recalled"] == 1)
        recall_rate  = recall_count / kept * 100 if kept else 0

        stats = {
            "total":       total,
            "kept":        kept,
            "dropped":     dropped,
            "elapsed":     elapsed,
            "recall_rate": recall_rate,
        }

        print(f"\n  ✅  Done in {elapsed:.1f}s")
        print(f"  Total input        : {total:,}")
        print(f"  Kept (usable)      : {kept:,}  ({kept/total*100:.1f}%)")
        print(f"  Dropped (too short): {dropped:,}  ({dropped/total*100:.1f}%)")
        print(f"  Recalled           : {recall_count:,}  ({recall_rate:.1f}%)")
        print(f"\n  💡 Recall rate = {recall_rate:.1f}%  ← your ML baseline")
        print(f"     Table: complaints_processed\n")

        logger.info(
            "Preprocessing done — kept=%d, dropped=%d, recall_rate=%.1f%%",
            kept, dropped, recall_rate,
        )
        return stats

    # Quick Visual Test

    def test_on_samples(self, n: int = 5):
        """
        Show before/after for n random complaints.
        Run this first to visually verify the pipeline is working.
        """
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            f"""
            SELECT make, model, model_year, summary
            FROM complaints
            WHERE summary IS NOT NULL AND length(summary) > 50
            ORDER BY RANDOM()
            LIMIT {n}
            """
        ).fetchall()
        conn.close()

        print(f"\n🔍  Sample preprocessing — {n} random complaints\n")
        print("=" * 60)

        for i, row in enumerate(rows, 1):
            raw   = row[3]
            clean = self.process(raw)
            print(f"[{i}]  {row[0]} {row[1]} {row[2]}")
            print(f"  BEFORE ({len(raw.split()):>3} words) : {raw[:130]}...")
            print(f"  AFTER  ({len(clean.split()):>3} words) : {clean[:130]}")
            print()