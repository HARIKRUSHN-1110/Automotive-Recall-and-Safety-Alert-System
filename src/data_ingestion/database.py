"""
database.py
-----------
SQLite database management

Responsibilities:
  - Manage database connections (open, close, reuse)
  - Create tables (complaints, recalls) on first run
  - Create indexes for fast queries
  - CRUD operations: insert, read, count, check duplicates

Every other module in this project talks to the database
through this file only — never raw SQL scattered elsewhere.

Usage:
    from src.data_ingestion.database import DatabaseManager

    db = DatabaseManager()
    db.init_db()                         # create tables if not exist

    db.insert_complaint(complaint)       # save one complaint
    db.insert_complaints(complaints)     # save many at once (fast)

    rows = db.get_complaints("BMW", "3 Series", 2020)
    count = db.count_complaints()
"""

import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

from src.data_ingestion.nhtsa_client import Complaint, Recall

logger = logging.getLogger(__name__)

# Default database file path (relative to project root)

DEFAULT_DB_PATH = "data/automotive_recall.db"

# SQL — Table Definitions

CREATE_COMPLAINTS_TABLE = """
CREATE TABLE IF NOT EXISTS complaints (
    odi_number          TEXT PRIMARY KEY,
    make                TEXT NOT NULL,
    model               TEXT NOT NULL,
    model_year          INTEGER NOT NULL,
    component           TEXT,
    summary             TEXT,
    crash               INTEGER DEFAULT 0,
    fire                INTEGER DEFAULT 0,
    injuries            INTEGER DEFAULT 0,
    deaths              INTEGER DEFAULT 0,
    date_complained     TEXT,
    date_of_incident    TEXT,
    vehicle_speed       INTEGER,
    created_at          TEXT NOT NULL
);
"""

CREATE_RECALLS_TABLE = """
CREATE TABLE IF NOT EXISTS recalls (
    campaign_number     TEXT PRIMARY KEY,
    manufacturer        TEXT,
    make                TEXT NOT NULL,
    model               TEXT NOT NULL,
    model_year          INTEGER NOT NULL,
    component           TEXT,
    summary             TEXT,
    consequence         TEXT,
    remedy              TEXT,
    notes               TEXT,
    recall_date         TEXT,
    park_it             INTEGER DEFAULT 0,
    created_at          TEXT NOT NULL
);
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_complaints_make    ON complaints (make);",
    "CREATE INDEX IF NOT EXISTS idx_complaints_model   ON complaints (model);",
    "CREATE INDEX IF NOT EXISTS idx_complaints_year    ON complaints (model_year);",
    "CREATE INDEX IF NOT EXISTS idx_complaints_vehicle ON complaints (make, model, model_year);",
    "CREATE INDEX IF NOT EXISTS idx_recalls_make       ON recalls (make);",
    "CREATE INDEX IF NOT EXISTS idx_recalls_model      ON recalls (model);",
    "CREATE INDEX IF NOT EXISTS idx_recalls_year       ON recalls (model_year);",
    "CREATE INDEX IF NOT EXISTS idx_recalls_vehicle    ON recalls (make, model, model_year);",
]


# DatabaseManager

class DatabaseManager:
    """
    Manages all SQLite database operations for the recall system.

    Usage:
        db = DatabaseManager()
        db.init_db()
        db.insert_complaint(complaint)

        # Or as a context manager
        with DatabaseManager() as db:
            db.insert_complaints(complaints)
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._ensure_directory()
        logger.info("DatabaseManager ready — db_path=%s", self.db_path)

    #  Setup 

    def _ensure_directory(self):
        """Create the data/ directory if it doesn't exist yet."""
        directory = os.path.dirname(self.db_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info("Created directory: %s", directory)

    def init_db(self):
        """
        Create tables and indexes if they don't already exist.
        Safe to call multiple times — never overwrites existing data.
        Call this once at application startup.
        """
        logger.info("Initialising database at %s", self.db_path)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(CREATE_COMPLAINTS_TABLE)
            cursor.execute(CREATE_RECALLS_TABLE)
            for index_sql in CREATE_INDEXES:
                cursor.execute(index_sql)
            conn.commit()
        logger.info("Database initialised successfully")

    # Connection Management 

    @contextmanager
    def _get_connection(self):
        """
        Open a database connection, yield it, then always close it.
        Rolls back automatically if an error occurs.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # INSERT — Complaints

    def insert_complaint(self, complaint: Complaint) -> bool:
        """
        Insert a single complaint. Skips silently if it already exists.

        Returns:
            True if inserted, False if skipped (duplicate).

        Note: INSERT OR IGNORE — this is the duplicate protection. If you run the ingestion twice (or restart it halfway through),
        it just skips already-stored rows silently instead of crashing with a "UNIQUE constraint failed" error.

        """
        sql = """
            INSERT OR IGNORE INTO complaints (
                odi_number, make, model, model_year,
                component, summary,
                crash, fire, injuries, deaths,
                date_complained, date_of_incident, vehicle_speed,
                created_at
            ) VALUES (
                :odi_number, :make, :model, :model_year,
                :component, :summary,
                :crash, :fire, :injuries, :deaths,
                :date_complained, :date_of_incident, :vehicle_speed,
                :created_at
            )
        """
        params = {
            "odi_number":       complaint.odi_number,
            "make":             complaint.make.upper(),
            "model":            complaint.model,
            "model_year":       complaint.model_year,
            "component":        complaint.component,
            "summary":          complaint.summary,
            "crash":            int(complaint.crash),
            "fire":             int(complaint.fire),
            "injuries":         complaint.injuries,
            "deaths":           complaint.deaths,
            "date_complained":  complaint.date_complained,
            "date_of_incident": complaint.date_of_incident,
            "vehicle_speed":    complaint.vehicle_speed,
            "created_at":       _now(),
        }
        with self._get_connection() as conn:
            cursor = conn.execute(sql, params)
            conn.commit()
            inserted = cursor.rowcount > 0
        if inserted:
            logger.debug("Inserted complaint %s", complaint.odi_number)
        else:
            logger.debug("Skipped duplicate complaint %s", complaint.odi_number)
        return inserted

    def insert_complaints(self, complaints: list) -> dict:
        """
        Insert a list of complaints in a single fast transaction.

        Returns:
            {"inserted": N, "skipped": N, "total": N}
        """
        if not complaints:
            return {"inserted": 0, "skipped": 0, "total": 0}

        sql = """
            INSERT OR IGNORE INTO complaints (
                odi_number, make, model, model_year,
                component, summary,
                crash, fire, injuries, deaths,
                date_complained, date_of_incident, vehicle_speed,
                created_at
            ) VALUES (
                :odi_number, :make, :model, :model_year,
                :component, :summary,
                :crash, :fire, :injuries, :deaths,
                :date_complained, :date_of_incident, :vehicle_speed,
                :created_at
            )
        """
        now = _now()
        params_list = [
            {
                "odi_number":       c.odi_number,
                "make":             c.make.upper(),
                "model":            c.model,
                "model_year":       c.model_year,
                "component":        c.component,
                "summary":          c.summary,
                "crash":            int(c.crash),
                "fire":             int(c.fire),
                "injuries":         c.injuries,
                "deaths":           c.deaths,
                "date_complained":  c.date_complained,
                "date_of_incident": c.date_of_incident,
                "vehicle_speed":    c.vehicle_speed,
                "created_at":       now,
            }
            for c in complaints
        ]
        with self._get_connection() as conn:
            cursor = conn.executemany(sql, params_list)
            conn.commit()
            inserted = cursor.rowcount
        skipped = len(complaints) - inserted
        result = {"inserted": inserted, "skipped": skipped, "total": len(complaints)}
        logger.info(
            "Batch insert complaints — inserted=%d, skipped=%d, total=%d",
            inserted, skipped, len(complaints),
        )
        return result
    
    """
    Note: insert_complaints() vs insert_complaint() — the batch version uses executemany() which wraps hundreds of inserts in a single transaction.
    One commit for 500 rows is ~100 times faster than 500 individual commits.

    """
    #  INSERT — Recalls 

    def insert_recall(self, recall: Recall) -> bool:
        """
        Insert a single recall. Skips silently if it already exists.

        Returns:
            True if inserted, False if skipped (duplicate).
        """
        sql = """
            INSERT OR IGNORE INTO recalls (
                campaign_number, manufacturer,
                make, model, model_year,
                component, summary, consequence, remedy, notes,
                recall_date, park_it, created_at
            ) VALUES (
                :campaign_number, :manufacturer,
                :make, :model, :model_year,
                :component, :summary, :consequence, :remedy, :notes,
                :recall_date, :park_it, :created_at
            )
        """
        params = {
            "campaign_number": recall.campaign_number,
            "manufacturer":    recall.manufacturer,
            "make":            recall.make.upper(),
            "model":           recall.model,
            "model_year":      recall.model_year,
            "component":       recall.component,
            "summary":         recall.summary,
            "consequence":     recall.consequence,
            "remedy":          recall.remedy,
            "notes":           recall.notes,
            "recall_date":     recall.recall_date,
            "park_it":         int(recall.park_it),
            "created_at":      _now(),
        }
        with self._get_connection() as conn:
            cursor = conn.execute(sql, params)
            conn.commit()
            inserted = cursor.rowcount > 0
        if inserted:
            logger.debug("Inserted recall %s", recall.campaign_number)
        else:
            logger.debug("Skipped duplicate recall %s", recall.campaign_number)
        return inserted

    def insert_recalls(self, recalls: list) -> dict:
        """
        Insert a list of recalls in a single fast transaction.

        Returns:
            {"inserted": N, "skipped": N, "total": N}
        """
        if not recalls:
            return {"inserted": 0, "skipped": 0, "total": 0}

        sql = """
            INSERT OR IGNORE INTO recalls (
                campaign_number, manufacturer,
                make, model, model_year,
                component, summary, consequence, remedy, notes,
                recall_date, park_it, created_at
            ) VALUES (
                :campaign_number, :manufacturer,
                :make, :model, :model_year,
                :component, :summary, :consequence, :remedy, :notes,
                :recall_date, :park_it, :created_at
            )
        """
        now = _now()
        params_list = [
            {
                "campaign_number": r.campaign_number,
                "manufacturer":    r.manufacturer,
                "make":            r.make.upper(),
                "model":           r.model,
                "model_year":      r.model_year,
                "component":       r.component,
                "summary":         r.summary,
                "consequence":     r.consequence,
                "remedy":          r.remedy,
                "notes":           r.notes,
                "recall_date":     r.recall_date,
                "park_it":         int(r.park_it),
                "created_at":      now,
            }
            for r in recalls
        ]
        with self._get_connection() as conn:
            cursor = conn.executemany(sql, params_list)
            conn.commit()
            inserted = cursor.rowcount
        skipped = len(recalls) - inserted
        result = {"inserted": inserted, "skipped": skipped, "total": len(recalls)}
        logger.info(
            "Batch insert recalls — inserted=%d, skipped=%d, total=%d",
            inserted, skipped, len(recalls),
        )
        return result

    #  READ — Complaints 

    def get_complaints(
        self,
        make: str,
        model: str,
        year: int,
        limit: int = 100,
    ) -> list:
        """
        Fetch complaints for a specific vehicle.

        Returns:
            List of dicts, newest first. Empty list if nothing found.
        """
        sql = """
            SELECT * FROM complaints
            WHERE make = :make AND model = :model AND model_year = :year
            ORDER BY date_complained DESC
            LIMIT :limit
        """
        with self._get_connection() as conn:
            cursor = conn.execute(sql, {
                "make": make.upper(), "model": model,
                "year": year, "limit": limit,
            })
            rows = [dict(row) for row in cursor.fetchall()]
        logger.debug("get_complaints(%s %s %d) → %d rows", make, model, year, len(rows))
        return rows

    def get_all_complaints(self, limit: int = 100000) -> list:
        """
        Fetch all complaints. Used by the ML pipeline for training.

        Returns:
            List of all complaint dicts up to `limit`.
        """
        sql = "SELECT * FROM complaints LIMIT :limit"
        with self._get_connection() as conn:
            rows = [dict(row) for row in conn.execute(sql, {"limit": limit})]
        logger.info("get_all_complaints() → %d rows", len(rows))
        return rows

    # READ — Recalls 

    def get_recalls(self, make: str, model: str, year: int) -> list:
        """
        Fetch all recalls for a specific vehicle.

        Returns:
            List of recall dicts. Empty list if no recalls found.
        """
        sql = """
            SELECT * FROM recalls
            WHERE make = :make AND model = :model AND model_year = :year
            ORDER BY recall_date DESC
        """
        with self._get_connection() as conn:
            cursor = conn.execute(sql, {
                "make": make.upper(), "model": model, "year": year,
            })
            rows = [dict(row) for row in cursor.fetchall()]
        logger.debug("get_recalls(%s %s %d) → %d rows", make, model, year, len(rows))
        return rows

    def get_all_recalls(self) -> list:
        """Fetch all recalls. Used by ML pipeline for label generation."""
        sql = "SELECT * FROM recalls"
        with self._get_connection() as conn:
            rows = [dict(row) for row in conn.execute(sql)]
        logger.info("get_all_recalls() → %d rows", len(rows))
        return rows

    # READ — Existence Checks

    def complaint_exists(self, odi_number: str) -> bool:
        """Return True if a complaint with this ODI number is already stored."""
        sql = "SELECT 1 FROM complaints WHERE odi_number = :odi LIMIT 1"
        with self._get_connection() as conn:
            return conn.execute(sql, {"odi": odi_number}).fetchone() is not None

    def recall_exists(self, campaign_number: str) -> bool:
        """Return True if a recall with this campaign number is already stored."""
        sql = "SELECT 1 FROM recalls WHERE campaign_number = :cn LIMIT 1"
        with self._get_connection() as conn:
            return conn.execute(sql, {"cn": campaign_number}).fetchone() is not None

    # READ — Counts & Stats

    def count_complaints(self) -> int:
        """Return total number of complaints in the database."""
        with self._get_connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM complaints").fetchone()[0]

    def count_recalls(self) -> int:
        """Return total number of recalls in the database."""
        with self._get_connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM recalls").fetchone()[0]

    def get_stats(self) -> dict:
        """
        Return a summary of what is currently in the database.
        Useful for a quick sanity check after ingestion.

        Example output:
            {
                "total_complaints": 48234,
                "total_recalls":    9871,
                "manufacturers":    ["BMW", "FORD", "TOYOTA", ...],
                "year_range":       {"min": 2015, "max": 2025}
            }
        """
        with self._get_connection() as conn:
            total_complaints = conn.execute(
                "SELECT COUNT(*) FROM complaints"
            ).fetchone()[0]

            total_recalls = conn.execute(
                "SELECT COUNT(*) FROM recalls"
            ).fetchone()[0]

            manufacturers = [
                row[0] for row in conn.execute(
                    "SELECT DISTINCT make FROM complaints ORDER BY make"
                ).fetchall()
            ]

            year_row = conn.execute(
                "SELECT MIN(model_year), MAX(model_year) FROM complaints"
            ).fetchone()

        stats = {
            "total_complaints": total_complaints,
            "total_recalls":    total_recalls,
            "manufacturers":    manufacturers,
            "year_range": {
                "min": year_row[0],
                "max": year_row[1],
            },
        }
        logger.info(
            "DB stats — complaints=%d, recalls=%d, manufacturers=%d",
            total_complaints, total_recalls, len(manufacturers),
        )
        return stats

    # Dunder 

    def __repr__(self) -> str:
        return f"DatabaseManager(db_path='{self.db_path}')"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

# Helper Function

def _now() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.utcnow().isoformat()