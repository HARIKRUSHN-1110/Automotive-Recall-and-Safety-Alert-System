"""
routes.py: All 4 endpoints in the Recall System API.

Feature matrix (must match training exactly — 1025 features):
    [TF-IDF(1000) | component_onehot(20) | year | vehicle_age | crash | fire]

Three HuggingFace artifacts:
    HUGGINGFACE_MODEL_URL → model.pkl(LGBMClassifier)
    HUGGINGFACE_VECTORIZER_URL→ vectorizer.pkl(TF-IDF, 1000 features)
    HUGGINGFACE_ENCODER_URL → label_encoder.pkl(component one-hot encoder)
"""

from fastapi import APIRouter, HTTPException, Query
import psycopg2
import psycopg2.extras
import requests
import pickle
import numpy as np
import os
import logging
from datetime import datetime
from collections import Counter
from scipy.sparse import csr_matrix, hstack
from dotenv import load_dotenv

from .schemas import (
    PredictResponse,
    ComplaintsResponse,
    ComplaintRecord,
    RecallsResponse,
    RecallRecord,
    StatsResponse,
)

# Setup
load_dotenv()
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Recall System"])

DATABASE_URL = os.getenv("DATABASE_URL")
HUGGINGFACE_MODEL_URL = os.getenv("HUGGINGFACE_MODEL_URL")  # model.pkl
CURRENT_YEAR = 2026

# ignoreswarning about missing feature names
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Helpers: derive sibling URLs from model URL
def _sibling_url(filename: str) -> str:
    """
    Replaces the filename in HUGGINGFACE_MODEL_URL with the given filename.
    Assumes all artifacts live in the same HuggingFace repo/folder.
    """
    # strip last segment
    base = HUGGINGFACE_MODEL_URL.rsplit("/", 1)[0]
    return f"{base}/{filename}"

# Artifact cache (download once per server process, reuse forever)

_model_cache     = None
_vectorizer_cache = None
_encoder_cache   = None

def _download_pkl(url: str, name: str):
    """Downloads a .pkl file from a URL and deserializes it."""
    env_token = os.getenv("HF_TOKEN")
    headers   = {"Authorization": f"Bearer {env_token}"} if env_token else {}
    try:
        r = requests.get(url, timeout=30, headers=headers)
        r.raise_for_status()
        return pickle.loads(r.content)
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=503,
            detail=f"Timeout downloading {name} from HuggingFace.")
    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=503,
            detail=f"Failed to download {name} (HTTP {r.status_code}): {url}")
    except pickle.UnpicklingError:
        raise HTTPException(status_code=503,
            detail=f"{name} file is corrupted or wrong format.")

def load_artifacts():
    """
    Downloads and caches all three ML artifacts from HuggingFace.
    Returns (model, vectorizer, encoder).
    """
    global _model_cache, _vectorizer_cache, _encoder_cache

    if all(x is not None for x in [_model_cache, _vectorizer_cache, _encoder_cache]):
        return _model_cache, _vectorizer_cache, _encoder_cache

    if not HUGGINGFACE_MODEL_URL:
        raise HTTPException(status_code=503,
            detail="HUGGINGFACE_MODEL_URL not set in .env")

    vectorizer_url = os.getenv("HUGGINGFACE_VECTORIZER_URL",
                               _sibling_url("tfidf_vectorizer.pkl"))
    encoder_url    = os.getenv("HUGGINGFACE_ENCODER_URL",
                               _sibling_url("label_encoder.pkl"))

    logger.info("Downloading ML artifacts from HuggingFace...")
    _model_cache      = _download_pkl(HUGGINGFACE_MODEL_URL, "model")
    _vectorizer_cache = _download_pkl(vectorizer_url, "vectorizer")
    _encoder_cache    = _download_pkl(encoder_url, "label_encoder")
    logger.info("All artifacts loaded successfully.")

    return _model_cache, _vectorizer_cache, _encoder_cache

# Feature builder — mirrors _build_features() from training exactly

def _build_feature_matrix(rows: list, vectorizer, encoder) -> csr_matrix:
    """
    Builds the same 1025-feature sparse matrix used during training.

    For each complaint row:
        [TF-IDF(1000) | component_onehot | year | vehicle_age | crash | fire]

    Then stacks all rows vertically into one matrix and takes the mean
    across complaints to get a single vector representing the vehicle.

    This should matche the training pipeline in feature_engineering.py exactly.
    """
    feature_rows = []

    for row in rows:
        # TF-IDF on summary text (1000 features)
        summary  = str(row["summary"]) if row["summary"] else ""
        X_tfidf  = vectorizer.transform([summary])          # (1, 1000)

        # Component one-hot encoding
        component = str(row["component"]).upper() if row["component"] else "UNKNOWN"
        try:
            X_comp = encoder.transform([[component]])       # (1, n_components)
        except Exception:
            # If component is unseen, use all-zeros (same shape as encoder output)
            n_comp = len(encoder.categories_[0])
            X_comp = csr_matrix(np.zeros((1, n_comp)))

        # Temporal features: year + vehicle_age
        year         = int(row["model_year"]) if row.get("model_year") else CURRENT_YEAR
        vehicle_age  = CURRENT_YEAR - year
        X_temporal   = csr_matrix([[year, vehicle_age]])    # (1, 2)

        # Severity flags: crash + fire
        crash      = int(bool(row["crash"])) if row["crash"] is not None else 0
        fire       = int(bool(row["fire"]))  if row["fire"] is not None else 0
        X_severity = csr_matrix([[crash, fire]])             # (1, 2)

        # Stack horizontally — same order as training
        X_row = hstack([X_tfidf, X_comp, X_temporal, X_severity]).tocsr()
        feature_rows.append(X_row)

    # Stack all complaint rows vertically (n_complaints, 1025)
    if len(feature_rows) == 1:
        return feature_rows[0]
    return csr_matrix(np.vstack([r.toarray() for r in feature_rows]))

# DB connection
def get_db_connection():
    """
    Establishes a connection to the Supabase database using
    the DATABASE_URL environment variable.

    Raises an HTTPException (503) if the connection fails.
    """
    if not DATABASE_URL:
        raise HTTPException(status_code=503,
            detail="DATABASE_URL not set in .env")
    try:
        return psycopg2.connect(
            DATABASE_URL,
            cursor_factory=psycopg2.extras.RealDictCursor
        )
    except psycopg2.OperationalError as e:
        logger.error(f"Supabase connection failed: {e}")
        raise HTTPException(status_code=503,
            detail="Cannot connect to Supabase. Check DATABASE_URL.")

# Utilities

def risk_label_from_score(score: float) -> str:
    if score < 40:   
        return "low"
    elif score < 70: 
        return "medium"
    else:            
        return "high"


def extract_top_components(rows, top_n=3) -> list[str]:
    components = [
        row["component"] for row in rows
        if row.get("component") and str(row["component"]).strip()
    ]
    return [c for c, _ in Counter(components).most_common(top_n)]

# GET /api/v1/predict

@router.get("/predict", response_model=PredictResponse,
    summary="Predict recall risk for a vehicle",
    description="""
Returns a recall risk score (0–100) for a given vehicle make/model/year.

Internally fetches all complaints for the vehicle from Supabase, builds
the same 1025-feature matrix used during training (TF-IDF + component
encoding + temporal + severity flags), runs each complaint through the
LGBMClassifier, and averages the recall probabilities.
""")
def predict_recall_risk(
    make:  str = Query(..., example="BMW",      description="Vehicle manufacturer"),
    model: str = Query(..., example="3 Series", description="Vehicle model"),
    year:  int = Query(..., ge=1990, le=2026,   example=2020, description="Model year"),
):
    conn = get_db_connection()
    try:
        cur = conn.cursor()

        # Fetch all fields needed to build the feature matrix
        cur.execute("""
            SELECT model_year, component, summary, crash, fire
            FROM complaints
            WHERE UPPER(make)  = UPPER(%s)
              AND UPPER(model) = UPPER(%s)
              AND model_year   = %s
            ORDER BY date_complained DESC
            LIMIT 200
        """, (make, model, year))
        rows = cur.fetchall()

        if not rows:
            raise HTTPException(status_code=404,
                detail=f"No complaints found for {make} {model} {year}.")

        # Load artifacts (cached after first call)
        ml_model, vectorizer, encoder = load_artifacts()

        # Build 1025-feature matrix matching training pipeline exactly
        X = _build_feature_matrix(rows, vectorizer, encoder)

        # Predict recall probability per complaint, then average
        # This gives a vehicle-level score rather than a per-complaint score
        probas     = ml_model.predict_proba(X)[:, 1]   # probability of recall
        avg_proba  = float(np.mean(probas))
        confidence = round(avg_proba, 4)
        risk_score = round(confidence * 100, 1)

        return PredictResponse(
            make=make, model=model, year=year,
            risk_score=risk_score,
            risk_label=risk_label_from_score(risk_score),
            confidence=confidence,
            complaint_count=len(rows),
            top_components=extract_top_components(rows),
            message=f"Prediction averaged across {len(rows)} complaints."
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# GET /api/v1/complaint

@router.get("/complaints", response_model=ComplaintsResponse,
    summary="Get NHTSA complaints for a vehicle",
    description="Returns complaint records from Supabase. Max 50 per request.")
def get_complaints(
    make:  str = Query(..., example="BMW",      description="Vehicle manufacturer"),
    model: str = Query(..., example="3 Series", description="Vehicle model"),
    year:  int = Query(..., ge=1990, le=2026,   example=2020, description="Model year"),
    limit: int = Query(default=10, ge=1, le=50, description="Max results (1-50)"),
):
    conn = get_db_connection()
    try:
        cur = conn.cursor()

        cur.execute("""
            SELECT COUNT(*) as total FROM complaints
            WHERE UPPER(make)  = UPPER(%s)
              AND UPPER(model) = UPPER(%s)
              AND model_year   = %s
        """, (make, model, year))
        total = cur.fetchone()["total"]

        if total == 0:
            raise HTTPException(status_code=404,
                detail=f"No complaints found for {make} {model} {year}.")

        cur.execute("""
            SELECT odi_number, make, model, model_year,
                   component, summary, crash, fire,
                   injuries, deaths,
                   date_complained, date_of_incident, vehicle_speed
            FROM complaints
            WHERE UPPER(make)  = UPPER(%s)
              AND UPPER(model) = UPPER(%s)
              AND model_year   = %s
            ORDER BY date_complained DESC
            LIMIT %s
        """, (make, model, year, limit))
        rows = cur.fetchall()

        complaints = [
            ComplaintRecord(
                odi_number=str(r["odi_number"]) if r["odi_number"] else None,
                make=r["make"],
                model=r["model"],
                year=r["model_year"],
                component=r["component"],
                summary=r["summary"],
                crash=(
                    bool(r["crash"]) if r["crash"] is not None else None
                ),
                fire=(
                    bool(r["fire"]) if r["fire"] is not None else None
                ),
                injuries=r["injuries"],
                deaths=r["deaths"],
                date_complained=(
                    str(r["date_complained"])
                    if r["date_complained"] else None
                ),
                date_of_incident=(
                    str(r["date_of_incident"])
                    if r["date_of_incident"] else None
                ),
                vehicle_speed=(
                    str(r["vehicle_speed"])
                    if r["vehicle_speed"] else None
                ),
            )
            for r in rows
        ]

        return ComplaintsResponse(
            make=make,
            model=model,
            year=year,
            total_found=total,
            returned=len(complaints),
            complaints=complaints,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Complaints query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# GET /api/v1/recalls

@router.get("/recalls", response_model=RecallsResponse,
    summary="Get official recalls for a vehicle",
    description="Returns confirmed NHTSA recalls from Supabase. Empty list = no recalls issued yet.")
def get_recalls(
    make:  str = Query(..., example="BMW",      description="Vehicle manufacturer"),
    model: str = Query(..., example="3 Series", description="Vehicle model"),
    year:  int = Query(..., ge=1990, le=2026,   example=2020, description="Model year"),
):
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT campaign_number, manufacturer, make, model, model_year,
                   component, summary, consequence, remedy,
                   notes, recall_date, park_it
            FROM recalls
            WHERE UPPER(make)  = UPPER(%s)
              AND UPPER(model) = UPPER(%s)
              AND model_year   = %s
            ORDER BY recall_date DESC
        """, (make, model, year))
        rows = cur.fetchall()

        recalls = [
            RecallRecord(
                campaign_number = r["campaign_number"],
                manufacturer    = r["manufacturer"],
                make            = r["make"],
                model           = r["model"],
                year            = r["model_year"],
                component       = r["component"],
                summary         = r["summary"],
                consequence     = r["consequence"],
                remedy          = r["remedy"],
                notes           = r["notes"],
                recall_date=(
                    str(r["recall_date"]) if r["recall_date"] else None
                ),
                park_it=(
                    bool(r["park_it"]) if r["park_it"] is not None else None
                ),
            )
            for r in rows
        ]

        return RecallsResponse(
            make=make,
            model=model,
            year=year,
            total_recalls=len(recalls),
            recalls=recalls,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recalls query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# GET /api/v1/stats

@router.get("/stats", response_model=StatsResponse,
    summary="System-wide dataset statistics",
    description="Aggregate counts and date range across the full Supabase dataset.")
def get_stats():
    conn = get_db_connection()
    try:
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) as total FROM complaints")
        total_complaints = cur.fetchone()["total"]

        cur.execute("SELECT COUNT(*) as total FROM recalls")
        total_recalls = cur.fetchone()["total"]

        cur.execute("SELECT COUNT(DISTINCT UPPER(make)) as total FROM complaints")
        manufacturers_covered = cur.fetchone()["total"]

        cur.execute("""
            SELECT MIN(date_complained) as start_date,
                   MAX(date_complained) as end_date
            FROM complaints
        """)
        date_row = cur.fetchone()

        return StatsResponse(
            total_complaints=total_complaints,
            total_recalls=total_recalls,
            manufacturers_covered=manufacturers_covered,
            date_range_start=(
                str(date_row["start_date"])
                if date_row["start_date"] else None
            ),
            date_range_end=(
                str(date_row["end_date"])
                if date_row["end_date"] else None
            ),
            last_updated=datetime.now().isoformat(),
            api_version="1.0.0"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stats query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()