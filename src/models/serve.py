# Model being used: LightGBM
# Threshold: 0.338
# F1: 60.5% | Recall: 70.2% | Precision: 53.2% | ROC-AUC: 0.916
# Trained on: 140,488 complaints | Test set: 35,123 complaints
# Class balance: 11.3% recalled, 88.7% not recalled

"""
serve.py
--------
Model serving module

Loads the trained production model and all required transformers, then exposes clean predict() and predict_proba() functions that
any part of the application can call.

Usage:
    from src.models.serve import ModelServer

    server = ModelServer()

    # Get risk probability (0.0 – 1.0)
    prob = server.predict_proba(
        make      = "BMW",
        model     = "3 Series",
        year      = 2020,
        summary   = "engine stalled on highway without warning",
        component = "ENGINE",
        crash     = False,
    )
    # → 0.73  (73% recall risk)

    # Get binary prediction (0 = safe, 1 = recall likely)
    pred = server.predict(...)
    # → 1
"""

import logging
import os
import pickle
import time
from dataclasses import dataclass
from typing import Optional
import warnings
from huggingface_hub import hf_hub_download

import numpy as np
from scipy.sparse import hstack, csr_matrix

logger = logging.getLogger(__name__)

# Paths

MODELS_DIR    = "data/models"
PROCESSED_DIR = "data/processed"

# Hugging face model serving Config
HF_REPO_ID = os.getenv("HF_REPO_ID", "")

# Threshold found in notebook (maximises F1, recall >= 70%)
# Change this value if we retrain the model with a different threshold
DEFAULT_THRESHOLD = 0.338

VALID_YEAR_MIN = 2000
VALID_YEAR_MAX = 2026

# Input / Output types

@dataclass
class PredictionInput:
    """
    Everything the model needs to make one prediction.
    All fields validated before reaching the model.
    """
    make:      str
    model:     str
    year:      int
    summary:   str            # cleaned or raw complaint text
    component: str = "UNKNOWN"
    crash:     bool = False
    fire:      bool = False


@dataclass
class PredictionResult:
    """
    Everything returned to the caller after one prediction.
    """
    risk_score:   int           # 0–100 (probability × 100)
    probability:  float         # 0.0–1.0
    prediction:   int           # 0 = no recall, 1 = recall likely
    risk_label:   str           # "Low", "Medium", "High"
    threshold:    float         # threshold used
    elapsed_ms:   float         # inference time in milliseconds

    @property
    def risk_color(self) -> str:
        """CSS color for the risk label — used by Streamlit app."""
        return {"Low": "green", "Medium": "orange", "High": "red"}[self.risk_label]

# Exceptions

class ModelNotLoadedError(Exception):
    """Raised when predict is called before load_model()."""
    pass


class InvalidInputError(Exception):
    """Raised when input fails validation."""
    pass

# ModelServer

class ModelServer:
    """
    Loads and serves the production recall prediction model.

    Lazy loading — nothing is loaded from disk until the first
    prediction is requested. Subsequent calls reuse the
    already-loaded model from memory.

    Usage:
        server = ModelServer()
        result = server.predict_proba("BMW", "3 Series", 2020,
                                      "engine stalled", "ENGINE")
    """

    def __init__(
        self,
        models_dir:    str   = MODELS_DIR,
        processed_dir: str   = PROCESSED_DIR,
        threshold:     float = DEFAULT_THRESHOLD,
    ):
        self.models_dir    = models_dir
        self.processed_dir = processed_dir
        self.threshold     = threshold

        # These are None until load_model() is called
        self._model     = None
        self._vectorizer = None
        self._encoder   = None
        self._feature_names = None

        logger.info("ModelServer created (not yet loaded)")

    # Loading 

    def load_model(self):
        """
        Load all model artifacts from disk into memory.

        Loads:
          - production_model.pkl   (LightGBM model)
          - tfidf_vectorizer.pkl   (fitted TF-IDF — same vocab as training)
          - label_encoder.pkl      (fitted component one-hot encoder)
          - feature_names.pkl      (column names for debugging)

        Safe to call multiple times — skips loading if already loaded.
        """
        if self._model is not None:
            logger.debug("Model already loaded — skipping")
            return

        logger.info("Loading model artifacts...")
        t0 = time.perf_counter()

        self._model = self._load_pickle(
            self.models_dir, "production_model.pkl"
        )
        self._vectorizer = self._load_pickle(
            self.processed_dir, "tfidf_vectorizer.pkl"
        )
        self._encoder = self._load_pickle(
            self.processed_dir, "label_encoder.pkl"
        )
        self._feature_names = self._load_pickle(
            self.processed_dir, "feature_names.pkl"
        )

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "Model loaded in %.0fms — threshold=%.3f",
            elapsed, self.threshold,
        )

    def _load_pickle(self, directory: str, filename: str):
        """Load one pickle file, raise a clear error if missing.
        1. first try to load the local file first if the local file exists -> load directly
        2. if the local file does not exist, try to load from Huggingface Hub.
        3. both missing -> raise error
        """
        path = os.path.join(directory, filename)

        if os.file.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
            
        if HF_REPO_ID:
            logger.info(
                f"Downloading model artifact from Huggingface Hub (%s)...",
                filename, HF_REPO_ID,
            )
            os.makedirs(directory, exist_ok=True)
            cached = hf_hub_download(
                repo_id = HF_REPO_ID,
                filename = filename,
                local_dir = directory,
            )
            with open(cached, "rb") as f:
                return pickle.load(f)
            
        raise FileNotFoundError(
            f"Model artifact not found: {path}\n"
            f"Have you run the training notebook? "
            f"Either run the training notebook or set HF_REPO_ID in .env"
        )

    # Public API

    def predict_proba(
        self,
        make:      str,
        model:     str,
        year:      int,
        summary:   str,
        component: str  = "UNKNOWN",
        crash:     bool = False,
        fire:      bool = False,
    ) -> PredictionResult:
        """
        Predict recall probability for a vehicle complaint.

        This is the main function called by the Streamlit app.
        Returns a PredictionResult with risk_score (0-100),
        probability, prediction, and risk label.

        Args:
            make:      Manufacturer e.g. "BMW"
            model:     Model name e.g. "3 Series"
            year:      Model year e.g. 2020
            summary:   Complaint text (raw or pre-cleaned)
            component: Failed component e.g. "ENGINE"
            crash:     Was there a crash?
            fire:      Was there a fire?

        Returns:
            PredictionResult with all risk info.

        Example:
            result = server.predict_proba(
                make="BMW", model="3 Series", year=2020,
                summary="engine stalled on highway",
                component="ENGINE", crash=False,
            )
            print(result.risk_score)   # e.g. 73
            print(result.risk_label)   # "High"
        """
        self._ensure_loaded()

        # Validate input
        inp = self._validate(make, model, year, summary, component, crash, fire)

        t0 = time.perf_counter()

        # Build feature vector
        X = self._build_features(inp)

        # Get probability from model
        #LightGBM sees the mismatch and says "I was trained with named columns but you're giving me an unnamed matrix" — and warns you about it.
        # After building the stacked matrix, convert it to a plain array so LightGBM stops looking for feature names

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            prob = float(self._model.predict_proba(X)[0, 1])

        # Apply threshold to get binary prediction
        pred = int(prob >= self.threshold)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Build result
        risk_score = min(100, int(prob * 100))
        risk_label = self._risk_label(risk_score)

        result = PredictionResult(
            risk_score  = risk_score,
            probability = prob,
            prediction  = pred,
            risk_label  = risk_label,
            threshold   = self.threshold,
            elapsed_ms  = elapsed_ms,
        )

        logger.debug(
            "predict_proba(%s %s %d) → %.3f (%s) in %.1fms",
            make, model, year, prob, risk_label, elapsed_ms,
        )
        return result

    def predict(
        self,
        make:      str,
        model:     str,
        year:      int,
        summary:   str,
        component: str  = "UNKNOWN",
        crash:     bool = False,
        fire:      bool = False,
    ) -> int:
        """
        Return binary prediction: 1 = recall likely, 0 = safe.

        Convenience wrapper around predict_proba().
        Use predict_proba() when you need the full risk score.
        """
        result = self.predict_proba(
            make, model, year, summary, component, crash, fire
        )
        return result.prediction

    def is_loaded(self) -> bool:
        """Return True if model artifacts are loaded in memory."""
        return self._model is not None

    # Feature Building

    def _build_features(self, inp: PredictionInput):
        """
        Transform a PredictionInput into the same feature matrix
        format used during training.

        Must use the EXACT same transformers (fitted on training data)
        so column order and vocabulary match perfectly.
        """
        current_year = 2026

        # 1. TF-IDF on summary text
        X_tfidf = self._vectorizer.transform([inp.summary])

        # 2. Component one-hot
        component = inp.component or "UNKNOWN"
        X_comp = self._encoder.transform([[component]])

        # 3. Temporal features
        X_temporal = csr_matrix([[
            inp.year,
            current_year - inp.year,   # vehicle_age
        ]])

        # 4. Severity flags
        X_severity = csr_matrix([[
            int(inp.crash),
            int(inp.fire),
        ]])

        # Stack in the same order as feature_engineering.py
        return hstack([X_tfidf, X_comp, X_temporal, X_severity]).tocsr()

    # Validation

    def _validate(
        self,
        make:      str,
        model:     str,
        year:      int,
        summary:   str,
        component: str,
        crash:     bool,
        fire:      bool,
    ) -> PredictionInput:
        """
        Validate all inputs before they reach the model.
        Raises InvalidInputError with a clear message on failure.
        """
        errors = []

        if not make or not isinstance(make, str):
            errors.append("make must be a non-empty string")

        if not model or not isinstance(model, str):
            errors.append("model must be a non-empty string")

        if not isinstance(year, int) or not (VALID_YEAR_MIN <= year <= VALID_YEAR_MAX):
            errors.append(
                f"year must be an integer between "
                f"{VALID_YEAR_MIN} and {VALID_YEAR_MAX}, got {year!r}"
            )

        if not summary or not isinstance(summary, str):
            errors.append("summary must be a non-empty string")
        elif len(summary.strip()) < 3:
            errors.append("summary is too short to be meaningful")

        if errors:
            raise InvalidInputError(
                "Input validation failed:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )

        return PredictionInput(
            make      = make.strip().upper(),
            model     = model.strip(),
            year      = year,
            summary   = summary.strip(),
            component = (component or "UNKNOWN").strip().upper(),
            crash     = bool(crash),
            fire      = bool(fire),
        )

    # Helpers

    def _ensure_loaded(self):
        """Load model if not already loaded."""
        if self._model is None:
            self.load_model()

    def _risk_label(self, risk_score: int) -> str:
        """
        Convert a 0–100 risk score to a human-readable label.
        Thresholds match config.yaml risk_thresholds.
        """
        if risk_score < 40:
            return "Low"
        if risk_score < 70:
            return "Medium"
        return "High"

    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded() else "not loaded"
        return (
            f"ModelServer(threshold={self.threshold}, "
            f"status={status})"
        )