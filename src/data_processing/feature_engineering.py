"""
feature_engineering.py
-----------------------
Converts cleaned complaint data into a numerical feature matrix ready for ML model training.

Four feature groups:
  1. TF-IDF on summary_clean       (1000 features)
  2. Component one-hot encoding    (~20 features)
  3. Temporal features             (2 features: year, quarter)
  4. Severity flags                (2 features: crash, fire)

Output (saved to data/processed/):
  X_train.pkl, X_test.pkl   — feature matrices
  y_train.pkl, y_test.pkl   — labels (was_recalled)
  tfidf_vectorizer.pkl      — fitted vectorizer (needed at prediction time)
  label_encoder.pkl         — fitted component encoder
  feature_names.pkl         — list of all feature column names

Usage:
    from src.data_processing.feature_engineering import FeatureEngineer

    fe = FeatureEngineer()
    fe.build_features()
"""

import logging
import os
import pickle
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)

# Config

DB_PATH        = "data/automotive_recall.db"
PROCESSED_DIR  = "data/processed"
RANDOM_SEED    = 42
TEST_SIZE      = 0.2

# TF-IDF settings
TFIDF_MAX_FEATURES = 1000
TFIDF_NGRAM_RANGE  = (1, 2)   # unigrams + bigrams ("engine stall" as one feature)
TFIDF_MIN_DF       = 3        # ignore words appearing in fewer than 3 complaints

# Top N components to keep — the rest become "OTHER"
# Keeps one-hot matrix small without losing important signal
TOP_N_COMPONENTS = 20

# FeatureEngineering

class FeatureEngineer:
    """
    Builds the feature matrix from complaints_processed table.

    Usage:
        fe = FeatureEngineer()
        fe.build_features()
    """

    def __init__(
        self,
        db_path:       str = DB_PATH,
        processed_dir: str = PROCESSED_DIR,
        random_seed:   int = RANDOM_SEED,
    ):
        self.db_path       = db_path
        self.processed_dir = processed_dir
        self.random_seed   = random_seed

        os.makedirs(processed_dir, exist_ok=True)
        logger.info("FeatureEngineer ready — output: %s", processed_dir)

    # Main Entry Point

    def build_features(self) -> dict:
        """
        Full pipeline: load → engineer features → split → save.

        Returns:
            Dict with shapes and stats:
            {
                "X_train_shape": (N, F),
                "X_test_shape":  (N, F),
                "n_features":    F,
                "recall_rate":   float,
                "feature_names": [str, ...],
            }
        """
        print("\n   Feature Engineering Pipeline")
        print(f"    Database : {self.db_path}")
        print(f"    Output   : {self.processed_dir}\n")

        # Step 1 — Load
        df = self._load_data()

        # Step 2 — Build each feature group
        print("  Building features...")
        X_tfidf,     tfidf_vectorizer = self._build_tfidf(df)
        X_component, label_encoder    = self._build_component_features(df)
        X_temporal                    = self._build_temporal_features(df)
        X_severity                    = self._build_severity_features(df)

        # Step 3 — Stack all features into one matrix
        X, feature_names = self._combine_features(
            X_tfidf, tfidf_vectorizer,
            X_component, label_encoder,
            X_temporal,
            X_severity,
        )

        # Step 4 — Labels
        y = df["was_recalled"].values
        recall_rate = y.mean() * 100

        # Step 5 — Train / test split (stratified to preserve recall ratio)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size    = TEST_SIZE,
            random_state = self.random_seed,
            stratify     = y,   # ensures same recall % in both splits
        )

        # Step 6 — Save everything
        self._save_artifacts(
            X_train, X_test, y_train, y_test,
            tfidf_vectorizer, label_encoder, feature_names,
        )

        # Step 7 — Report
        stats = {
            "X_train_shape": X_train.shape,
            "X_test_shape":  X_test.shape,
            "n_features":    X.shape[1],
            "recall_rate":   recall_rate,
            "feature_names": feature_names,
        }

        self._print_summary(stats, y_train, y_test)
        return stats

    # Step 1: Load

    def _load_data(self) -> pd.DataFrame:
        """Load the complaints_processed table into a DataFrame."""
        conn = sqlite3.connect(self.db_path)
        df   = pd.read_sql_query(
            "SELECT * FROM complaints_processed", conn
        )
        conn.close()

        print(f"  Loaded {len(df):,} processed complaints")

        # Fill any remaining nulls with safe defaults
        df["summary_clean"] = df["summary_clean"].fillna("")
        df["component"]     = df["component"].fillna("UNKNOWN")
        df["model_year"]    = df["model_year"].fillna(2018).astype(int)
        df["crash"]         = df["crash"].fillna(0).astype(int)
        df["fire"]          = df["fire"].fillna(0).astype(int)

        return df

    # Step 2.1 TF-IDF

    def _build_tfidf(self, df: pd.DataFrame):
        """
        Build TF-IDF matrix from cleaned complaint summaries.

        Uses bigrams (ngram_range=(1,2)) so "engine stall" is treated
        as one feature rather than "engine" and "stall" separately.
        This captures meaningful two-word phrases that signal defects.

        Returns:
            (sparse matrix of shape [n, 1000], fitted TfidfVectorizer)
        """
        vectorizer = TfidfVectorizer(
            max_features = TFIDF_MAX_FEATURES,
            ngram_range  = TFIDF_NGRAM_RANGE,
            min_df       = TFIDF_MIN_DF,
            sublinear_tf = True,   # apply log(1+tf) — reduces impact of very common words appearing many times in one doc
        )
        X_tfidf = vectorizer.fit_transform(df["summary_clean"])
        print(f"  TF-IDF matrix      : {X_tfidf.shape}  "
              f"(vocab size: {len(vectorizer.vocabulary_):,})")
        return X_tfidf, vectorizer

    # Step 2.2: Component One-Hot
    def _build_component_features(self, df: pd.DataFrame):
        """
        One-hot encode the component field.

        Only the top N most common components get their own column.
        Everything else is bucketed as "OTHER" — this prevents the
        matrix from getting very wide due to rare component names.

        Returns:
            (sparse matrix of shape [n, ~20], fitted OneHotEncoder)
        """
        # Keep only top N components, bucket the rest as OTHER
        top_components = (
            df["component"]
            .value_counts()
            .head(TOP_N_COMPONENTS)
            .index.tolist()
        )
        df_comp = df["component"].apply(
            lambda c: c if c in top_components else "OTHER"
        ).values.reshape(-1, 1)

        encoder = OneHotEncoder(
            sparse_output = True,
            handle_unknown = "ignore",   # unknown components at prediction time
        )
        X_component = encoder.fit_transform(df_comp)
        n_cats = X_component.shape[1]
        print(f"  Component one-hot  : {X_component.shape}  ({n_cats} categories)")
        return X_component, encoder

    # Step 2.3: Temporal Features

    def _build_temporal_features(self, df: pd.DataFrame) -> csr_matrix:
        """
        Extract year and vehicle age as numerical features.

        Features:
          - model_year      : raw year (2015–2025)
          - vehicle_age     : current year minus model_year
                              older vehicles → more time to accumulate
                              complaints and be recalled

        Why not include month/quarter from date_complained?
        The date_complained field has many nulls in NHTSA data.
        model_year is always present and more reliable.

        Returns:
            Dense matrix converted to sparse, shape [n, 2]
        """
        current_year = datetime.utcnow().year

        temporal = pd.DataFrame({
            "model_year":  df["model_year"],
            "vehicle_age": current_year - df["model_year"],
        }).values.astype(float)

        print(f"  Temporal features  : {temporal.shape}  (year, vehicle_age)")
        return csr_matrix(temporal)

    # Step 2.4: Severity Flags

    def _build_severity_features(self, df: pd.DataFrame) -> csr_matrix:
        """
        Binary severity flags — crash and fire.

        These are already 0/1 integers so no transformation needed.
        They are among the strongest predictors of recall probability.

        Returns:
            Sparse matrix of shape [n, 2]
        """
        severity = df[["crash", "fire"]].values.astype(float)
        print(f"  Severity flags     : {severity.shape}  (crash, fire)")
        return csr_matrix(severity)

    # Step 3: Combine

    def _combine_features(
        self,
        X_tfidf,     tfidf_vectorizer,
        X_component, label_encoder,
        X_temporal,
        X_severity,
    ):
        """
        Horizontally stack all feature groups into one matrix.

        scipy.sparse.hstack is used because TF-IDF and one-hot
        matrices are sparse (mostly zeros). Keeping them sparse
        avoids materialising a huge dense matrix in memory.

        Returns:
            (combined sparse matrix [n, total_features], feature_names list)
        """
        X = hstack([X_tfidf, X_component, X_temporal, X_severity])

        # Build feature name list (useful for model interpretation later)
        tfidf_names     = [f"tfidf_{w}" for w in tfidf_vectorizer.get_feature_names_out()]
        component_names = [f"comp_{c}" for c in label_encoder.categories_[0]]
        temporal_names  = ["model_year", "vehicle_age"]
        severity_names  = ["crash", "fire"]

        feature_names = tfidf_names + component_names + temporal_names + severity_names

        print(f"\n  Combined matrix    : {X.shape}")
        print(f"    TF-IDF           : {X_tfidf.shape[1]} features")
        print(f"    Component        : {X_component.shape[1]} features")
        print(f"    Temporal         : {X_temporal.shape[1]} features")
        print(f"    Severity         : {X_severity.shape[1]} features")
        print(f"    ─────────────────────────────")
        print(f"    Total            : {X.shape[1]} features")

        return X, feature_names

    # Step 5: Split 
    # (handled inline in build_features)

    # Step 6: Save 

    def _save_artifacts(
        self,
        X_train, X_test, y_train, y_test,
        tfidf_vectorizer, label_encoder, feature_names,
    ):
        """
        Save all matrices and fitted objects to data/processed/.

        Why save the fitted vectorizer and encoder?
        At prediction time, a new complaint must be transformed using
        EXACTLY the same vocabulary and categories used during training.
        If you refit on new data the column order changes and prediction breaks.
        """
        print(f"\n  Saving artifacts to {self.processed_dir}/...")

        artifacts = {
            "X_train.pkl":          X_train,
            "X_test.pkl":           X_test,
            "y_train.pkl":          y_train,
            "y_test.pkl":           y_test,
            "tfidf_vectorizer.pkl": tfidf_vectorizer,
            "label_encoder.pkl":    label_encoder,
            "feature_names.pkl":    feature_names,
        }

        for filename, obj in artifacts.items():
            path = os.path.join(self.processed_dir, filename)
            with open(path, "wb") as f:
                pickle.dump(obj, f)
            logger.debug("Saved %s", path)

        print(f"  Saved {len(artifacts)} files")

    #Step 7: Summary

    def _print_summary(self, stats: dict, y_train, y_test):
        """Print a clean summary of what was built."""
        train_shape  = stats["X_train_shape"]
        test_shape   = stats["X_test_shape"]
        recall_rate  = stats["recall_rate"]

        train_recalled = y_train.sum()
        test_recalled  = y_test.sum()

        print(f"\n{'='*50}")
        print(f"  Feature Engineering Complete")
        print(f"{'='*50}")
        print(f"  Train set  : {train_shape[0]:>8,} rows × {train_shape[1]:,} features")
        print(f"  Test set   : {test_shape[0]:>8,} rows × {test_shape[1]:,} features")
        print(f"  Train recalled : {train_recalled:,} ({train_recalled/len(y_train)*100:.1f}%)")
        print(f"  Test  recalled : {test_recalled:,}  ({test_recalled/len(y_test)*100:.1f}%)")
        print(f"\n  Stratified split preserved {recall_rate:.1f}% recall rate in both sets")
        print(f"  Files saved to {self.processed_dir}/")
        print(f"{'='*50}\n")