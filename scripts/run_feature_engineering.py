"""
run_feature_engineering.py
---------------------------
Build the feature matrix for ML training.

Prerequisites:
    - complaints_processed table must exist
    - python scripts/run_preprocessing.py

Output saved to data/processed/:
    X_train.pkl, X_test.pkl      — feature matrices (sparse)
    y_train.pkl, y_test.pkl      — recall labels (0/1)
    tfidf_vectorizer.pkl         — fitted TF-IDF (needed at prediction time)
    label_encoder.pkl            — fitted component encoder
    feature_names.pkl            — list of all feature column names
"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.feature_engineering import FeatureEngineer

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

if __name__ == "__main__":
    fe    = FeatureEngineer()
    stats = fe.build_features()

    # Quick shape verification
    print("Verification:")
    print(f"  X_train : {stats['X_train_shape']}")
    print(f"  X_test  : {stats['X_test_shape']}")
    print(f"  Features: {stats['n_features']:,}")