"""Centralized filesystem paths for the baseball project."""

from pathlib import Path


# Absolute path to the baseball directory.
BASEBALL_DIR = Path(__file__).resolve().parent

# Baseball data directories.
DATA_DIR = BASEBALL_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Directory containing trained baseball models.
MODELS_DIR = BASEBALL_DIR / "models"

# Processed pitcher dataset.
PITCHER_DATASET_PATH = (
    PROCESSED_DATA_DIR / "pitcher_training_2026.csv"
)

# Saved XGBoost pitcher model.
XGB_MODEL_PATH = MODELS_DIR / "xgb_model.joblib"
