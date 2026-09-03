"""Centralized filesystem paths for the football project."""

from pathlib import Path


# Absolute path to the football directory.
FOOTBALL_DIR = Path(__file__).resolve().parent

# Football data directories.
DATA_DIR = FOOTBALL_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
