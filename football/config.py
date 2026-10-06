"""Centralized filesystem paths for the football project."""

from pathlib import Path


# Absolute path to the football directory.
FOOTBALL_DIR = Path(__file__).resolve().parent

# Football data directories.
DATA_DIR = FOOTBALL_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Explicit, Git-ignored local snapshot consumed by the optional QB UI panel.
QB_PASSING_YARDS_PROJECTION_SNAPSHOT_PATH = (
    PROCESSED_DATA_DIR / "qb_passing_yards_projection_snapshots" / "current.json"
)
