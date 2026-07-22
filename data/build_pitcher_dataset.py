"""
File: data/build_pitcher_dataset.py

Purpose:
    Builds the current-season dataset used to train the MLB pitcher
    strikeout prediction model.

Main responsibilities:
    - Download current-season Statcast data.
    - Identify every pitcher who has appeared during the season.
    - Keep actual starting-pitcher appearances only.
    - Convert pitch-level data into one row per start.
    - Create pitcher-specific rolling features.
    - Preserve pitchers with limited current-season history.
    - Save the completed dataset as a CSV file.

Current development stage:
    Retrieve current-season Statcast data and identify pitchers
    automatically without requiring a minimum number of starts.
"""

from datetime import date
from pathlib import Path

import pandas as pd
from pybaseball import statcast

from data.fetch_statcast import aggregate_to_starts
from features.engineer import rolling_features
from data.fetch_mlb_starters import fetch_mlb_starters

# Season used to build the training dataset.
SEASON = 2026

# MLB regular-season data generally begins near the end of March.
# We can make this more precise later using the official schedule.
START_DATE = f"{SEASON}-03-20"

# Use today's date so the script can be rerun throughout the season.
END_DATE = date.today().isoformat()

# Location where the completed dataset will be saved.
OUTPUT_PATH = Path("data/processed/pitcher_training_2026.csv")

def fetch_current_season_statcast(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Download all MLB Statcast pitches within the selected date range.

    Parameters:
        start_date:
            First date to include, formatted as YYYY-MM-DD.

        end_date:
            Last date to include, formatted as YYYY-MM-DD.

    Returns:
        A DataFrame containing one row per recorded pitch.
    """

    print(
        f"Downloading MLB Statcast data from "
        f"{start_date} through {end_date}..."
    )

    pitch_data = statcast(
        start_dt=start_date,
        end_dt=end_date,
    )

    if pitch_data.empty:
        raise ValueError(
            "Statcast returned no data for the selected date range."
        )

    print(f"Downloaded {len(pitch_data):,} pitches.")

    return pitch_data

def summarize_pitchers(pitch_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary of every pitcher found in the Statcast data.

    This does not exclude pitchers based on innings or appearances.
    New and recently debuted pitchers therefore remain available.

    Parameters:
        pitch_data:
            Pitch-level Statcast DataFrame.

    Returns:
        A DataFrame containing each pitcher's ID, name, pitch count,
        game count, and first and most recent appearance dates.
    """

    required_columns = {
        "pitcher",
        "player_name",
        "game_pk",
        "game_date",
    }

    missing_columns = required_columns.difference(pitch_data.columns)

    if missing_columns:
        raise ValueError(
            "Statcast data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    pitcher_summary = (
        pitch_data.groupby("pitcher")
        .agg(
            pitcher_name=("player_name", "first"),
            total_pitches=("pitcher", "size"),
            games_appeared=("game_pk", "nunique"),
            first_appearance=("game_date", "min"),
            latest_appearance=("game_date", "max"),
        )
        .reset_index()
        .rename(columns={"pitcher": "pitcher_id"})
    )

    pitcher_summary["first_appearance"] = pd.to_datetime(
        pitcher_summary["first_appearance"]
    )

    pitcher_summary["latest_appearance"] = pd.to_datetime(
        pitcher_summary["latest_appearance"]
    )

    pitcher_summary = pitcher_summary.sort_values(
        ["games_appeared", "total_pitches"],
        ascending=False,
    ).reset_index(drop=True)

    return pitcher_summary

def filter_to_official_starts(
    pitch_data: pd.DataFrame,
    starters: list[dict],
) -> pd.DataFrame:
    """
    Keep only Statcast pitches thrown by each game's official starters.

    A row is retained only when both its game ID and pitcher ID match
    an official starter record.

    Parameters:
        pitch_data:
            Pitch-level Statcast data.

        starters:
            Official starter records returned by fetch_mlb_starters().

    Returns:
        A DataFrame containing pitches from official starts only.
    """

    required_columns = {"game_pk", "pitcher"}
    missing_columns = required_columns.difference(pitch_data.columns)

    if missing_columns:
        raise ValueError(
            "Statcast data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    starter_frame = pd.DataFrame(starters)

    if starter_frame.empty:
        raise ValueError("No official starter records were provided.")

    starter_keys = starter_frame[
        ["game_pk", "pitcher_id"]
    ].drop_duplicates()

    starter_pitches = pitch_data.merge(
        starter_keys,
        left_on=["game_pk", "pitcher"],
        right_on=["game_pk", "pitcher_id"],
        how="inner",
        validate="many_to_one",
    )

    starter_pitches = starter_pitches.drop(
        columns="pitcher_id"
    )

    return starter_pitches

if __name__ == "__main__":
    season_pitch_data = fetch_current_season_statcast(
        start_date=START_DATE,
        end_date=END_DATE,
    )

    official_starters = fetch_mlb_starters(
        start_date=START_DATE,
        end_date=END_DATE,
    )

    starter_pitch_data = filter_to_official_starts(
        pitch_data=season_pitch_data,
        starters=official_starters,
    )

    print(
        f"Kept {len(starter_pitch_data):,} pitches from "
        f"{len(official_starters):,} official starts."
    )

    pitchers = summarize_pitchers(starter_pitch_data)

    print("\nStarting pitchers discovered:")
    print(pitchers.head(20).to_string(index=False))

    print(
        f"\nTotal unique starting pitchers discovered: "
        f"{len(pitchers):,}"
    )