"""Build and validate the current-season official-starter pitch dataset."""

from datetime import date
from pathlib import Path

import pandas as pd
from pybaseball import statcast

from data.fetch_statcast import aggregate_to_starts
from data.fetch_mlb_starters import fetch_mlb_starters
from features.engineer import (
    add_opponent_k_rate,
    build_team_k_history,
    rolling_features,
)

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
    """Download all MLB Statcast pitches within the selected date range."""

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
    """Summarize every pitcher found in the filtered Statcast data.

    This does not exclude pitchers based on innings or appearances.
    New and recently debuted pitchers therefore remain available.
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
    """Keep only Statcast pitches thrown by each game's official starters.

    A row is retained only when both its game ID and pitcher ID match
    an official starter record.
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

    # Matching on both game_pk and pitcher_id prevents the home and away
    # starters in the same game from being collapsed together.
    starter_keys = starter_frame[
        [
            "game_pk",
            "pitcher_id",
            "home_away",
            "team_id",
            "team_name",
            "opponent_id",
            "opponent_name",
        ]    
    ].drop_duplicates(
        subset=["game_pk", "pitcher_id"]
    )
       

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

    start_data = aggregate_to_starts(starter_pitch_data)

    team_k_history = build_team_k_history(
    pitch_data=season_pitch_data,
    starters=official_starters,
    )

    start_data = add_opponent_k_rate(
        start_data=start_data,
        team_k_history=team_k_history,
    )

    official_starter_frame = pd.DataFrame(official_starters)

    official_start_keys = official_starter_frame[
        ["game_pk", "pitcher_id"]
    ].drop_duplicates()

    duplicate_starts = official_starter_frame[
        official_starter_frame.duplicated(
            subset=["game_pk", "pitcher_id"],
            keep=False,
        )
    ]

    # These diagnostics protect the official-starter matching assumptions before
    # any future feature-building step consumes the dataset.
    print(
        f"\nOfficial starter records: "
        f"{len(official_starter_frame):,}"
    )

    print(
        f"Unique official starter-game pairs: "
        f"{len(official_start_keys):,}"
    )

    print(
        f"Duplicate official starter records: "
        f"{len(duplicate_starts):,}"
    )

    if not duplicate_starts.empty:
        print(duplicate_starts.to_string(index=False))

    unique_official_starters = (
        official_starter_frame.drop_duplicates(
            subset=["game_pk", "pitcher_id"]
        )
    )

    matched_start_keys = start_data[
        ["game_pk", "pitcher_id"]
    ].drop_duplicates()

    unmatched_starts = unique_official_starters.merge(
        matched_start_keys,
        on=["game_pk", "pitcher_id"],
        how="left",
        indicator=True,
    )

    unmatched_starts = unmatched_starts[
        unmatched_starts["_merge"] == "left_only"
    ].drop(columns="_merge")

    # Unmatched starts usually indicate a schedule/boxscore record that did not
    # line up with Statcast pitch rows, so they are surfaced instead of hidden.
    print(
        f"\nOfficial starts without matching Statcast pitches: "
        f"{len(unmatched_starts):,}"
    )

    if not unmatched_starts.empty:
        print(unmatched_starts.to_string(index=False))

    print(
        f"\nAggregated the filtered pitches into "
        f"{len(start_data):,} starter-game rows."
    )

    print(
        f"Kept {len(starter_pitch_data):,} pitches from "
        f"{len(official_start_keys):,} unique official starts."
    )

    pitchers = summarize_pitchers(starter_pitch_data)

    print("\nStarting pitchers discovered:")
    print(pitchers.head(20).to_string(index=False))

    print(
        f"\nTotal unique starting pitchers discovered: "
        f"{len(pitchers):,}"
    )

    feature_data = rolling_features(start_data)

    model_features = [
        "rolling_k",
        "rolling_swstr",
        "rolling_velocity",
        "rolling_pitches",
        "opponent_k_rate",
    ]

    model_data = feature_data.dropna(
        subset=model_features
    ).copy()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_data.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nCreated rolling features for "
        f"{len(feature_data):,} starter-game rows."
    )

    print(
        f"Removed "
        f"{len(feature_data) - len(model_data):,} rows "
        f"without five previous starts."
    )

    print(
        f"Saved {len(model_data):,} model-ready rows to "
        f"{OUTPUT_PATH}."
    )
