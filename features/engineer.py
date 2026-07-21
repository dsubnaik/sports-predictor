"""
File: features/engineer.py

Purpose:
    Creates machine-learning features for the MLB pitcher strikeout model.

Main responsibilities:
    - Calculates pitcher-specific rolling statistics.
    - Calculates opponent strikeout rates.
    - Prevents target leakage by excluding the current game.
    - Provides tests for feature-engineering logic.
"""
import sys

import pandas as pd
from pybaseball import team_batting

# Allow Python to import files from the project's root directory.
# This is needed when running this file directly with:
# python features/engineer.py
sys.path.append(".")

from data.fetch_statcast import aggregate_to_starts, fetch_pitcher_statcast


def rolling_features(df):
    """
    Add rolling statistics from each pitcher's previous five starts.

    The current game's statistics are excluded with shift(1).
    This prevents the model from using information from the game
    it is supposed to predict.

    Required input columns:
        pitcher_id
        game_date
        strikeouts
        swstr_pct
        avg_velocity
        total_pitches

    New columns:
        rolling_k
        rolling_swstr
        rolling_velocity
        rolling_pitches
    """

    # Make a copy so the original DataFrame is not changed.
    df = df.copy()

    # Convert game_date to a proper datetime value.
    # This ensures the games are sorted chronologically.
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Sort each pitcher's starts from oldest to newest.
    df = df.sort_values(
        by=["pitcher_id", "game_date"]
    ).reset_index(drop=True)

    # Separate the rows by pitcher.
    # Every rolling calculation will restart for each pitcher.
    grouped = df.groupby("pitcher_id", group_keys=False)

    # Calculate average strikeouts over the previous five starts.
    #
    # shift(1) excludes the current start.
    # rolling(5) selects the previous five starts.
    # mean() calculates their average.
    df["rolling_k"] = grouped["strikeouts"].transform(
        lambda values: values.shift(1).rolling(window=5).mean()
    )

    # Calculate average swinging-strike percentage
    # over the previous five starts.
    df["rolling_swstr"] = grouped["swstr_pct"].transform(
        lambda values: values.shift(1).rolling(window=5).mean()
    )

    # Calculate average pitch velocity
    # over the previous five starts.
    df["rolling_velocity"] = grouped["avg_velocity"].transform(
        lambda values: values.shift(1).rolling(window=5).mean()
    )

    # Calculate average number of pitches thrown
    # over the previous five starts.
    df["rolling_pitches"] = grouped["total_pitches"].transform(
        lambda values: values.shift(1).rolling(window=5).mean()
    )

    # Return the DataFrame with the new feature columns.
    return df


def fetch_opponent_k(year):
    """
    Fetch every MLB team's batting statistics for a season
    and calculate each team's strikeout rate.

    Strikeout rate is calculated as:

        strikeouts / plate appearances

    Returns:
        A DataFrame containing:
            Team
            k_rate
    """

    # Download season batting statistics for every MLB team.
    batting = team_batting(year)

    # Calculate each team's strikeout rate.
    batting["k_rate"] = batting["SO"] / batting["PA"]

    # Keep only the columns needed for opponent strikeout features.
    opponent_k = batting[["Team", "k_rate"]].copy()

    return opponent_k


def test_rolling_features():
    """
    Test rolling_features() using predictable fake data.

    Pitcher 1 has strikeout totals:
        1, 2, 3, 4, 5, 6

    Pitcher 2 has strikeout totals:
        10, 11, 12, 13, 14, 15

    Expected sixth-start rolling averages:

        Pitcher 1:
        (1 + 2 + 3 + 4 + 5) / 5 = 3

        Pitcher 2:
        (10 + 11 + 12 + 13 + 14) / 5 = 12
    """

    # Create six fake starts for two different pitchers.
    test_data = pd.DataFrame(
        {
            "pitcher_id": [
                1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2,
            ],
            "game_date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-04",
                    "2026-01-05",
                    "2026-01-06",
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-04",
                    "2026-01-05",
                    "2026-01-06",
                ]
            ),
            "strikeouts": [
                1, 2, 3, 4, 5, 6,
                10, 11, 12, 13, 14, 15,
            ],
            "swstr_pct": [
                0.10, 0.11, 0.12, 0.13, 0.14, 0.15,
                0.20, 0.21, 0.22, 0.23, 0.24, 0.25,
            ],
            "avg_velocity": [
                90, 91, 92, 93, 94, 95,
                96, 97, 98, 99, 100, 101,
            ],
            "total_pitches": [
                80, 82, 84, 86, 88, 90,
                90, 92, 94, 96, 98, 100,
            ],
        }
    )

    # Run the feature-engineering function.
    result = rolling_features(test_data)

    # Display the most important test columns.
    print("\nRolling feature test results:\n")

    print(
        result[
            [
                "pitcher_id",
                "game_date",
                "strikeouts",
                "rolling_k",
                "rolling_swstr",
                "rolling_velocity",
                "rolling_pitches",
            ]
        ].to_string(index=False)
    )

    # Find the sixth start for pitcher 1.
    pitcher_1_sixth_start = result[
        result["pitcher_id"] == 1
    ].iloc[5]

    # Find the sixth start for pitcher 2.
    pitcher_2_sixth_start = result[
        result["pitcher_id"] == 2
    ].iloc[5]

    # Confirm that each pitcher's rolling average
    # only uses that pitcher's previous starts.
    assert pitcher_1_sixth_start["rolling_k"] == 3.0
    assert pitcher_2_sixth_start["rolling_k"] == 12.0

    # Confirm that the first five starts do not have rolling values.
    # Five previous starts are required before a rolling average exists.
    pitcher_1_first_five = result[
        result["pitcher_id"] == 1
    ].iloc[:5]

    pitcher_2_first_five = result[
        result["pitcher_id"] == 2
    ].iloc[:5]

    assert pitcher_1_first_five["rolling_k"].isna().all()
    assert pitcher_2_first_five["rolling_k"].isna().all()

    print("\nRolling feature test passed successfully.")


def test_real_pitcher_data():
    """
    Optional test using real Statcast data.

    This function downloads one pitcher's data, aggregates it
    into starts, and creates the rolling features.

    It is not called by default because downloading Statcast
    data takes longer than the fake-data test.
    """

    # MLB player ID used for the test.
    pitcher_id = 543243

    # Download the pitcher's Statcast data for the selected dates.
    statcast_data = fetch_pitcher_statcast(
        pitcher_id,
        "2023-03-30",
        "2023-10-01",
    )

    # Convert pitch-level data into one row per pitching start.
    starts = aggregate_to_starts(statcast_data)

    # Create the rolling features.
    features = rolling_features(starts)

    # Display all rows instead of shortening the output.
    pd.set_option("display.max_rows", None)

    print("\nReal pitcher feature results:\n")

    print(
        features[
            [
                "pitcher_id",
                "game_date",
                "strikeouts",
                "rolling_k",
                "rolling_swstr",
                "rolling_velocity",
                "rolling_pitches",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    # Run the small controlled test first.
    # This does not require downloading any baseball data.
    test_rolling_features()

    # Leave this commented out until the controlled test passes.
    # Remove the # when you want to test real Statcast data.
    #
    test_real_pitcher_data()