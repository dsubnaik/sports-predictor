"""Create model features for pitcher strikeout prediction."""

import sys

import pandas as pd
from pybaseball import team_batting

# Allow Python to import files from the project's root directory.
# This is needed when running this file directly with:
# python features/engineer.py
sys.path.append(".")

from baseball.data.fetch_statcast import aggregate_to_starts, fetch_pitcher_statcast


def rolling_features(df):
    """Add rolling statistics from each pitcher's previous five starts.

    The current game's statistics are excluded with shift(1).
    This prevents the model from using information from the game
    it is supposed to predict.
    """

    df = df.copy()

    # Chronological order is required because rolling features must only use
    # starts that happened before the row being predicted.
    df["game_date"] = pd.to_datetime(df["game_date"])

    df = df.sort_values(
        by=["pitcher_id", "game_date"]
    ).reset_index(drop=True)

    # Group by pitcher so one pitcher's history never leaks into another's
    # rolling averages.
    grouped = df.groupby("pitcher_id", group_keys=False)

    # shift(1) excludes the current start and prevents target leakage.
    df["rolling_k"] = grouped["strikeouts"].transform(
        lambda values: values.shift(1).rolling(window=5).mean()
    )

    df["rolling_swstr"] = grouped["swstr_pct"].transform(
        lambda values: values.shift(1).rolling(window=5).mean()
    )

    df["rolling_velocity"] = grouped["avg_velocity"].transform(
        lambda values: values.shift(1).rolling(window=5).mean()
    )

    df["rolling_pitches"] = grouped["total_pitches"].transform(
        lambda values: values.shift(1).rolling(window=5).mean()
    )

    return df


def fetch_opponent_k(year):
    """Fetch MLB team batting stats and calculate strikeout rate.

    Strikeout rate is calculated as:

        strikeouts / plate appearances
    """

    batting = team_batting(year)
    batting["k_rate"] = batting["SO"] / batting["PA"]
    opponent_k = batting[["Team", "k_rate"]].copy()

    return opponent_k


def test_rolling_features():
    """Demo rolling_features() with predictable fake data.

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

    result = rolling_features(test_data)

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

    pitcher_1_sixth_start = result[
        result["pitcher_id"] == 1
    ].iloc[5]

    pitcher_2_sixth_start = result[
        result["pitcher_id"] == 2
    ].iloc[5]

    # Each pitcher's rolling average must use only that pitcher's history.
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
    """Demo feature generation against one real Statcast pitcher sample.

    It is not called by default because downloading Statcast data is slow.
    """

    pitcher_id = 543243

    statcast_data = fetch_pitcher_statcast(
        pitcher_id,
        "2023-03-30",
        "2023-10-01",
    )

    starts = aggregate_to_starts(statcast_data)
    features = rolling_features(starts)

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
def build_team_k_history(
    pitch_data: pd.DataFrame,
    starters: list[dict],
) -> pd.DataFrame:
    """Calculate each team's strikeout rate using games before each date."""

    starter_frame = pd.DataFrame(starters)

    team_games = starter_frame[
        [
            "game_pk",
            "game_date",
            "home_away",
            "team_id",
        ]
    ].drop_duplicates()

    home_teams = (
        team_games[team_games["home_away"] == "home"]
        [["game_pk", "team_id"]]
        .rename(columns={"team_id": "home_team_id"})
    )

    away_teams = (
        team_games[team_games["home_away"] == "away"]
        [["game_pk", "team_id"]]
        .rename(columns={"team_id": "away_team_id"})
    )

    game_teams = home_teams.merge(
        away_teams,
        on="game_pk",
        how="inner",
        validate="one_to_one",
    )

    batting_data = pitch_data.merge(
        game_teams,
        on="game_pk",
        how="inner",
        validate="many_to_one",
    )

    batting_data["batting_team_id"] = batting_data[
        "home_team_id"
    ]

    top_inning = batting_data["inning_topbot"] == "Top"

    batting_data.loc[
        top_inning,
        "batting_team_id",
    ] = batting_data.loc[
        top_inning,
        "away_team_id",
    ]

    batting_data["game_date"] = pd.to_datetime(
        batting_data["game_date"]
    )

    plate_appearances = (
        batting_data[
            [
                "game_pk",
                "game_date",
                "batting_team_id",
                "at_bat_number",
            ]
        ]
        .drop_duplicates()
        .groupby(
            [
                "batting_team_id",
                "game_date",
            ]
        )
        .size()
        .rename("plate_appearances")
        .reset_index()
    )

    strikeout_events = {
        "strikeout",
        "strikeout_double_play",
    }

    strikeouts = (
        batting_data[
            batting_data["events"].isin(strikeout_events)
        ]
        .groupby(
            [
                "batting_team_id",
                "game_date",
            ]
        )
        .size()
        .rename("strikeouts")
        .reset_index()
    )

    daily_stats = plate_appearances.merge(
        strikeouts,
        on=[
            "batting_team_id",
            "game_date",
        ],
        how="left",
    )

    daily_stats["strikeouts"] = (
        daily_stats["strikeouts"]
        .fillna(0)
        .astype(int)
    )

    daily_stats = daily_stats.sort_values(
        [
            "batting_team_id",
            "game_date",
        ]
    ).reset_index(drop=True)

    grouped = daily_stats.groupby(
        "batting_team_id"
    )

    daily_stats["prior_strikeouts"] = grouped[
        "strikeouts"
    ].transform(
        lambda values: values.cumsum().shift(1)
    )

    daily_stats["prior_plate_appearances"] = grouped[
        "plate_appearances"
    ].transform(
        lambda values: values.cumsum().shift(1)
    )

    daily_stats["team_k_rate"] = (
        daily_stats["prior_strikeouts"]
        / daily_stats["prior_plate_appearances"]
    )

    return daily_stats[
        [
            "batting_team_id",
            "game_date",
            "team_k_rate",
        ]
    ]
def add_opponent_k_rate(
    start_data: pd.DataFrame,
    team_k_history: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the opponent's pregame strikeout rate to each pitcher start."""

    result = start_data.merge(
        team_k_history,
        left_on=[
            "opponent_id",
            "game_date",
        ],
        right_on=[
            "batting_team_id",
            "game_date",
        ],
        how="left",
        validate="many_to_one",
    )

    result = result.rename(
        columns={
            "team_k_rate": "opponent_k_rate",
        }
    )

    result = result.drop(
        columns="batting_team_id"
    )

    return result
if __name__ == "__main__":
    # Run the small controlled test first.
    # This does not require downloading any baseball data.
    test_rolling_features()

    # Leave this commented out until the controlled test passes.
    # Remove the # when you want to test real Statcast data.
    #
    test_real_pitcher_data()
