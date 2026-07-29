"""Fetch pitcher Statcast data and aggregate it into start-level rows."""

import pandas as pd
from pybaseball import playerid_lookup, statcast_pitcher


def fetch_pitcher_statcast(player_id, start_date, end_date):
    """Download pitch-level Statcast data for one pitcher and date range."""

    df = statcast_pitcher(
        start_date,
        end_date,
        player_id,
    )

    return df


def aggregate_to_starts(df):
    """Convert official-starter pitch rows into one row per pitcher start.

    The input should already be limited to official starters. Grouping by both
    game_pk and pitcher keeps the two starters in the same game separate.
    """

    if df.empty:
        raise ValueError(
            "Cannot aggregate Statcast data because the DataFrame is empty."
        )

    agg = df.groupby(["game_pk", "pitcher"]).agg(
        game_date=("game_date", "first"),
        pitcher_name=("player_name", "first"),
        # Statcast records strikeout events on the pitch ending the plate
        # appearance, not on every pitch in the at-bat.
        strikeouts=(
            "events",
            lambda values: (values == "strikeout").sum(),
        ),
        # Blocked swinging strikes still reflect a whiff and belong in
        # swinging-strike rate.
        swinging_strikes=(
            "description",
            lambda values: values.isin(
                [
                    "swinging_strike",
                    "swinging_strike_blocked",
                ]
            ).sum(),
        ),
        total_pitches=("release_speed", "count"),
        avg_velocity=("release_speed", "mean"),
        avg_spin_rate=("release_spin_rate", "mean"),
    ).reset_index()

    agg = agg.rename(columns={"pitcher": "pitcher_id"})
    agg["swstr_pct"] = (
        agg["swinging_strikes"] / agg["total_pitches"]
    )

    # Later rolling features depend on starts being chronologically ordered.
    agg["game_date"] = pd.to_datetime(agg["game_date"])
    agg = agg.sort_values("game_date").reset_index(drop=True)

    return agg


def get_player_id(player_name):
    """Look up the MLBAM player ID for a full player name."""

    # Split once so suffixes such as "Jr" remain part of the last-name lookup.
    name_parts = player_name.strip().split(" ", 1)

    if len(name_parts) != 2:
        raise ValueError(
            "Player name must include both a first and last name."
        )

    first_name = name_parts[0]
    last_name = name_parts[1]

    result = playerid_lookup(
        last_name,
        first_name,
    )

    if result.empty:
        raise ValueError(
            f"No MLB player was found with the name '{player_name}'."
        )

    return int(result["key_mlbam"].iloc[0])


if __name__ == "__main__":
    test_player_name = "Aaron Nola"
    player_id = get_player_id(test_player_name)
    print(f"{test_player_name}'s MLB player ID is {player_id}.")
