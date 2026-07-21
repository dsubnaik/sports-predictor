"""
File: data/fetch_statcast.py

Purpose:
    Downloads pitch-level Statcast data for MLB pitchers and converts
    that data into one row per pitching appearance.

Main responsibilities:
    - Download every pitch thrown by a selected pitcher.
    - Look up a pitcher's MLB player ID by name.
    - Aggregate pitch-level data into game-level statistics.
    - Preserve the pitcher ID for multi-pitcher feature engineering.
    - Calculate strikeouts, swinging-strike rate, velocity, and spin rate.

Important note:
    The current aggregation creates one row per pitching appearance.
    It does not yet guarantee that every appearance was a start.
    Filtering out relief appearances will be added next.
"""

import pandas as pd
from pybaseball import playerid_lookup, statcast_pitcher


def fetch_pitcher_statcast(player_id, start_date, end_date):
    """
    Download pitch-level Statcast data for one pitcher.

    Parameters:
        player_id:
            MLB's internal player ID.

            Example:
                Sonny Gray's MLB ID is 543243.

        start_date:
            Beginning of the requested date range.
            Must use the format YYYY-MM-DD.

        end_date:
            End of the requested date range.
            Must use the format YYYY-MM-DD.

    Returns:
        A pandas DataFrame where each row represents one pitch.
    """

    # Download every recorded pitch thrown by the pitcher
    # between the selected start and end dates.
    df = statcast_pitcher(
        start_date,
        end_date,
        player_id,
    )

    return df


def aggregate_to_starts(df):
    """
    Convert pitch-level Statcast data into one row per game.

    Each row contains summary statistics for one pitcher's
    appearance in a game.

    Required input columns:
        game_pk
        game_date
        pitcher
        player_name
        events
        description
        release_speed
        release_spin_rate

    Returned columns:
        game_pk
        game_date
        pitcher_id
        pitcher_name
        strikeouts
        swinging_strikes
        total_pitches
        avg_velocity
        avg_spin_rate
        swstr_pct
    """

    # Stop early when no Statcast data was returned.
    if df.empty:
        raise ValueError(
            "Cannot aggregate Statcast data because the DataFrame is empty."
        )

    # Identify games in which the pitcher appeared during the first inning.
    #
    # A normal starting pitcher begins the game in inning 1.
    # Relief appearances beginning in later innings will be excluded.
    starter_game_ids = (
        df.loc[df["inning"] == 1, "game_pk"]
        .dropna()
        .unique()
    )

    # Keep only pitches from games identified as starts.
    df = df[df["game_pk"].isin(starter_game_ids)].copy()

    # Stop with a clear error if no starts were found.
    if df.empty:
        raise ValueError(
            "No starting-pitcher appearances were found in the selected date range."
        )
    # Create one summary row for each game.
    #
    # game_pk is MLB's unique ID for a game.
    # Grouping by game_pk combines every pitch from the same game.
    agg = df.groupby("game_pk").agg(
        # Keep the date of the game.
        game_date=("game_date", "first"),

        # Keep the pitcher's MLB player ID.
        #
        # This is required when combining several pitchers because
        # rolling features must be calculated separately for each one.
        pitcher_id=("pitcher", "first"),

        # Keep the pitcher's displayed name.
        pitcher_name=("player_name", "first"),

        # Count strikeouts.
        #
        # The events column is normally only populated on the pitch
        # that ends a plate appearance.
        strikeouts=(
            "events",
            lambda values: (values == "strikeout").sum(),
        ),

        # Count pitches recorded as swinging strikes.
        #
        # swinging_strike:
        #     The batter swung and missed normally.
        #
        # swinging_strike_blocked:
        #     The batter swung and missed, but the catcher blocked
        #     the pitch in the dirt.
        swinging_strikes=(
            "description",
            lambda values: values.isin(
                [
                    "swinging_strike",
                    "swinging_strike_blocked",
                ]
            ).sum(),
        ),

        # Count the number of pitches with a recorded release speed.
        #
        # Most Statcast pitches include release_speed, so this acts
        # as the pitch count for the appearance.
        total_pitches=("release_speed", "count"),

        # Calculate the average velocity of all recorded pitches.
        avg_velocity=("release_speed", "mean"),

        # Calculate the average spin rate of all recorded pitches.
        avg_spin_rate=("release_spin_rate", "mean"),
    ).reset_index()

    # Swinging-strike rate is the number of swinging strikes
    # divided by the total number of pitches.
    agg["swstr_pct"] = (
        agg["swinging_strikes"] / agg["total_pitches"]
    )

    # Convert game_date into pandas datetime values.
    # This allows correct chronological sorting later.
    agg["game_date"] = pd.to_datetime(agg["game_date"])

    # Sort the appearances from oldest to newest.
    agg = agg.sort_values("game_date").reset_index(drop=True)

    return agg


def get_player_id(player_name):
    """
    Look up a player's MLB ID using their full name.

    Parameters:
        player_name:
            Player name written as:
                FirstName LastName

            Examples:
                Aaron Nola
                Sonny Gray
                Luis Garcia Jr

    Returns:
        The MLBAM player ID from the first matching result.
    """

    # Split the name only at the first space.
    #
    # Example:
    #     "Luis Garcia Jr"
    #
    # Becomes:
    #     first = "Luis"
    #     last = "Garcia Jr"
    name_parts = player_name.strip().split(" ", 1)

    # A full name must contain at least a first and last name.
    if len(name_parts) != 2:
        raise ValueError(
            "Player name must include both a first and last name."
        )

    first_name = name_parts[0]
    last_name = name_parts[1]

    # Search pybaseball's player database.
    result = playerid_lookup(
        last_name,
        first_name,
    )

    # Raise a clear error when no matching player is found.
    if result.empty:
        raise ValueError(
            f"No MLB player was found with the name '{player_name}'."
        )

    # Return the MLBAM ID from the first matching result.
    return int(result["key_mlbam"].iloc[0])


if __name__ == "__main__":
    """
    Run a simple lookup test when this file is executed directly.

    Command:
        python data/fetch_statcast.py
    """

    test_player_name = "Aaron Nola"

    player_id = get_player_id(test_player_name)

    print(f"{test_player_name}'s MLB player ID is {player_id}.")