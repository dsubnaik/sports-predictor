"""Identify the primary quarterback for each completed team-game."""

from numbers import Real

import pandas as pd

from football.data.build_quarterback_dataset import OUTPUT_COLUMNS


TEAM_GAME_KEYS = [
    "season",
    "week",
    "game_id",
    "team",
]

PRIMARY_QUARTERBACK_COLUMNS = [
    *OUTPUT_COLUMNS,
    "primary_attempts",
    "secondary_attempts",
    "quarterbacks_with_attempts",
    "low_attempt_primary_qb",
    "similar_attempt_split",
    "exact_attempt_tie",
]


def identify_primary_quarterbacks(
    data: pd.DataFrame,
    low_attempt_threshold: int = 15,
    similar_attempt_ratio: float = 0.70,
) -> pd.DataFrame:
    """Return one primary-quarterback row per team-game."""

    _validate_inputs(
        data=data,
        low_attempt_threshold=low_attempt_threshold,
        similar_attempt_ratio=similar_attempt_ratio,
    )

    quarterback_data = data.loc[:, OUTPUT_COLUMNS].copy()
    quarterback_data["_player_id_sort"] = quarterback_data["player_id"].astype(str)

    ranked = quarterback_data.sort_values(
        by=[
            "season",
            "week",
            "game_id",
            "team",
            "passing_attempts",
            "_player_id_sort",
        ],
        ascending=[True, True, True, True, False, True],
        kind="mergesort",
    )

    selected = ranked.drop_duplicates(
        subset=TEAM_GAME_KEYS,
        keep="first",
    ).copy()

    attempts = quarterback_data.loc[
        quarterback_data["passing_attempts"] > 0,
        [*TEAM_GAME_KEYS, "passing_attempts"],
    ]

    attempt_summary = (
        attempts.sort_values(
            by=[*TEAM_GAME_KEYS, "passing_attempts"],
            ascending=[True, True, True, True, False],
            kind="mergesort",
        )
        .groupby(TEAM_GAME_KEYS, sort=False)
        .agg(
            quarterbacks_with_attempts=("passing_attempts", "size"),
            secondary_attempts=("passing_attempts", _secondary_attempts),
        )
        .reset_index()
    )

    max_attempts = quarterback_data.groupby(TEAM_GAME_KEYS, sort=False)[
        "passing_attempts"
    ].transform("max")
    exact_ties = (
        quarterback_data.loc[
            quarterback_data["passing_attempts"] == max_attempts,
            TEAM_GAME_KEYS,
        ]
        .groupby(TEAM_GAME_KEYS, sort=False)
        .size()
        .rename("top_attempt_count")
        .reset_index()
    )

    selected["primary_attempts"] = selected["passing_attempts"]
    selected = selected.merge(
        attempt_summary,
        on=TEAM_GAME_KEYS,
        how="left",
    )
    selected = selected.merge(
        exact_ties,
        on=TEAM_GAME_KEYS,
        how="left",
    )

    selected["quarterbacks_with_attempts"] = (
        selected["quarterbacks_with_attempts"].fillna(0).astype(int)
    )
    selected["secondary_attempts"] = selected["secondary_attempts"].fillna(0)
    selected["top_attempt_count"] = selected["top_attempt_count"].fillna(0).astype(int)
    selected["low_attempt_primary_qb"] = (
        selected["primary_attempts"] < low_attempt_threshold
    )
    selected["similar_attempt_split"] = (
        selected["primary_attempts"] > 0
    ) & (
        selected["secondary_attempts"]
        >= similar_attempt_ratio * selected["primary_attempts"]
    )
    selected["exact_attempt_tie"] = selected["top_attempt_count"] > 1

    selected = selected.drop(columns=["_player_id_sort", "top_attempt_count"])

    return selected.sort_values(
        by=TEAM_GAME_KEYS,
        kind="mergesort",
    ).reset_index(drop=True)[PRIMARY_QUARTERBACK_COLUMNS]


def _secondary_attempts(attempts: pd.Series) -> int | float:
    """Return the second-highest positive passing-attempt total."""

    if len(attempts) < 2:
        return 0

    return attempts.iloc[1]


def _validate_inputs(
    data: pd.DataFrame,
    low_attempt_threshold: int,
    similar_attempt_ratio: float,
) -> None:
    missing_columns = set(OUTPUT_COLUMNS).difference(data.columns)

    if missing_columns:
        raise ValueError(
            "Quarterback data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if isinstance(low_attempt_threshold, bool) or not isinstance(
        low_attempt_threshold, Real
    ):
        raise ValueError("low_attempt_threshold must be a numeric value")

    if low_attempt_threshold < 0:
        raise ValueError("low_attempt_threshold must be greater than or equal to 0")

    if isinstance(similar_attempt_ratio, bool) or not isinstance(
        similar_attempt_ratio, Real
    ):
        raise ValueError("similar_attempt_ratio must be a numeric value")

    if not 0 <= similar_attempt_ratio <= 1:
        raise ValueError("similar_attempt_ratio must be between 0 and 1")
