"""Build defense-game logs from primary opposing quarterback performances."""

import pandas as pd

from football.features.primary_quarterbacks import PRIMARY_QUARTERBACK_COLUMNS


DEFENSE_GAME_KEYS = [
    "season",
    "week",
    "game_id",
    "defense",
]

DEFENSE_VS_QUARTERBACK_COLUMNS = [
    "season",
    "week",
    "game_id",
    "defense",
    "offense_team",
    "opposing_qb_id",
    "opposing_qb_name",
    "passing_attempts_allowed",
    "completions_allowed",
    "passing_yards_allowed",
    "passing_touchdowns_allowed",
    "opposing_interceptions",
    "low_attempt_primary_qb",
    "similar_attempt_split",
    "exact_attempt_tie",
    "quarterbacks_with_attempts",
]


COLUMN_MAP = {
    "opponent": "defense",
    "team": "offense_team",
    "player_id": "opposing_qb_id",
    "player_name": "opposing_qb_name",
    "passing_attempts": "passing_attempts_allowed",
    "completions": "completions_allowed",
    "passing_yards": "passing_yards_allowed",
    "passing_touchdowns": "passing_touchdowns_allowed",
    "interceptions": "opposing_interceptions",
}


def build_defense_vs_primary_quarterback_logs(data: pd.DataFrame) -> pd.DataFrame:
    """Return one defense-game row from each primary opposing quarterback row."""

    missing_columns = set(PRIMARY_QUARTERBACK_COLUMNS).difference(data.columns)

    if missing_columns:
        raise ValueError(
            "Primary quarterback data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    defense_logs = (
        data.loc[:, PRIMARY_QUARTERBACK_COLUMNS]
        .copy()
        .rename(columns=COLUMN_MAP)
        .loc[:, DEFENSE_VS_QUARTERBACK_COLUMNS]
        .drop_duplicates()
    )

    conflicting_duplicates = defense_logs[
        defense_logs.duplicated(
            subset=DEFENSE_GAME_KEYS,
            keep=False,
        )
    ]

    if not conflicting_duplicates.empty:
        conflicting_keys = (
            conflicting_duplicates.loc[:, DEFENSE_GAME_KEYS]
            .drop_duplicates()
            .sort_values(
                by=DEFENSE_GAME_KEYS,
                kind="mergesort",
            )
            .to_dict(orient="records")
        )

        raise ValueError(
            "Conflicting defense-game records found for keys: "
            f"{conflicting_keys}"
        )

    return defense_logs.sort_values(
        by=DEFENSE_GAME_KEYS,
        kind="mergesort",
    ).reset_index(drop=True)
