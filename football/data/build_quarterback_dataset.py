"""Build quarterback-game rows for passing-yards prediction."""

import pandas as pd


OUTPUT_COLUMNS = [
    "season",
    "week",
    "game_id",
    "player_id",
    "player_name",
    "team",
    "opponent",
    "passing_attempts",
    "completions",
    "passing_yards",
    "passing_touchdowns",
    "interceptions",
]

QUARTERBACK_GAME_KEYS = [
    "season",
    "week",
    "game_id",
    "player_id",
]


def build_quarterback_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Return a dataset with one row per quarterback-game."""

    missing_columns = set(OUTPUT_COLUMNS).difference(data.columns)

    if missing_columns:
        raise ValueError(
            "Quarterback data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    quarterback_data = data.loc[:, OUTPUT_COLUMNS].copy()
    quarterback_data = quarterback_data.drop_duplicates()

    conflicting_duplicates = quarterback_data[
        quarterback_data.duplicated(
            subset=QUARTERBACK_GAME_KEYS,
            keep=False,
        )
    ]

    if not conflicting_duplicates.empty:
        conflicting_keys = (
            conflicting_duplicates.loc[
                :,
                QUARTERBACK_GAME_KEYS,
            ]
            .drop_duplicates()
            .sort_values(
                by=QUARTERBACK_GAME_KEYS,
                kind="mergesort",
            )
            .to_dict(orient="records")
        )

        raise ValueError(
            "Conflicting quarterback-game records found for keys: "
            f"{conflicting_keys}"
        )

    quarterback_data = quarterback_data.sort_values(
        by=[
            "season",
            "week",
            "game_id",
            "player_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    return quarterback_data
