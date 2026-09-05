"""Build normalized running-back game rows from nflverse player stats."""

import pandas as pd


SOURCE_COLUMNS = [
    "season",
    "week",
    "season_type",
    "game_id",
    "player_id",
    "player_display_name",
    "position",
    "team",
    "opponent_team",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "targets",
    "receiving_yards",
    "receiving_tds",
]

SOURCE_COLUMN_MAPPING = {
    "season": "season",
    "week": "week",
    "game_id": "game_id",
    "player_id": "player_id",
    "player_display_name": "player_name",
    "team": "team",
    "opponent_team": "opponent",
    "position": "position",
    "carries": "rushing_attempts",
    "rushing_yards": "rushing_yards",
    "rushing_tds": "rushing_touchdowns",
    "receptions": "receptions",
    "targets": "targets",
    "receiving_yards": "receiving_yards",
    "receiving_tds": "receiving_touchdowns",
}

OUTPUT_COLUMNS = [
    "season",
    "week",
    "game_id",
    "player_id",
    "player_name",
    "team",
    "opponent",
    "position",
    "rushing_attempts",
    "rushing_yards",
    "rushing_touchdowns",
    "receptions",
    "targets",
    "receiving_yards",
    "receiving_touchdowns",
]

RUNNING_BACK_GAME_KEYS = [
    "season",
    "week",
    "game_id",
    "player_id",
]


def build_running_back_dataset(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Return one normalized regular-season row per running back per game.

    The input is expected to be nflverse player game stats, such as the
    DataFrame returned by ``football.data.fetch_nflverse.load_player_game_stats``.
    Only rows with ``position == "RB"`` and ``season_type == "REG"`` are retained.
    """

    validate_player_stats_schema(player_stats)

    running_back_rows = player_stats.loc[
        (player_stats["position"] == "RB")
        & (player_stats["season_type"] == "REG"),
        SOURCE_COLUMN_MAPPING.keys(),
    ].rename(columns=SOURCE_COLUMN_MAPPING)

    if running_back_rows.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    running_back_data = running_back_rows.loc[:, OUTPUT_COLUMNS].copy()
    running_back_data = running_back_data.drop_duplicates()

    conflicting_duplicates = running_back_data[
        running_back_data.duplicated(
            subset=RUNNING_BACK_GAME_KEYS,
            keep=False,
        )
    ]

    if not conflicting_duplicates.empty:
        conflicting_keys = (
            conflicting_duplicates.loc[
                :,
                RUNNING_BACK_GAME_KEYS,
            ]
            .drop_duplicates()
            .sort_values(
                by=RUNNING_BACK_GAME_KEYS,
                kind="mergesort",
            )
            .to_dict(orient="records")
        )

        raise ValueError(
            "Conflicting running-back game records found for keys: "
            f"{conflicting_keys}"
        )

    return running_back_data.sort_values(
        by=[
            "season",
            "week",
            "game_id",
            "team",
            "player_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)


def validate_player_stats_schema(player_stats: pd.DataFrame) -> None:
    """Validate nflverse player-stat columns required for RB normalization."""

    missing_columns = set(SOURCE_COLUMNS).difference(player_stats.columns)

    if missing_columns:
        raise ValueError(
            "NFL player stats source data is missing required columns: "
            f"{sorted(missing_columns)}"
        )
