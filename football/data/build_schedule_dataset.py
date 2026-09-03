"""Build normalized team-game schedule rows for quarterback matchup research."""

import pandas as pd


SOURCE_COLUMNS = [
    "season",
    "week",
    "game_id",
    "game_type",
    "gameday",
    "gametime",
    "home_team",
    "away_team",
]

OUTPUT_COLUMNS = [
    "season",
    "week",
    "game_id",
    "game_date",
    "game_time",
    "team",
    "opponent",
    "home_away",
]

GAME_ID_DUPLICATE_COLUMNS = [
    "season",
    "week",
    "game_id",
    "game_type",
    "gameday",
    "gametime",
    "home_team",
    "away_team",
]

# Deterministic perspective order used after season/week/game_id sorting.
HOME_AWAY_ORDER = ["home", "away"]


def normalize_schedule_dataset(schedules: pd.DataFrame) -> pd.DataFrame:
    """Normalize nflverse one-row-per-game schedules into team-game rows."""

    validate_schedule_schema(schedules)

    schedule_data = schedules.drop_duplicates().loc[:, SOURCE_COLUMNS].copy()
    _validate_duplicate_game_ids(schedule_data)
    schedule_data = schedule_data.drop_duplicates().reset_index(drop=True)

    regular_season_games = schedule_data.loc[
        schedule_data["game_type"] == "REG"
    ].copy()

    if regular_season_games.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    _validate_teams(regular_season_games)

    home_rows = regular_season_games.assign(
        game_date=regular_season_games["gameday"],
        game_time=regular_season_games["gametime"],
        team=regular_season_games["home_team"],
        opponent=regular_season_games["away_team"],
        home_away="home",
    )
    away_rows = regular_season_games.assign(
        game_date=regular_season_games["gameday"],
        game_time=regular_season_games["gametime"],
        team=regular_season_games["away_team"],
        opponent=regular_season_games["home_team"],
        home_away="away",
    )

    team_games = pd.concat(
        [home_rows.loc[:, OUTPUT_COLUMNS], away_rows.loc[:, OUTPUT_COLUMNS]],
        ignore_index=True,
    )

    team_games["home_away"] = pd.Categorical(
        team_games["home_away"],
        categories=HOME_AWAY_ORDER,
        ordered=True,
    )

    team_games = team_games.sort_values(
        by=["season", "week", "game_id", "home_away"],
        kind="mergesort",
    ).reset_index(drop=True)
    team_games["home_away"] = team_games["home_away"].astype(str)

    return team_games.loc[:, OUTPUT_COLUMNS]


def validate_schedule_schema(schedules: pd.DataFrame) -> None:
    """Validate nflverse schedule columns required for normalization."""

    missing_columns = sorted(set(SOURCE_COLUMNS).difference(schedules.columns))

    if missing_columns:
        raise ValueError(
            "NFL schedule source data is missing required columns: "
            f"{missing_columns}"
        )


def _validate_duplicate_game_ids(schedule_data: pd.DataFrame) -> None:
    """Reject duplicate game IDs that disagree on required schedule fields."""

    duplicate_game_ids = schedule_data.loc[
        schedule_data.duplicated(subset=["game_id"], keep=False),
        GAME_ID_DUPLICATE_COLUMNS,
    ]

    if duplicate_game_ids.empty:
        return

    conflicting_game_ids = []
    for game_id, game_rows in duplicate_game_ids.groupby("game_id", sort=True):
        unique_schedule_rows = game_rows.drop_duplicates()
        if len(unique_schedule_rows) > 1:
            conflicting_game_ids.append(game_id)

    if conflicting_game_ids:
        raise ValueError(
            "Conflicting schedule records found for game_id values: "
            f"{conflicting_game_ids}"
        )


def _validate_teams(schedule_data: pd.DataFrame) -> None:
    """Validate each regular-season game has two distinct teams."""

    missing_team_rows = schedule_data.loc[
        _is_missing_team(schedule_data["home_team"])
        | _is_missing_team(schedule_data["away_team"])
    ]

    if not missing_team_rows.empty:
        game_ids = sorted(missing_team_rows["game_id"].dropna().unique())
        raise ValueError(
            "NFL schedule source data has missing home_team or away_team "
            f"for game_id values: {game_ids}"
        )

    same_team_rows = schedule_data.loc[
        schedule_data["home_team"] == schedule_data["away_team"]
    ]

    if not same_team_rows.empty:
        game_ids = sorted(same_team_rows["game_id"].dropna().unique())
        raise ValueError(
            "NFL schedule source data has the same home_team and away_team "
            f"for game_id values: {game_ids}"
        )


def _is_missing_team(team_values: pd.Series) -> pd.Series:
    """Return rows where a team value is null or blank."""

    return team_values.isna() | (team_values.astype(str).str.strip() == "")
