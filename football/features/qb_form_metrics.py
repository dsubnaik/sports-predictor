"""Leakage-safe quarterback passing form summaries."""

from __future__ import annotations

from numbers import Integral

import pandas as pd

from football.features.primary_quarterbacks import PRIMARY_QUARTERBACK_COLUMNS


REQUIRED_COLUMNS = [
    "season",
    "week",
    "game_id",
    "player_id",
    "player_name",
    "team",
    "passing_attempts",
    "passing_yards",
    "primary_attempts",
    "low_attempt_primary_qb",
    "similar_attempt_split",
    "exact_attempt_tie",
]

OUTPUT_COLUMNS = [
    "season",
    "report_week",
    "player_id",
    "player_name",
    "latest_team",
    "qb_season_avg",
    "qb_last3_avg",
    "qb_season_attempts_avg",
    "qb_last3_attempts_avg",
    "qb_season_games",
    "qb_last3_games",
    "flagged_games",
]

GAME_KEYS = [
    "season",
    "week",
    "game_id",
    "player_id",
]

FLAG_COLUMNS = [
    "low_attempt_primary_qb",
    "similar_attempt_split",
    "exact_attempt_tie",
]

CONFLICT_CHECK_COLUMNS = [
    *GAME_KEYS,
    "player_name",
    "team",
    "passing_attempts",
    "passing_yards",
    "primary_attempts",
    *FLAG_COLUMNS,
]


def build_qb_form_metrics(
    primary_qbs: pd.DataFrame,
    report_season: int,
    report_week: int,
) -> pd.DataFrame:
    """Return prior-game passing form metrics for each quarterback."""

    _validate_report_value(report_season, "report_season")
    _validate_report_value(report_week, "report_week")
    _validate_required_columns(primary_qbs)

    prior_games = primary_qbs.loc[
        (primary_qbs["season"] == report_season)
        & (primary_qbs["week"] < report_week),
        REQUIRED_COLUMNS,
    ].copy()

    if prior_games.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    prior_games = _validated_unique_games(prior_games)

    prior_games = prior_games.sort_values(
        by=GAME_KEYS,
        kind="mergesort",
    )

    season_summary = (
        prior_games.groupby("player_id", sort=False)
        .agg(
            qb_season_avg=("passing_yards", "mean"),
            qb_season_attempts_avg=("primary_attempts", "mean"),
            qb_season_games=("game_id", "nunique"),
        )
        .reset_index()
    )

    last3_games = prior_games.groupby("player_id", group_keys=False).tail(3)
    last3_summary = (
        last3_games.groupby("player_id", sort=False)
        .agg(
            qb_last3_avg=("passing_yards", "mean"),
            qb_last3_attempts_avg=("primary_attempts", "mean"),
            qb_last3_games=("game_id", "nunique"),
        )
        .reset_index()
    )

    latest_games = (
        prior_games.groupby("player_id", group_keys=False)
        .tail(1)
        .loc[:, ["player_id", "player_name", "team"]]
        .rename(columns={"team": "latest_team"})
    )
    flagged_summary = _count_flagged_games(prior_games)

    result = (
        season_summary.merge(last3_summary, on="player_id", how="left")
        .merge(latest_games, on="player_id", how="left")
        .merge(flagged_summary, on="player_id", how="left")
    )
    result["season"] = report_season
    result["report_week"] = report_week
    result["flagged_games"] = result["flagged_games"].fillna(0).astype(int)

    return result.loc[:, OUTPUT_COLUMNS].sort_values(
        by=["player_name", "player_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def _count_flagged_games(prior_games: pd.DataFrame) -> pd.DataFrame:
    """Count unique prior quarterback-games with any quality flag set."""

    flagged_games = prior_games.loc[:, GAME_KEYS].copy()
    flagged_games["is_flagged"] = prior_games.loc[:, FLAG_COLUMNS].any(axis=1)

    return (
        flagged_games.groupby(GAME_KEYS, as_index=False, sort=False)["is_flagged"]
        .max()
        .groupby("player_id", sort=False)
        .agg(flagged_games=("is_flagged", "sum"))
        .reset_index()
    )


def _validated_unique_games(prior_games: pd.DataFrame) -> pd.DataFrame:
    """Return unique QB-games or raise on conflicting duplicate keys."""

    unique_games = prior_games.drop_duplicates(subset=CONFLICT_CHECK_COLUMNS).copy()
    conflicts = unique_games.loc[
        unique_games.duplicated(subset=GAME_KEYS, keep=False),
        GAME_KEYS,
    ]

    if not conflicts.empty:
        conflicting_keys = (
            conflicts.drop_duplicates()
            .sort_values(by=GAME_KEYS, kind="mergesort")
            .to_dict(orient="records")
        )

        raise ValueError(
            "Conflicting primary quarterback-game records found for keys: "
            f"{conflicting_keys}"
        )

    return unique_games


def _validate_required_columns(data: pd.DataFrame) -> None:
    """Raise a clear error when required primary-QB columns are absent."""

    required_columns = set(REQUIRED_COLUMNS).union(PRIMARY_QUARTERBACK_COLUMNS)
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(
            "Primary quarterback form data is missing required columns: "
            f"{sorted(missing_columns)}"
        )


def _validate_report_value(value: int, name: str) -> None:
    """Require positive integer report season and week values."""

    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be an integer greater than or equal to 1")
