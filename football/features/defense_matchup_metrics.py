"""Leakage-safe defensive passing matchup summaries."""

from __future__ import annotations

from numbers import Integral

import pandas as pd


REQUIRED_COLUMNS = [
    "season",
    "week",
    "game_id",
    "defense",
    "passing_yards_allowed",
    "low_attempt_primary_qb",
    "similar_attempt_split",
    "exact_attempt_tie",
]

OUTPUT_COLUMNS = [
    "season",
    "report_week",
    "defense",
    "defense_season_avg_allowed",
    "defense_last3_avg_allowed",
    "defense_season_games",
    "defense_last3_games",
    "flagged_games",
    "matchup_rank",
]

GAME_KEYS = [
    "season",
    "week",
    "game_id",
    "defense",
]

FLAG_COLUMNS = [
    "low_attempt_primary_qb",
    "similar_attempt_split",
    "exact_attempt_tie",
]


def build_defense_matchup_metrics(
    defense_logs: pd.DataFrame,
    report_season: int,
    report_week: int,
) -> pd.DataFrame:
    """Return prior-game passing matchup metrics for each defense.

    Rankings use pandas ``rank(method="min", ascending=False)`` so tied season
    averages receive the same best rank and the next rank leaves a gap.
    """

    _validate_report_value(report_season, "report_season")
    _validate_report_value(report_week, "report_week")
    _validate_required_columns(defense_logs)

    prior_games = defense_logs.loc[
        (defense_logs["season"] == report_season)
        & (defense_logs["week"] < report_week),
        REQUIRED_COLUMNS,
    ].copy()

    if prior_games.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    prior_games = prior_games.sort_values(
        by=GAME_KEYS,
        kind="mergesort",
    )

    season_summary = (
        prior_games.groupby("defense", sort=False)
        .agg(
            defense_season_avg_allowed=(
                "passing_yards_allowed",
                "mean",
            ),
            defense_season_games=("game_id", "nunique"),
        )
        .reset_index()
    )

    last3_games = prior_games.groupby("defense", group_keys=False).tail(3)
    last3_summary = (
        last3_games.groupby("defense", sort=False)
        .agg(
            defense_last3_avg_allowed=(
                "passing_yards_allowed",
                "mean",
            ),
            defense_last3_games=("game_id", "nunique"),
        )
        .reset_index()
    )

    flagged_summary = _count_flagged_games(prior_games)

    result = (
        season_summary.merge(last3_summary, on="defense", how="left")
        .merge(flagged_summary, on="defense", how="left")
    )
    result["season"] = report_season
    result["report_week"] = report_week
    result["flagged_games"] = result["flagged_games"].fillna(0).astype(int)
    result["matchup_rank"] = (
        result["defense_season_avg_allowed"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    return result.loc[:, OUTPUT_COLUMNS].sort_values(
        by=["matchup_rank", "defense"],
        kind="mergesort",
    ).reset_index(drop=True)


def _count_flagged_games(prior_games: pd.DataFrame) -> pd.DataFrame:
    """Count unique prior defense-games with any quality flag set."""

    flagged_games = prior_games.loc[:, GAME_KEYS].copy()
    flagged_games["is_flagged"] = prior_games.loc[:, FLAG_COLUMNS].any(axis=1)

    return (
        flagged_games.groupby(GAME_KEYS, as_index=False, sort=False)[
            "is_flagged"
        ]
        .max()
        .groupby("defense", sort=False)
        .agg(flagged_games=("is_flagged", "sum"))
        .reset_index()
    )


def _validate_required_columns(data: pd.DataFrame) -> None:
    """Raise a clear error when required defense-log columns are absent."""

    missing_columns = set(REQUIRED_COLUMNS).difference(data.columns)

    if missing_columns:
        raise ValueError(
            "Defense matchup data is missing required columns: "
            f"{sorted(missing_columns)}"
        )


def _validate_report_value(value: int, name: str) -> None:
    """Require positive integer report season and week values."""

    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be an integer greater than or equal to 1")
