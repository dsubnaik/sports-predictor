"""Leakage-safe running-back form summaries.

This module summarizes completed historical RB games after applying a strict
history-season and prior-game cutoff. Same-game usage descriptors from
``running_back_usage`` are safe here only because report-week and future rows
are removed before any averages, sample sizes, last-three windows, or latest
teams are calculated.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd

from football.features.running_back_usage import RUNNING_BACK_USAGE_COLUMNS


REQUIRED_COLUMNS = [
    "season",
    "week",
    "game_id",
    "player_id",
    "player_name",
    "team",
    "rushing_attempts",
    "rushing_yards",
    "receptions",
    "targets",
    "receiving_yards",
    "opportunities",
    "carry_share",
    "target_share",
    "opportunity_share",
]

OUTPUT_COLUMNS = [
    "report_season",
    "report_week",
    "historical_season",
    "player_id",
    "player_name",
    "latest_team",
    "rb_season_rushing_yards_avg",
    "rb_last3_rushing_yards_avg",
    "rb_season_rushing_attempts_avg",
    "rb_last3_rushing_attempts_avg",
    "rb_season_receptions_avg",
    "rb_last3_receptions_avg",
    "rb_season_targets_avg",
    "rb_last3_targets_avg",
    "rb_season_receiving_yards_avg",
    "rb_last3_receiving_yards_avg",
    "rb_season_opportunities_avg",
    "rb_last3_opportunities_avg",
    "rb_season_carry_share_avg",
    "rb_last3_carry_share_avg",
    "rb_season_target_share_avg",
    "rb_last3_target_share_avg",
    "rb_season_opportunity_share_avg",
    "rb_last3_opportunity_share_avg",
    "rb_season_yards_per_carry",
    "rb_last3_yards_per_carry",
    "rb_season_games",
    "rb_last3_games",
]

GAME_KEYS = [
    "season",
    "week",
    "game_id",
    "player_id",
]

CONFLICT_CHECK_COLUMNS = RUNNING_BACK_USAGE_COLUMNS

NUMERIC_COLUMNS = [
    "rushing_attempts",
    "rushing_yards",
    "receptions",
    "targets",
    "receiving_yards",
    "opportunities",
    "carry_share",
    "target_share",
    "opportunity_share",
]

NON_NEGATIVE_COLUMNS = [
    "rushing_attempts",
    "receptions",
    "targets",
    "opportunities",
    "carry_share",
    "target_share",
    "opportunity_share",
]


def build_rb_form_metrics(
    annotated_rbs: pd.DataFrame,
    report_season: int,
    report_week: int,
    historical_season: int | None = None,
) -> pd.DataFrame:
    """Return prior-game RB form metrics for each historical running back.

    ``annotated_rbs`` must come from ``annotate_running_back_usage``. For
    current-season history, only weeks strictly before ``report_week`` are used.
    For an earlier explicit historical season, the complete selected season is
    used while the output preserves the requested report season and week.
    """

    _validate_report_value(report_season, "report_season")
    _validate_report_value(report_week, "report_week")
    selected_historical_season = _resolve_historical_season(
        report_season,
        historical_season,
    )
    _validate_required_columns(annotated_rbs)

    season_games = annotated_rbs.loc[
        annotated_rbs["season"] == selected_historical_season,
        RUNNING_BACK_USAGE_COLUMNS,
    ].copy()
    if selected_historical_season == report_season:
        season_games = season_games.loc[season_games["week"] < report_week].copy()

    if season_games.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    prior_games = _validated_unique_games(season_games).loc[:, REQUIRED_COLUMNS]
    prior_games = prior_games.sort_values(
        by=GAME_KEYS,
        kind="mergesort",
    )

    season_summary = _summarize_games(prior_games, "rb_season")
    last3_games = prior_games.groupby("player_id", group_keys=False).tail(3)
    last3_summary = _summarize_games(last3_games, "rb_last3")

    latest_games = (
        prior_games.groupby("player_id", group_keys=False)
        .tail(1)
        .loc[:, ["player_id", "player_name", "team"]]
        .rename(columns={"team": "latest_team"})
    )

    result = (
        season_summary.merge(last3_summary, on="player_id", how="left")
        .merge(latest_games, on="player_id", how="left")
    )
    result["report_season"] = report_season
    result["report_week"] = report_week
    result["historical_season"] = selected_historical_season

    return result.loc[:, OUTPUT_COLUMNS].sort_values(
        by=["player_name", "player_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def _summarize_games(games: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Return average workload metrics and weighted yards per carry."""

    summary = (
        games.groupby("player_id", sort=False)
        .agg(
            **{
                f"{prefix}_rushing_yards_avg": ("rushing_yards", "mean"),
                f"{prefix}_rushing_attempts_avg": ("rushing_attempts", "mean"),
                f"{prefix}_receptions_avg": ("receptions", "mean"),
                f"{prefix}_targets_avg": ("targets", "mean"),
                f"{prefix}_receiving_yards_avg": ("receiving_yards", "mean"),
                f"{prefix}_opportunities_avg": ("opportunities", "mean"),
                f"{prefix}_carry_share_avg": ("carry_share", "mean"),
                f"{prefix}_target_share_avg": ("target_share", "mean"),
                f"{prefix}_opportunity_share_avg": (
                    "opportunity_share",
                    "mean",
                ),
                f"{prefix}_rushing_yards_total": ("rushing_yards", "sum"),
                f"{prefix}_rushing_attempts_total": ("rushing_attempts", "sum"),
                f"{prefix}_games": ("game_id", "nunique"),
            }
        )
        .reset_index()
    )

    summary[f"{prefix}_yards_per_carry"] = _safe_yards_per_carry(
        summary[f"{prefix}_rushing_yards_total"],
        summary[f"{prefix}_rushing_attempts_total"],
    )

    return summary.drop(
        columns=[
            f"{prefix}_rushing_yards_total",
            f"{prefix}_rushing_attempts_total",
        ]
    )


def _safe_yards_per_carry(
    rushing_yards: pd.Series,
    rushing_attempts: pd.Series,
) -> pd.Series:
    """Return finite yards per carry, using 0.0 for zero-attempt rows."""

    yards_per_carry = pd.Series(0.0, index=rushing_yards.index)
    valid_attempts = rushing_attempts != 0
    yards_per_carry.loc[valid_attempts] = (
        rushing_yards.loc[valid_attempts] / rushing_attempts.loc[valid_attempts]
    )

    return yards_per_carry


def _validated_unique_games(prior_games: pd.DataFrame) -> pd.DataFrame:
    """Return unique RB-games or raise on conflicting duplicate keys."""

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
            "Conflicting running-back form records found for keys: "
            f"{conflicting_keys}"
        )

    return unique_games


def _resolve_historical_season(
    report_season: int,
    historical_season: int | None,
) -> int:
    """Return the selected historical season or reject invalid overrides."""

    if historical_season is None:
        return report_season

    _validate_report_value(historical_season, "historical_season")
    if historical_season > report_season:
        raise ValueError(
            "historical_season must be less than or equal to report_season"
        )

    return historical_season


def _validate_required_columns(data: pd.DataFrame) -> None:
    """Raise clear errors when annotated RB form input is invalid."""

    required_columns = set(REQUIRED_COLUMNS).union(RUNNING_BACK_USAGE_COLUMNS)
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(
            "Running back form data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if data.empty:
        return

    invalid_numeric_columns = [
        column
        for column in NUMERIC_COLUMNS
        if (
            pd.api.types.is_bool_dtype(data[column])
            or not pd.api.types.is_numeric_dtype(data[column])
        )
    ]

    if invalid_numeric_columns:
        raise ValueError(
            "Running back form columns must be numeric: "
            f"{sorted(invalid_numeric_columns)}"
        )

    missing_values = data.loc[:, NUMERIC_COLUMNS].isna().any()
    missing_value_columns = missing_values[missing_values].index.tolist()

    if missing_value_columns:
        raise ValueError(
            "Running back form columns cannot contain missing values: "
            f"{sorted(missing_value_columns)}"
        )

    finite_values = np.isfinite(
        data.loc[:, NUMERIC_COLUMNS].to_numpy(dtype=float)
    )
    non_finite_columns = data.loc[:, NUMERIC_COLUMNS].columns[
        ~finite_values.all(axis=0)
    ].tolist()

    if non_finite_columns:
        raise ValueError(
            "Running back form columns must contain finite values: "
            f"{sorted(non_finite_columns)}"
        )

    negative_values = (data.loc[:, NON_NEGATIVE_COLUMNS] < 0).any()
    negative_value_columns = negative_values[negative_values].index.tolist()

    if negative_value_columns:
        raise ValueError(
            "Running back form columns cannot contain negative values: "
            f"{sorted(negative_value_columns)}"
        )


def _validate_report_value(value: int, name: str) -> None:
    """Require positive integer season and week values."""

    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be an integer greater than or equal to 1")
