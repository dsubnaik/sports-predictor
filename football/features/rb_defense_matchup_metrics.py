"""Build leakage-safe defense-versus-RB matchup metrics and rankings.

Defense RB game logs are retrospective completed-game results. This module
makes them prediction-safe by selecting the requested historical season and,
for current-season history, filtering to games strictly before the target
report week before calculating averages, recent windows, sample sizes, or
rankings.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd

from football.features.defense_rb_game_logs import (
    DEFENSE_RB_GAME_KEYS,
    DEFENSE_RB_GAME_LOG_COLUMNS,
)


OUTPUT_COLUMNS = [
    "report_season",
    "report_week",
    "historical_season",
    "defense",
    "defense_season_rb_rushing_attempts_avg_allowed",
    "defense_last3_rb_rushing_attempts_avg_allowed",
    "defense_season_rb_rushing_yards_avg_allowed",
    "defense_last3_rb_rushing_yards_avg_allowed",
    "defense_season_rb_rushing_touchdowns_avg_allowed",
    "defense_last3_rb_rushing_touchdowns_avg_allowed",
    "defense_season_rb_receptions_avg_allowed",
    "defense_last3_rb_receptions_avg_allowed",
    "defense_season_rb_targets_avg_allowed",
    "defense_last3_rb_targets_avg_allowed",
    "defense_season_rb_receiving_yards_avg_allowed",
    "defense_last3_rb_receiving_yards_avg_allowed",
    "defense_season_rb_receiving_touchdowns_avg_allowed",
    "defense_last3_rb_receiving_touchdowns_avg_allowed",
    "defense_season_rb_opportunities_avg_allowed",
    "defense_last3_rb_opportunities_avg_allowed",
    "defense_season_rb_players_used_avg",
    "defense_last3_rb_players_used_avg",
    "defense_season_rb_yards_per_carry_allowed",
    "defense_last3_rb_yards_per_carry_allowed",
    "defense_season_games",
    "defense_last3_games",
    "matchup_rank",
]

IDENTITY_COLUMNS = [
    "season",
    "week",
    "game_id",
    "defense",
    "offense_team",
]

TEXT_IDENTITY_COLUMNS = [
    "game_id",
    "defense",
    "offense_team",
]

NUMERIC_IDENTITY_COLUMNS = [
    "season",
    "week",
]

PRODUCTION_COLUMNS = [
    "rb_rushing_attempts_allowed",
    "rb_rushing_yards_allowed",
    "rb_rushing_touchdowns_allowed",
    "rb_receptions_allowed",
    "rb_targets_allowed",
    "rb_receiving_yards_allowed",
    "rb_receiving_touchdowns_allowed",
    "rb_opportunities_allowed",
    "rb_players_used",
]

NON_NEGATIVE_PRODUCTION_COLUMNS = [
    "rb_rushing_attempts_allowed",
    "rb_rushing_touchdowns_allowed",
    "rb_receptions_allowed",
    "rb_targets_allowed",
    "rb_receiving_touchdowns_allowed",
    "rb_opportunities_allowed",
    "rb_players_used",
]

CONFLICT_CHECK_COLUMNS = DEFENSE_RB_GAME_LOG_COLUMNS


def build_rb_defense_matchup_metrics(
    defense_logs: pd.DataFrame,
    report_season: int,
    report_week: int,
    historical_season: int | None = None,
) -> pd.DataFrame:
    """Return leakage-safe RB matchup metrics for each historical defense.

    ``defense_logs`` must come from ``build_defense_rb_game_logs``. When the
    selected historical season equals ``report_season``, only games with a
    week strictly before ``report_week`` are eligible. An explicitly earlier
    historical season uses its complete available history. No Week 1 fallback
    is applied here.

    ``matchup_rank`` orders defenses by season-average RB rushing yards
    allowed, descending. Rank 1 therefore identifies the defense that allowed
    the most RB rushing yards per eligible game; it is a historical matchup
    ranking, not a prediction or an RB talent ranking.
    """

    _validate_report_value(report_season, "report_season")
    _validate_report_value(report_week, "report_week")
    selected_historical_season = _resolve_historical_season(
        report_season,
        historical_season,
    )
    _validate_defense_logs(defense_logs)

    eligible_games = defense_logs.loc[
        defense_logs["season"] == selected_historical_season,
        DEFENSE_RB_GAME_LOG_COLUMNS,
    ].copy()
    if selected_historical_season == report_season:
        eligible_games = eligible_games.loc[
            eligible_games["week"] < report_week
        ].copy()

    if eligible_games.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    eligible_games = _validated_unique_games(eligible_games)
    eligible_games = eligible_games.sort_values(
        by=DEFENSE_RB_GAME_KEYS,
        kind="mergesort",
    )

    season_summary = _summarize_games(eligible_games, "defense_season")
    last3_games = eligible_games.groupby(
        "defense",
        group_keys=False,
    ).tail(3)
    last3_summary = _summarize_games(last3_games, "defense_last3")

    result = season_summary.merge(
        last3_summary,
        on="defense",
        how="left",
    )
    result["report_season"] = report_season
    result["report_week"] = report_week
    result["historical_season"] = selected_historical_season
    result["matchup_rank"] = (
        result["defense_season_rb_rushing_yards_avg_allowed"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    return result.loc[:, OUTPUT_COLUMNS].sort_values(
        by=["matchup_rank", "defense"],
        kind="mergesort",
    ).reset_index(drop=True)


def _summarize_games(games: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Return per-game averages and weighted rushing efficiency."""

    summary = (
        games.groupby("defense", sort=False)
        .agg(
            **{
                f"{prefix}_rb_rushing_attempts_avg_allowed": (
                    "rb_rushing_attempts_allowed",
                    "mean",
                ),
                f"{prefix}_rb_rushing_yards_avg_allowed": (
                    "rb_rushing_yards_allowed",
                    "mean",
                ),
                f"{prefix}_rb_rushing_touchdowns_avg_allowed": (
                    "rb_rushing_touchdowns_allowed",
                    "mean",
                ),
                f"{prefix}_rb_receptions_avg_allowed": (
                    "rb_receptions_allowed",
                    "mean",
                ),
                f"{prefix}_rb_targets_avg_allowed": (
                    "rb_targets_allowed",
                    "mean",
                ),
                f"{prefix}_rb_receiving_yards_avg_allowed": (
                    "rb_receiving_yards_allowed",
                    "mean",
                ),
                f"{prefix}_rb_receiving_touchdowns_avg_allowed": (
                    "rb_receiving_touchdowns_allowed",
                    "mean",
                ),
                f"{prefix}_rb_opportunities_avg_allowed": (
                    "rb_opportunities_allowed",
                    "mean",
                ),
                f"{prefix}_rb_players_used_avg": (
                    "rb_players_used",
                    "mean",
                ),
                f"{prefix}_rb_rushing_yards_total": (
                    "rb_rushing_yards_allowed",
                    "sum",
                ),
                f"{prefix}_rb_rushing_attempts_total": (
                    "rb_rushing_attempts_allowed",
                    "sum",
                ),
                f"{prefix}_games": ("game_id", "nunique"),
            }
        )
        .reset_index()
    )

    summary[f"{prefix}_rb_yards_per_carry_allowed"] = _safe_yards_per_carry(
        summary[f"{prefix}_rb_rushing_yards_total"],
        summary[f"{prefix}_rb_rushing_attempts_total"],
    )

    return summary.drop(
        columns=[
            f"{prefix}_rb_rushing_yards_total",
            f"{prefix}_rb_rushing_attempts_total",
        ]
    )


def _safe_yards_per_carry(
    rushing_yards: pd.Series,
    rushing_attempts: pd.Series,
) -> pd.Series:
    """Return finite yards per carry, using 0.0 for zero attempts."""

    yards_per_carry = pd.Series(0.0, index=rushing_yards.index)
    valid_attempts = rushing_attempts != 0
    yards_per_carry.loc[valid_attempts] = (
        rushing_yards.loc[valid_attempts]
        / rushing_attempts.loc[valid_attempts]
    )
    return yards_per_carry


def _validated_unique_games(games: pd.DataFrame) -> pd.DataFrame:
    """Return exact-unique defense games or reject conflicting keys."""

    unique_games = games.drop_duplicates(
        subset=CONFLICT_CHECK_COLUMNS
    ).copy()
    conflicts = unique_games.loc[
        unique_games.duplicated(subset=DEFENSE_RB_GAME_KEYS, keep=False),
        DEFENSE_RB_GAME_KEYS,
    ]

    if not conflicts.empty:
        conflicting_keys = (
            conflicts.drop_duplicates()
            .sort_values(by=DEFENSE_RB_GAME_KEYS, kind="mergesort")
            .to_dict(orient="records")
        )
        raise ValueError(
            "Conflicting RB defense-game records found for keys: "
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


def _validate_defense_logs(data: pd.DataFrame) -> None:
    """Raise clear errors for invalid defense-versus-RB game logs."""

    missing_columns = set(DEFENSE_RB_GAME_LOG_COLUMNS).difference(data.columns)
    if missing_columns:
        raise ValueError(
            "RB defense matchup data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if data.empty:
        return

    missing_identities = data.loc[:, IDENTITY_COLUMNS].isna().any()
    missing_identity_columns = missing_identities[
        missing_identities
    ].index.tolist()
    if missing_identity_columns:
        raise ValueError(
            "RB defense matchup identity columns cannot contain missing values: "
            f"{sorted(missing_identity_columns)}"
        )

    blank_identities = data.loc[:, TEXT_IDENTITY_COLUMNS].apply(
        lambda values: values.astype("string").str.strip().eq("").any()
    )
    blank_identity_columns = blank_identities[blank_identities].index.tolist()
    if blank_identity_columns:
        raise ValueError(
            "RB defense matchup identity columns cannot contain blank values: "
            f"{sorted(blank_identity_columns)}"
        )

    _validate_numeric_identity_columns(data)
    _validate_production_columns(data)


def _validate_numeric_identity_columns(data: pd.DataFrame) -> None:
    """Require positive, finite whole-number season and week values."""

    invalid_numeric_columns = [
        column
        for column in NUMERIC_IDENTITY_COLUMNS
        if (
            _contains_boolean(data[column])
            or not pd.api.types.is_numeric_dtype(data[column])
        )
    ]
    if invalid_numeric_columns:
        raise ValueError(
            "RB defense matchup season and week columns must be numeric: "
            f"{sorted(invalid_numeric_columns)}"
        )

    identity_values = data.loc[:, NUMERIC_IDENTITY_COLUMNS].to_numpy(dtype=float)
    non_finite_columns = data.loc[:, NUMERIC_IDENTITY_COLUMNS].columns[
        ~np.isfinite(identity_values).all(axis=0)
    ].tolist()
    if non_finite_columns:
        raise ValueError(
            "RB defense matchup season and week columns must contain finite values: "
            f"{sorted(non_finite_columns)}"
        )

    invalid_values = (
        (identity_values < 1)
        | (identity_values != np.floor(identity_values))
    ).any(axis=0)
    invalid_value_columns = data.loc[:, NUMERIC_IDENTITY_COLUMNS].columns[
        invalid_values
    ].tolist()
    if invalid_value_columns:
        raise ValueError(
            "RB defense matchup season and week columns must contain positive "
            f"integers: {sorted(invalid_value_columns)}"
        )


def _validate_production_columns(data: pd.DataFrame) -> None:
    """Validate numeric defense-game production used by the metrics."""

    invalid_numeric_columns = [
        column
        for column in PRODUCTION_COLUMNS
        if (
            _contains_boolean(data[column])
            or not pd.api.types.is_numeric_dtype(data[column])
        )
    ]
    if invalid_numeric_columns:
        raise ValueError(
            "RB defense matchup production columns must be numeric: "
            f"{sorted(invalid_numeric_columns)}"
        )

    missing_values = data.loc[:, PRODUCTION_COLUMNS].isna().any()
    missing_value_columns = missing_values[missing_values].index.tolist()
    if missing_value_columns:
        raise ValueError(
            "RB defense matchup production columns cannot contain missing values: "
            f"{sorted(missing_value_columns)}"
        )

    production_values = data.loc[:, PRODUCTION_COLUMNS].to_numpy(dtype=float)
    non_finite_columns = data.loc[:, PRODUCTION_COLUMNS].columns[
        ~np.isfinite(production_values).all(axis=0)
    ].tolist()
    if non_finite_columns:
        raise ValueError(
            "RB defense matchup production columns must contain finite values: "
            f"{sorted(non_finite_columns)}"
        )

    negative_values = (
        data.loc[:, NON_NEGATIVE_PRODUCTION_COLUMNS] < 0
    ).any()
    negative_value_columns = negative_values[negative_values].index.tolist()
    if negative_value_columns:
        raise ValueError(
            "RB defense matchup count columns cannot contain negative values: "
            f"{sorted(negative_value_columns)}"
        )


def _contains_boolean(values: pd.Series) -> bool:
    """Return whether a numeric-contract series contains boolean values."""

    return bool(
        values.map(lambda value: isinstance(value, (bool, np.bool_))).any()
    )


def _validate_report_value(value: int, name: str) -> None:
    """Require positive integer season and week parameters."""

    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be an integer greater than or equal to 1")
