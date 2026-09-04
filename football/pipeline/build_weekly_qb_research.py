"""Build reusable weekly quarterback research tables."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import pandas as pd

from football.data.build_schedule_dataset import normalize_schedule_dataset
from football.data.fetch_nflverse import (
    load_depth_charts,
    load_player_game_stats,
    load_schedules,
    normalize_quarterback_game_stats,
)
from football.features.defense_matchup_metrics import build_defense_matchup_metrics
from football.features.defense_vs_quarterbacks import (
    DEFENSE_VS_QUARTERBACK_COLUMNS,
    build_defense_vs_primary_quarterback_logs,
)
from football.features.expected_quarterbacks import resolve_expected_quarterbacks
from football.features.primary_quarterbacks import (
    PRIMARY_QUARTERBACK_COLUMNS,
    identify_primary_quarterbacks,
)
from football.features.qb_form_metrics import build_qb_form_metrics
from football.reports.build_weekly_qb_matchup_report import (
    build_weekly_qb_matchup_report,
)


Loader = Callable[..., Any]


@dataclass(frozen=True)
class WeeklyQBResearchResult:
    """Connected weekly QB research tables for a report slate."""

    summary: pd.DataFrame
    qb_game_logs: pd.DataFrame
    defense_game_logs: pd.DataFrame


def build_weekly_qb_research(
    report_season: int,
    report_week: int,
    as_of_date: object,
    history_season: int | None = None,
    manual_qb_overrides: pd.DataFrame | None = None,
    player_stats_loader: Loader | None = None,
    schedule_loader: Loader | None = None,
    depth_chart_loader: Loader | None = None,
) -> WeeklyQBResearchResult:
    """Build summary, QB logs, and defense logs for weekly QB research."""

    _validate_report_value(report_season, "report_season")
    _validate_report_value(report_week, "report_week")
    selected_history_season = _resolve_history_season(
        report_season,
        report_week,
        history_season,
    )

    player_stats = load_player_game_stats(
        selected_history_season,
        loader=player_stats_loader,
    )
    quarterback_games = normalize_quarterback_game_stats(player_stats)
    primary_qbs = identify_primary_quarterbacks(quarterback_games)
    defense_logs = build_defense_vs_primary_quarterback_logs(primary_qbs)
    history_cutoff_week = _history_cutoff_week(
        quarterback_games,
        selected_history_season,
        report_season,
        report_week,
    )

    qb_metrics = build_qb_form_metrics(
        primary_qbs,
        selected_history_season,
        history_cutoff_week,
    )
    defense_metrics = build_defense_matchup_metrics(
        defense_logs,
        selected_history_season,
        history_cutoff_week,
    )

    schedules = load_schedules(report_season, loader=schedule_loader)
    schedule_rows = normalize_schedule_dataset(schedules)
    depth_charts = load_depth_charts(report_season, loader=depth_chart_loader)
    expected_qbs = resolve_expected_quarterbacks(
        schedule_rows,
        depth_charts,
        report_season,
        report_week,
        as_of_date=as_of_date,
        manual_overrides=manual_qb_overrides,
    )

    summary = build_weekly_qb_matchup_report(
        expected_qbs,
        qb_metrics,
        defense_metrics,
    )
    summary = _preserve_history_context(
        summary,
        selected_history_season,
        history_cutoff_week,
    )

    qb_game_logs = _build_relevant_qb_logs(
        primary_qbs,
        expected_qbs,
        selected_history_season,
        history_cutoff_week,
    )
    defense_game_logs = _build_relevant_defense_logs(
        defense_logs,
        expected_qbs,
        selected_history_season,
        history_cutoff_week,
    )

    return WeeklyQBResearchResult(
        summary=summary,
        qb_game_logs=qb_game_logs,
        defense_game_logs=defense_game_logs,
    )


def _resolve_history_season(
    report_season: int,
    report_week: int,
    history_season: int | None,
) -> int:
    if history_season is None:
        return report_season - 1 if report_week == 1 else report_season

    _validate_report_value(history_season, "history_season")
    if history_season > report_season:
        raise ValueError("history_season must be less than or equal to report_season")

    return history_season


def _history_cutoff_week(
    quarterback_games: pd.DataFrame,
    history_season: int,
    report_season: int,
    report_week: int,
) -> int:
    if history_season == report_season:
        return report_week

    history_weeks = quarterback_games.loc[
        quarterback_games["season"] == history_season,
        "week",
    ]
    if history_weeks.empty:
        return 1

    return int(history_weeks.max()) + 1


def _preserve_history_context(
    summary: pd.DataFrame,
    history_season: int,
    history_cutoff_week: int,
) -> pd.DataFrame:
    result = summary.copy()
    if result.empty:
        return result

    result["qb_history_season"] = result["qb_history_season"].fillna(history_season)
    result["qb_history_cutoff_week"] = result["qb_history_cutoff_week"].fillna(
        history_cutoff_week
    )
    result["defense_history_season"] = result["defense_history_season"].fillna(
        history_season
    )
    result["defense_history_cutoff_week"] = result[
        "defense_history_cutoff_week"
    ].fillna(history_cutoff_week)
    return result


def _build_relevant_qb_logs(
    primary_qbs: pd.DataFrame,
    expected_qbs: pd.DataFrame,
    history_season: int,
    history_cutoff_week: int,
) -> pd.DataFrame:
    expected_player_ids = _non_blank_values(expected_qbs["expected_player_id"])
    rows = primary_qbs.loc[
        (primary_qbs["season"] == history_season)
        & (primary_qbs["week"] < history_cutoff_week)
        & (primary_qbs["player_id"].isin(expected_player_ids)),
        PRIMARY_QUARTERBACK_COLUMNS,
    ].copy()

    return rows.sort_values(
        by=["player_name", "season", "week", "game_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def _build_relevant_defense_logs(
    defense_logs: pd.DataFrame,
    expected_qbs: pd.DataFrame,
    history_season: int,
    history_cutoff_week: int,
) -> pd.DataFrame:
    defenses = _non_blank_values(expected_qbs["opponent"])
    rows = defense_logs.loc[
        (defense_logs["season"] == history_season)
        & (defense_logs["week"] < history_cutoff_week)
        & (defense_logs["defense"].isin(defenses)),
        DEFENSE_VS_QUARTERBACK_COLUMNS,
    ].copy()

    return rows.sort_values(
        by=["defense", "season", "week", "game_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def _non_blank_values(values: pd.Series) -> list[object]:
    value_strings = values.astype("string")
    return values.loc[values.notna() & (value_strings.str.strip() != "")].tolist()


def _validate_report_value(value: int, name: str) -> None:
    """Require positive integer season and week values."""

    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be an integer greater than or equal to 1")
