"""Build reusable weekly running-back research tables.

This orchestration layer composes public RB components only. It preserves all
expected backfield participants; it does not choose a primary running back or
make predictions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import pandas as pd

from football.data.build_running_back_dataset import build_running_back_dataset
from football.data.build_schedule_dataset import normalize_schedule_dataset
from football.data.fetch_nflverse import (
    load_depth_charts,
    load_player_game_stats,
    load_schedules,
)
from football.data.normalize_depth_charts import normalize_nflverse_depth_charts
from football.features.defense_rb_game_logs import (
    DEFENSE_RB_GAME_LOG_COLUMNS,
    build_defense_rb_game_logs,
)
from football.features.expected_running_backs import resolve_expected_running_backs
from football.features.rb_defense_matchup_metrics import (
    build_rb_defense_matchup_metrics,
)
from football.features.rb_form_metrics import build_rb_form_metrics
from football.features.running_back_usage import (
    RUNNING_BACK_USAGE_COLUMNS,
    annotate_running_back_usage,
)
from football.reports.build_weekly_rb_matchup_report import (
    OUTPUT_COLUMNS as RB_REPORT_COLUMNS,
    build_weekly_rb_matchup_report,
)


Loader = Callable[..., Any]


@dataclass(frozen=True)
class WeeklyRBResearchResult:
    """Connected weekly RB research tables for a report slate."""

    summary: pd.DataFrame
    rb_game_logs: pd.DataFrame
    defense_game_logs: pd.DataFrame


def build_weekly_rb_research(
    report_season: int,
    report_week: int,
    as_of_date: object,
    history_season: int | None = None,
    manual_rb_overrides: pd.DataFrame | None = None,
    player_stats_loader: Loader | None = None,
    schedule_loader: Loader | None = None,
    depth_chart_loader: Loader | None = None,
) -> WeeklyRBResearchResult:
    """Build weekly RB summary and relevant historical research logs.

    ``as_of_date`` is the global dated depth-chart cutoff, matching the QB
    pipeline convention. Callers should provide a cutoff that is safe for the
    whole report slate; this function does not infer game-specific cutoffs.
    """

    _validate_report_value(report_season, "report_season")
    _validate_report_value(report_week, "report_week")

    schedules = load_schedules(report_season, loader=schedule_loader)
    schedule_rows = normalize_schedule_dataset(schedules)
    selected_schedule = schedule_rows.loc[
        (schedule_rows["season"] == report_season)
        & (schedule_rows["week"] == report_week)
    ].copy()
    selected_history_season = _resolve_history_season(
        report_season,
        report_week,
        history_season,
    )
    if selected_schedule.empty:
        return _empty_result()

    player_stats = load_player_game_stats(
        selected_history_season,
        loader=player_stats_loader,
    )
    running_back_games = build_running_back_dataset(player_stats)
    annotated_rbs = annotate_running_back_usage(running_back_games)
    defense_logs = build_defense_rb_game_logs(annotated_rbs)
    history_cutoff_week = _history_cutoff_week(
        annotated_rbs,
        selected_history_season,
        report_season,
        report_week,
    )

    rb_metrics = build_rb_form_metrics(
        annotated_rbs,
        report_season,
        report_week,
        historical_season=selected_history_season,
    )
    defense_metrics = build_rb_defense_matchup_metrics(
        defense_logs,
        report_season,
        report_week,
        historical_season=selected_history_season,
    )

    raw_depth_charts = load_depth_charts(
        report_season,
        loader=depth_chart_loader,
    )
    canonical_depth_charts = normalize_nflverse_depth_charts(raw_depth_charts)
    requested_teams = selected_schedule.loc[:, ["team"]].copy()
    expected_rbs = resolve_expected_running_backs(
        requested_teams,
        canonical_depth_charts,
        report_season,
        report_week,
        as_of_date,
        manual_overrides=manual_rb_overrides,
    )

    summary = build_weekly_rb_matchup_report(
        schedule_rows,
        expected_rbs,
        rb_metrics,
        defense_metrics,
        report_season,
        report_week,
    )
    summary = _preserve_history_context(summary, selected_history_season)

    return WeeklyRBResearchResult(
        summary=summary,
        rb_game_logs=_build_relevant_rb_logs(
            annotated_rbs,
            expected_rbs,
            selected_history_season,
            history_cutoff_week,
        ),
        defense_game_logs=_build_relevant_defense_logs(
            defense_logs,
            selected_schedule,
            selected_history_season,
            history_cutoff_week,
        ),
    )


def _empty_result() -> WeeklyRBResearchResult:
    return WeeklyRBResearchResult(
        summary=pd.DataFrame(columns=RB_REPORT_COLUMNS),
        rb_game_logs=pd.DataFrame(columns=RUNNING_BACK_USAGE_COLUMNS),
        defense_game_logs=pd.DataFrame(columns=DEFENSE_RB_GAME_LOG_COLUMNS),
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
    annotated_rbs: pd.DataFrame,
    history_season: int,
    report_season: int,
    report_week: int,
) -> int:
    if history_season == report_season:
        return report_week

    history_weeks = annotated_rbs.loc[
        annotated_rbs["season"].eq(history_season), "week"
    ]
    if history_weeks.empty:
        return 1
    return int(history_weeks.max()) + 1


def _preserve_history_context(
    summary: pd.DataFrame,
    history_season: int,
) -> pd.DataFrame:
    result = summary.copy()
    if not result.empty:
        result["historical_season"] = result["historical_season"].fillna(
            history_season
        )
    return result


def _build_relevant_rb_logs(
    annotated_rbs: pd.DataFrame,
    expected_rbs: pd.DataFrame,
    history_season: int,
    history_cutoff_week: int,
) -> pd.DataFrame:
    resolved_mask = ~expected_rbs["resolution_missing"].fillna(False).astype(bool)
    player_ids = _non_blank_values(expected_rbs.loc[resolved_mask, "player_id"])
    rows = annotated_rbs.loc[
        annotated_rbs["season"].eq(history_season)
        & annotated_rbs["week"].lt(history_cutoff_week)
        & annotated_rbs["player_id"].isin(player_ids),
        RUNNING_BACK_USAGE_COLUMNS,
    ].copy()
    return rows.sort_values(
        by=["player_name", "player_id", "season", "week", "game_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def _build_relevant_defense_logs(
    defense_logs: pd.DataFrame,
    selected_schedule: pd.DataFrame,
    history_season: int,
    history_cutoff_week: int,
) -> pd.DataFrame:
    defenses = _non_blank_values(selected_schedule["opponent"])
    rows = defense_logs.loc[
        defense_logs["season"].eq(history_season)
        & defense_logs["week"].lt(history_cutoff_week)
        & defense_logs["defense"].isin(defenses),
        DEFENSE_RB_GAME_LOG_COLUMNS,
    ].copy()
    return rows.sort_values(
        by=["defense", "season", "week", "game_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def _non_blank_values(values: pd.Series) -> list[object]:
    strings = values.astype("string")
    return values.loc[values.notna() & strings.str.strip().ne("")].tolist()


def _validate_report_value(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be an integer greater than or equal to 1")
