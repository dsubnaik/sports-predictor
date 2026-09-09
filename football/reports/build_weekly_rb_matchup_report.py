"""Build weekly running-back matchup report rows from prepared inputs.

The report is research context, not a prediction. RB and defense metrics are
assumed to be leakage-safe upstream inputs. Multiple expected RBs per team,
unresolved participants, and missing history rows are intentionally retained.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd

from football.data.build_schedule_dataset import OUTPUT_COLUMNS as SCHEDULE_COLUMNS
from football.features.expected_running_backs import (
    OUTPUT_COLUMNS as EXPECTED_RB_COLUMNS,
)
from football.features.rb_defense_matchup_metrics import (
    OUTPUT_COLUMNS as RB_DEFENSE_COLUMNS,
)
from football.features.rb_form_metrics import OUTPUT_COLUMNS as RB_FORM_COLUMNS


SCHEDULE_CONTEXT_COLUMNS = [
    "season",
    "week",
    "game_id",
    "game_date",
    "game_time",
    "team",
    "opponent",
    "home_away",
]

PARTICIPANT_COLUMNS = [
    "report_season",
    "report_week",
    "team",
    "player_id",
    "player_name",
    "position",
    "participant_order",
    "selection_source",
    "depth_chart_date",
    "depth_chart_week",
    "depth_position",
    "depth_rank",
    "resolution_missing",
    "selection_notes",
]

RB_METRIC_COLUMNS = [
    column for column in RB_FORM_COLUMNS if column != "player_name"
]

DEFENSE_METRIC_COLUMNS = RB_DEFENSE_COLUMNS.copy()

RB_SUMMARY_METRIC_COLUMNS = [
    column
    for column in RB_METRIC_COLUMNS
    if column
    not in [
        "report_season",
        "report_week",
        "historical_season",
        "player_id",
    ]
]

DEFENSE_SUMMARY_METRIC_COLUMNS = [
    column
    for column in DEFENSE_METRIC_COLUMNS
    if column
    not in [
        "report_season",
        "report_week",
        "historical_season",
        "defense",
    ]
]

OUTPUT_COLUMNS = [
    "report_season",
    "report_week",
    "game_id",
    "game_date",
    "game_time",
    "team",
    "opponent",
    "home_away",
    "player_id",
    "player_name",
    "position",
    "participant_order",
    "selection_source",
    "depth_chart_date",
    "depth_chart_week",
    "depth_position",
    "depth_rank",
    "resolution_missing",
    "selection_notes",
    "historical_season",
    *RB_SUMMARY_METRIC_COLUMNS,
    *DEFENSE_SUMMARY_METRIC_COLUMNS,
    "rb_history_missing",
    "defense_history_missing",
    "participant_resolution_missing",
    "participant_team_mismatch",
    "multiple_expected_rbs",
    "limited_rb_sample",
    "limited_defense_sample",
]

SCHEDULE_KEYS = ["season", "week", "team"]
EXPECTED_KEYS = ["report_season", "report_week", "team", "player_id"]
SAMPLE_LIMIT = 3


def build_weekly_rb_matchup_report(
    schedule_rows: pd.DataFrame,
    expected_running_backs: pd.DataFrame,
    rb_form_metrics: pd.DataFrame,
    rb_defense_matchup_metrics: pd.DataFrame,
    report_season: int,
    report_week: int,
) -> pd.DataFrame:
    """Return one weekly matchup row per scheduled expected RB participant."""

    _validate_report_value(report_season, "report_season")
    _validate_report_value(report_week, "report_week")
    _validate_required_columns(schedule_rows, SCHEDULE_COLUMNS, "Schedule data")
    _validate_required_columns(
        expected_running_backs,
        EXPECTED_RB_COLUMNS,
        "Expected running-back data",
    )
    _validate_required_columns(
        rb_form_metrics,
        RB_FORM_COLUMNS,
        "Running-back form metrics",
    )
    _validate_required_columns(
        rb_defense_matchup_metrics,
        RB_DEFENSE_COLUMNS,
        "RB defense matchup metrics",
    )

    schedule = _selected_schedule(schedule_rows, report_season, report_week)
    if schedule.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    _validate_report_context(
        expected_running_backs,
        ["report_season", "report_week"],
        report_season,
        report_week,
        "Expected running-back data",
    )
    _validate_report_context(
        rb_form_metrics,
        ["report_season", "report_week"],
        report_season,
        report_week,
        "Running-back form metrics",
    )
    _validate_report_context(
        rb_defense_matchup_metrics,
        ["report_season", "report_week"],
        report_season,
        report_week,
        "RB defense matchup metrics",
    )
    historical_season = _resolve_historical_season(
        rb_form_metrics,
        rb_defense_matchup_metrics,
    )

    expected = _selected_expected(expected_running_backs, schedule)
    rbs = _validated_unique_metrics(
        rb_form_metrics.loc[:, RB_METRIC_COLUMNS],
        key_column="player_id",
        label="running-back form metrics",
    ).drop(columns=["report_season", "report_week"])
    defenses = _validated_unique_metrics(
        rb_defense_matchup_metrics.loc[:, DEFENSE_METRIC_COLUMNS],
        key_column="defense",
        label="RB defense matchup metrics",
    ).drop(columns=["report_season", "report_week"])

    result = schedule.merge(
        expected,
        on="team",
        how="left",
        validate="one_to_many",
    )
    result = _fill_missing_expected_rows(result, report_season, report_week)
    result = result.merge(
        rbs,
        on="player_id",
        how="left",
        validate="many_to_one",
    )
    result = result.merge(
        defenses,
        left_on="opponent",
        right_on="defense",
        how="left",
        validate="many_to_one",
    ).drop(columns=["defense"])

    rb_history_season = result.pop("historical_season_x")
    defense_history_season = result.pop("historical_season_y")
    result["historical_season"] = rb_history_season.combine_first(
        defense_history_season
    )
    if historical_season is not pd.NA:
        result["historical_season"] = result["historical_season"].fillna(
            historical_season
        )

    result["report_season"] = report_season
    result["report_week"] = report_week
    result["participant_resolution_missing"] = result[
        "resolution_missing"
    ].fillna(False).astype(bool)
    result["rb_history_missing"] = (
        ~result["participant_resolution_missing"] & rb_history_season.isna()
    )
    result["defense_history_missing"] = defense_history_season.isna()
    result["participant_team_mismatch"] = (
        result["latest_team"].notna()
        & result["team"].ne(result["latest_team"])
    )
    resolved_mask = ~result["participant_resolution_missing"].astype(bool)
    resolved_counts = (
        result.loc[resolved_mask]
        .groupby("team", sort=False)["player_id"]
        .transform("count")
    )
    result["_resolved_count"] = pd.Series(0, index=result.index, dtype="Int64")
    result.loc[
        resolved_mask, "_resolved_count"
    ] = resolved_counts
    result["multiple_expected_rbs"] = result["_resolved_count"].gt(1)
    result["limited_rb_sample"] = (
        ~result["rb_history_missing"]
        & result["rb_season_games"].notna()
        & result["rb_season_games"].lt(SAMPLE_LIMIT)
    )
    result["limited_defense_sample"] = (
        ~result["defense_history_missing"]
        & result["defense_season_games"].notna()
        & result["defense_season_games"].lt(SAMPLE_LIMIT)
    )

    result = result.loc[:, OUTPUT_COLUMNS].sort_values(
        by=[
            "matchup_rank",
            "game_id",
            "team",
            "participant_order",
            "player_name",
            "player_id",
        ],
        kind="mergesort",
        na_position="last",
    )
    return result.reset_index(drop=True)


def _selected_schedule(
    schedule_rows: pd.DataFrame,
    report_season: int,
    report_week: int,
) -> pd.DataFrame:
    schedule = schedule_rows.loc[
        (schedule_rows["season"] == report_season)
        & (schedule_rows["week"] == report_week),
        SCHEDULE_CONTEXT_COLUMNS,
    ].drop_duplicates().copy()
    _reject_conflicting_duplicates(
        schedule,
        ["season", "week", "team"],
        "Schedule data contains conflicting team rows",
    )
    return schedule.rename(
        columns={"season": "report_season", "week": "report_week"}
    )


def _selected_expected(
    expected_running_backs: pd.DataFrame,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    expected = expected_running_backs.loc[:, PARTICIPANT_COLUMNS].drop_duplicates().copy()
    scheduled_teams = set(schedule["team"])
    unknown_teams = sorted(set(expected["team"]) - scheduled_teams)
    if unknown_teams:
        raise ValueError(
            "Expected running-back data contains teams not in the selected "
            f"schedule: {unknown_teams}"
        )

    _validate_expected_duplicates(expected)
    return expected.drop(columns=["report_season", "report_week"])


def _fill_missing_expected_rows(
    result: pd.DataFrame,
    report_season: int,
    report_week: int,
) -> pd.DataFrame:
    missing = result["selection_source"].isna()
    if not missing.any():
        return result

    result = result.copy()
    result.loc[missing, "position"] = "RB"
    result.loc[missing, "selection_source"] = "unresolved"
    result.loc[missing, "resolution_missing"] = True
    result.loc[missing, "selection_notes"] = (
        "No expected RB participant rows supplied for scheduled team"
    )
    return result


def _validate_expected_duplicates(expected: pd.DataFrame) -> None:
    if expected.empty:
        return

    resolution_missing = expected["resolution_missing"].fillna(False).astype(bool)
    resolved = expected.loc[~resolution_missing].copy()
    unresolved = expected.loc[resolution_missing].copy()

    mixed_teams = sorted(set(resolved["team"]).intersection(unresolved["team"]))
    if mixed_teams:
        raise ValueError(
            "Expected running-back data cannot mix resolved and unresolved "
            f"participant rows for teams: {mixed_teams}"
        )

    _reject_conflicting_duplicates(
        resolved,
        ["team", "player_id"],
        "Expected running-back data contains conflicting participant rows",
    )
    _reject_conflicting_duplicates(
        resolved,
        ["team", "participant_order"],
        "Expected running-back data contains conflicting participant order rows",
    )
    _reject_conflicting_duplicates(
        unresolved,
        ["team"],
        "Expected running-back data contains duplicate unresolved rows",
    )


def _resolve_historical_season(
    rb_form_metrics: pd.DataFrame,
    defense_metrics: pd.DataFrame,
) -> object:
    rb_historical = _single_historical_season(
        rb_form_metrics, "Running-back form metrics"
    )
    defense_historical = _single_historical_season(
        defense_metrics, "RB defense matchup metrics"
    )
    if rb_historical is not pd.NA and defense_historical is not pd.NA:
        if rb_historical != defense_historical:
            raise ValueError(
                "Running-back form metrics and RB defense matchup metrics "
                "must use the same historical_season"
            )
    if rb_historical is not pd.NA:
        return rb_historical
    return defense_historical


def _single_historical_season(data: pd.DataFrame, label: str) -> object:
    if data.empty:
        return pd.NA
    values = data["historical_season"].drop_duplicates()
    if len(values) > 1:
        raise ValueError(f"{label} contains multiple historical_season values")
    return values.iloc[0]


def _validate_report_context(
    data: pd.DataFrame,
    columns: list[str],
    report_season: int,
    report_week: int,
    label: str,
) -> None:
    if data.empty:
        return
    invalid = data.loc[
        ~(
            data[columns[0]].eq(report_season)
            & data[columns[1]].eq(report_week)
        ),
        columns,
    ]
    if invalid.empty:
        return
    contexts = (
        invalid.drop_duplicates()
        .sort_values(columns, kind="mergesort")
        .to_dict("records")
    )
    raise ValueError(
        f"{label} contains rows outside the requested report context: "
        f"{contexts}"
    )


def _validated_unique_metrics(
    metrics: pd.DataFrame,
    key_column: str,
    label: str,
) -> pd.DataFrame:
    unique_metrics = metrics.drop_duplicates().copy()
    conflicts = unique_metrics.loc[
        unique_metrics.duplicated(subset=[key_column], keep=False),
        [key_column],
    ]
    if conflicts.empty:
        return unique_metrics

    keys = sorted(conflicts[key_column].drop_duplicates().tolist())
    raise ValueError(
        f"Conflicting {label} rows found for {key_column} values: {keys}"
    )


def _reject_conflicting_duplicates(
    data: pd.DataFrame,
    keys: list[str],
    message: str,
) -> None:
    if data.empty:
        return
    conflicts = data.loc[data.duplicated(subset=keys, keep=False), keys]
    if conflicts.empty:
        return
    conflicting_keys = (
        conflicts.drop_duplicates()
        .sort_values(keys, kind="mergesort")
        .to_dict("records")
    )
    raise ValueError(f"{message} for keys: {conflicting_keys}")


def _validate_required_columns(
    data: pd.DataFrame,
    required_columns: list[str],
    label: str,
) -> None:
    missing_columns = sorted(set(required_columns).difference(data.columns))
    if missing_columns:
        raise ValueError(
            f"{label} is missing required columns: {missing_columns}"
        )


def _validate_report_value(value: object, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be an integer greater than or equal to 1")
