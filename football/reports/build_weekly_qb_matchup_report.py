"""Build weekly quarterback matchup report rows from prepared inputs."""

from __future__ import annotations

import pandas as pd


EXPECTED_QB_COLUMNS = [
    "season",
    "report_week",
    "game_id",
    "game_date",
    "game_time",
    "team",
    "opponent",
    "home_away",
    "expected_player_id",
    "expected_player_name",
    "selection_source",
    "depth_rank",
    "depth_chart_date",
    "starter_uncertain",
    "selection_notes",
]

QB_METRIC_COLUMNS = [
    "season",
    "report_week",
    "player_id",
    "qb_season_avg",
    "qb_last3_avg",
    "qb_season_attempts_avg",
    "qb_last3_attempts_avg",
    "qb_season_games",
    "qb_last3_games",
    "flagged_games",
]

DEFENSE_METRIC_COLUMNS = [
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

OUTPUT_COLUMNS = [
    "season",
    "report_week",
    "game_id",
    "game_date",
    "game_time",
    "team",
    "opponent",
    "home_away",
    "expected_player_id",
    "expected_player_name",
    "selection_source",
    "depth_rank",
    "depth_chart_date",
    "starter_uncertain",
    "selection_notes",
    "qb_history_season",
    "qb_history_cutoff_week",
    "qb_season_avg",
    "qb_last3_avg",
    "qb_season_attempts_avg",
    "qb_last3_attempts_avg",
    "qb_season_games",
    "qb_last3_games",
    "qb_flagged_games",
    "defense_history_season",
    "defense_history_cutoff_week",
    "defense_season_avg_allowed",
    "defense_last3_avg_allowed",
    "defense_season_games",
    "defense_last3_games",
    "defense_flagged_games",
    "matchup_rank",
    "missing_qb_history",
    "missing_defense_history",
]

EXPECTED_KEYS = ["season", "report_week", "game_id", "team"]


def build_weekly_qb_matchup_report(
    expected_quarterbacks: pd.DataFrame,
    qb_form_metrics: pd.DataFrame,
    defense_matchup_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Return one matchup summary row per scheduled team.

    The expected-quarterback schedule rows are the base table. Quarterback
    metrics join on expected player ID, and defense metrics join on opponent.
    """

    _validate_required_columns(
        expected_quarterbacks,
        EXPECTED_QB_COLUMNS,
        "Expected quarterback data",
    )
    _validate_required_columns(
        qb_form_metrics,
        QB_METRIC_COLUMNS,
        "Quarterback form metrics",
    )
    _validate_required_columns(
        defense_matchup_metrics,
        DEFENSE_METRIC_COLUMNS,
        "Defense matchup metrics",
    )

    if expected_quarterbacks.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    expected = expected_quarterbacks.loc[:, EXPECTED_QB_COLUMNS].copy()
    _validate_unique_expected_rows(expected)

    qbs = _validated_unique_metrics(
        qb_form_metrics.loc[:, QB_METRIC_COLUMNS],
        key_column="player_id",
        label="quarterback form metrics",
    ).rename(
        columns={
            "season": "qb_history_season",
            "report_week": "qb_history_cutoff_week",
            "flagged_games": "qb_flagged_games",
        }
    )

    defenses = _validated_unique_metrics(
        defense_matchup_metrics.loc[:, DEFENSE_METRIC_COLUMNS],
        key_column="defense",
        label="defense matchup metrics",
    ).rename(
        columns={
            "season": "defense_history_season",
            "report_week": "defense_history_cutoff_week",
            "flagged_games": "defense_flagged_games",
        }
    )

    result = expected.merge(
        qbs,
        left_on="expected_player_id",
        right_on="player_id",
        how="left",
    ).drop(columns=["player_id"])

    result = result.merge(
        defenses,
        left_on="opponent",
        right_on="defense",
        how="left",
    ).drop(columns=["defense"])

    unresolved_qb = result["expected_player_id"].isna() | (
        result["expected_player_id"].astype("string").str.strip() == ""
    )
    result["missing_qb_history"] = unresolved_qb | result[
        "qb_history_season"
    ].isna()
    result["missing_defense_history"] = result[
        "defense_history_season"
    ].isna()

    result = result.loc[:, OUTPUT_COLUMNS].sort_values(
        by=["matchup_rank", "game_id", "team"],
        kind="mergesort",
        na_position="last",
    )

    return result.reset_index(drop=True)


def _validate_required_columns(
    data: pd.DataFrame,
    required_columns: list[str],
    label: str,
) -> None:
    """Raise a clear error when a required input column is absent."""

    missing_columns = set(required_columns).difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"{label} is missing required columns: {sorted(missing_columns)}"
        )


def _validate_unique_expected_rows(expected: pd.DataFrame) -> None:
    """Reject multiple expected-QB rows for the same scheduled team-game."""

    duplicates = expected.loc[
        expected.duplicated(subset=EXPECTED_KEYS, keep=False),
        EXPECTED_KEYS,
    ]
    if duplicates.empty:
        return

    duplicate_keys = (
        duplicates.drop_duplicates()
        .sort_values(by=EXPECTED_KEYS, kind="mergesort")
        .to_dict(orient="records")
    )
    raise ValueError(
        "Expected quarterback data contains duplicate scheduled team-game "
        f"rows for keys: {duplicate_keys}"
    )


def _validated_unique_metrics(
    metrics: pd.DataFrame,
    key_column: str,
    label: str,
) -> pd.DataFrame:
    """Drop identical metric duplicates and reject conflicting key duplicates."""

    unique_metrics = metrics.drop_duplicates().copy()
    conflicts = unique_metrics.loc[
        unique_metrics.duplicated(subset=[key_column], keep=False),
        [key_column],
    ]
    if conflicts.empty:
        return unique_metrics

    conflicting_keys = sorted(conflicts[key_column].drop_duplicates().tolist())
    raise ValueError(
        f"Conflicting {label} rows found for {key_column} values: "
        f"{conflicting_keys}"
    )
