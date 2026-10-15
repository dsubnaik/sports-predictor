"""Build a leakage-safe raw dataset for RB rushing-yards modeling.

Each row is one outcome-observed, regular-season RB player-game with a finite
``rushing_yards`` target. This is not reconstructed pregame eligibility:
absent player-games are never manufactured as zeroes. Target-game usage is
never used for that row's features.

Week 1 uses the preceding regular season only. Week 2 and later use only
strictly earlier weeks of the target season. Missing history remains missing;
this raw builder performs neither imputation nor fallback fitting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football.data.build_running_back_dataset import OUTPUT_COLUMNS as RB_GAME_COLUMNS
from football.features.defense_rb_game_logs import build_defense_rb_game_logs
from football.features.running_back_usage import (
    RUNNING_BACK_USAGE_COLUMNS,
    annotate_running_back_usage,
)


TARGET_KEY = ["season", "week", "game_id", "player_id"]
METADATA_COLUMNS = [
    "season", "week", "game_id", "player_id", "player_name", "team", "opponent",
    "home_away", "rb_prior_latest_team", "player_team_changed",
]
TARGET_COLUMN = "target_rushing_yards"
FEATURE_COLUMNS = [
    "rb_season_rushing_yards_avg", "rb_last3_rushing_yards_avg",
    "rb_season_rushing_attempts_avg", "rb_last3_rushing_attempts_avg",
    "rb_season_opportunities_avg", "rb_last3_opportunities_avg",
    "rb_season_carry_share_avg", "rb_last3_carry_share_avg",
    "rb_season_opportunity_share_avg", "rb_last3_opportunity_share_avg",
    "rb_season_history_games", "rb_last3_history_games", "rb_missing_history",
    "defense_season_rb_rushing_yards_allowed_avg",
    "defense_last3_rb_rushing_yards_allowed_avg",
    "defense_season_rb_rushing_attempts_allowed_avg",
    "defense_last3_rb_rushing_attempts_allowed_avg",
    "defense_season_rb_yards_per_carry_allowed",
    "defense_last3_rb_yards_per_carry_allowed",
    "defense_season_history_games", "defense_last3_history_games",
    "defense_missing_history",
]
OUTPUT_COLUMNS = [*METADATA_COLUMNS, *FEATURE_COLUMNS, TARGET_COLUMN]
SCHEDULE_CONTEXT_COLUMNS = ["season", "week", "game_id", "team", "opponent", "home_away"]
SCHEDULE_KEY = ["season", "week", "game_id", "team"]


def build_rb_rushing_yards_training_dataset(
    running_back_games: pd.DataFrame, schedule_rows: pd.DataFrame
) -> pd.DataFrame:
    """Return one prior-only feature row per finite observed RB target.

    Inputs use the normalized RB and team-game schedule contracts. Exact
    duplicate RB rows are collapsed by ``annotate_running_back_usage``;
    conflicting player-game rows and conflicting schedule perspectives raise.
    Shares are calculated from all normalized RB rows before targets are
    selected, so target filtering cannot remove historical teammates.
    """
    _validate_rb_schema(running_back_games)
    schedule = _normalize_schedule_context(schedule_rows)
    annotated = annotate_running_back_usage(running_back_games)
    contextual = _attach_schedule_context(annotated, schedule)
    finite_target = _finite_numeric(contextual["rushing_yards"])
    targets = contextual.loc[finite_target].copy()
    if targets.empty:
        return _empty_output()

    # Finite observed rushing yards define completed player/defense results.
    # Shares were calculated before this filter from the full backfield.
    completed = contextual.loc[finite_target].copy()
    defense_logs = build_defense_rb_game_logs(completed.loc[:, RUNNING_BACK_USAGE_COLUMNS])
    rows: list[dict[str, object]] = []
    for target in targets.sort_values(TARGET_KEY, kind="mergesort").itertuples(index=False):
        history_season, history_week = _history_cutoff(target.season, target.week)
        player_prior = _prior_rows(
            completed, "player_id", target.player_id, history_season, history_week
        )
        defense_prior = _prior_rows(
            defense_logs, "defense", target.opponent, history_season, history_week
        )
        row = {column: getattr(target, column) for column in METADATA_COLUMNS[:8]}
        row.update(_rb_features(player_prior, target.team))
        row.update(_defense_features(defense_prior))
        row[TARGET_COLUMN] = float(target.rushing_yards)
        rows.append(row)
    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    result = result.sort_values(TARGET_KEY, kind="mergesort").reset_index(drop=True)
    _validate_unique_target_keys(result)
    return result


def _validate_rb_schema(data: pd.DataFrame) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Running back games must be a pandas DataFrame")
    missing = sorted(set(RB_GAME_COLUMNS).difference(data.columns))
    if missing:
        raise ValueError(f"Running back games are missing required columns: {missing}")
    duplicates = data.columns[data.columns.duplicated()].tolist()
    if duplicates:
        raise ValueError(f"Running back games contain duplicate columns: {duplicates}")


def _normalize_schedule_context(schedule_rows: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(schedule_rows, pd.DataFrame):
        raise TypeError("Normalized schedule data must be a pandas DataFrame")
    missing = sorted(set(SCHEDULE_CONTEXT_COLUMNS).difference(schedule_rows.columns))
    if missing:
        raise ValueError(f"Normalized schedule data is missing required columns: {missing}")
    context = schedule_rows.loc[:, SCHEDULE_CONTEXT_COLUMNS].drop_duplicates().copy()
    conflicts = context.loc[context.duplicated(SCHEDULE_KEY, keep=False)]
    if not conflicts.empty:
        keys = conflicts.loc[:, SCHEDULE_KEY].drop_duplicates().sort_values(
            SCHEDULE_KEY, kind="mergesort"
        ).to_dict(orient="records")
        raise ValueError(f"Conflicting normalized schedule rows found for keys: {keys}")
    invalid = context.loc[
        context[["team", "opponent", "home_away"]].isna().any(axis=1)
        | context["team"].astype("string").str.strip().eq("")
        | context["opponent"].astype("string").str.strip().eq("")
        | ~context["home_away"].isin(["home", "away"])
        | context["team"].eq(context["opponent"])
    ]
    if not invalid.empty:
        raise ValueError(
            "Normalized schedule rows have invalid context for keys: "
            f"{invalid.loc[:, SCHEDULE_KEY].to_dict(orient='records')}"
        )
    return context.sort_values(SCHEDULE_KEY, kind="mergesort").reset_index(drop=True)


def _attach_schedule_context(running_backs: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    joined = running_backs.merge(
        schedule, on=SCHEDULE_KEY, how="left", validate="many_to_one",
        suffixes=("", "_schedule"), indicator=True,
    )
    missing = joined.loc[joined["_merge"] != "both", TARGET_KEY]
    if not missing.empty:
        raise ValueError(
            "Running back games are missing schedule context for keys: "
            f"{missing.sort_values(TARGET_KEY, kind='mergesort').to_dict(orient='records')}"
        )
    mismatch = joined.loc[joined["opponent"] != joined["opponent_schedule"], TARGET_KEY]
    if not mismatch.empty:
        raise ValueError(
            "Running back games disagree with schedule opponents for keys: "
            f"{mismatch.sort_values(TARGET_KEY, kind='mergesort').to_dict(orient='records')}"
        )
    return joined.drop(columns=["opponent_schedule", "_merge"]).sort_values(
        TARGET_KEY, kind="mergesort"
    ).reset_index(drop=True)


def _history_cutoff(season: int, week: int) -> tuple[int, int]:
    return (season - 1, np.iinfo(np.int64).max) if week == 1 else (season, week)


def _prior_rows(data: pd.DataFrame, entity_column: str, entity: object,
                season: int, week: int) -> pd.DataFrame:
    return data.loc[
        data[entity_column].eq(entity) & data["season"].eq(season) & data["week"].lt(week)
    ].sort_values(["season", "week", "game_id"], kind="mergesort")


def _rb_features(prior: pd.DataFrame, target_team: object) -> dict[str, object]:
    last3 = prior.tail(3)
    count = len(prior)
    latest_team = prior["team"].iloc[-1] if count else pd.NA
    return {
        "rb_season_rushing_yards_avg": prior["rushing_yards"].mean(),
        "rb_last3_rushing_yards_avg": last3["rushing_yards"].mean(),
        "rb_season_rushing_attempts_avg": prior["rushing_attempts"].mean(),
        "rb_last3_rushing_attempts_avg": last3["rushing_attempts"].mean(),
        "rb_season_opportunities_avg": prior["opportunities"].mean(),
        "rb_last3_opportunities_avg": last3["opportunities"].mean(),
        "rb_season_carry_share_avg": prior["carry_share"].mean(),
        "rb_last3_carry_share_avg": last3["carry_share"].mean(),
        "rb_season_opportunity_share_avg": prior["opportunity_share"].mean(),
        "rb_last3_opportunity_share_avg": last3["opportunity_share"].mean(),
        "rb_season_history_games": count,
        "rb_last3_history_games": len(last3),
        "rb_missing_history": count == 0,
        "rb_prior_latest_team": latest_team,
        "player_team_changed": bool(count and latest_team != target_team),
    }


def _defense_features(prior: pd.DataFrame) -> dict[str, object]:
    last3 = prior.tail(3)
    return {
        "defense_season_rb_rushing_yards_allowed_avg": prior["rb_rushing_yards_allowed"].mean(),
        "defense_last3_rb_rushing_yards_allowed_avg": last3["rb_rushing_yards_allowed"].mean(),
        "defense_season_rb_rushing_attempts_allowed_avg": prior["rb_rushing_attempts_allowed"].mean(),
        "defense_last3_rb_rushing_attempts_allowed_avg": last3["rb_rushing_attempts_allowed"].mean(),
        "defense_season_rb_yards_per_carry_allowed": _weighted_yards_per_carry(prior),
        "defense_last3_rb_yards_per_carry_allowed": _weighted_yards_per_carry(last3),
        "defense_season_history_games": len(prior),
        "defense_last3_history_games": len(last3),
        "defense_missing_history": len(prior) == 0,
    }


def _weighted_yards_per_carry(games: pd.DataFrame) -> float:
    attempts = games["rb_rushing_attempts_allowed"].sum()
    return np.nan if attempts == 0 else float(games["rb_rushing_yards_allowed"].sum() / attempts)


def _finite_numeric(values: pd.Series) -> pd.Series:
    return pd.Series(np.isfinite(pd.to_numeric(values, errors="coerce")), index=values.index)


def _validate_unique_target_keys(data: pd.DataFrame) -> None:
    if data.duplicated(TARGET_KEY).any():
        raise ValueError("RB rushing-yards training dataset has duplicate target keys")


def _empty_output() -> pd.DataFrame:
    return pd.DataFrame(columns=OUTPUT_COLUMNS)
