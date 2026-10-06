"""Leakage-safe pre-kickoff QB passing-yards feature construction.

This module deliberately creates candidate rows from schedules and dated depth
charts, not from target-game QB participation.  Historical QB outcomes are
used only from the builder's permitted history window: the previous season for
week one, otherwise strictly earlier weeks in the target season.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from football.data.build_quarterback_dataset import build_quarterback_dataset
from football.features.defense_vs_quarterbacks import build_defense_vs_primary_quarterback_logs
from football.features.primary_quarterbacks import identify_primary_quarterbacks
from football.training.build_qb_passing_yards_dataset import FEATURE_COLUMNS, TARGET_KEY
from football.training.qb_passing_yards_production_model import (
    QBPassingYardsLoadedArtifact,
    predict_qb_passing_yards_production_model,
)


SCHEDULE_KEY = ("season", "week", "game_id", "team")
SELECTION_STATUS = "likely_primary"


@dataclass(frozen=True)
class QBPassingYardsWeeklySkip:
    game_id: object
    team: object | None
    reason: str


@dataclass(frozen=True)
class QBPassingYardsWeeklyFeatureRow:
    season: int
    week: int
    game_id: object
    scheduled_kickoff_utc: str
    player_id: object
    player_name: str | None
    team: object
    opponent: object
    home_away: str
    selection_status: str
    feature_values: tuple[float | bool | None, ...]


@dataclass(frozen=True)
class QBPassingYardsWeeklyFeatureRows:
    rows: tuple[QBPassingYardsWeeklyFeatureRow, ...]
    skips: tuple[QBPassingYardsWeeklySkip, ...]
    scheduled_games_considered: int
    eligible_games: int
    team_games_considered: int
    likely_primary_selections: int
    unknown_team_games: int
    cold_start_projections: int
    defense_missing_projections: int


@dataclass(frozen=True)
class QBPassingYardsWeeklyProjection:
    season: int
    week: int
    game_id: object
    scheduled_kickoff_utc: str
    player_id: object
    player_name: str | None
    team: object
    opponent: object
    home_away: str
    predicted_passing_yards: float
    qb_cold_start: bool
    defense_missing_history: bool
    model_schema_version: int
    artifact_identifier: str
    as_of_utc: str


@dataclass(frozen=True)
class QBPassingYardsWeeklyProjectionReport:
    projections: tuple[QBPassingYardsWeeklyProjection, ...]
    feature_diagnostics: QBPassingYardsWeeklyFeatureRows


def build_qb_passing_yards_weekly_feature_rows(
    historical_quarterback_games: pd.DataFrame,
    schedule_rows: pd.DataFrame,
    depth_chart_snapshots: pd.DataFrame,
    *,
    target_season: int,
    target_week: int,
    as_of_utc: object,
) -> QBPassingYardsWeeklyFeatureRows:
    """Build feature rows for scheduled, not-yet-started team-games.

    ``schedule_rows`` requires a timezone-aware ``scheduled_kickoff`` column.
    This explicit timestamp prevents treating a same-day depth chart or a game
    at the as-of instant as pregame information.
    """
    as_of = _utc_timestamp(as_of_utc, "as_of_utc")
    schedule = _schedule_context(schedule_rows, target_season, target_week)
    target = schedule.loc[(schedule["season"] == target_season) & (schedule["week"] == target_week)].copy()
    if "season_type" in schedule_rows.columns:
        regular = schedule_rows.loc[(schedule_rows["season"] == target_season) & (schedule_rows["week"] == target_week), ["game_id", "season_type"]].drop_duplicates()
        allowed = set(regular.loc[regular["season_type"].astype(str).str.upper().isin({"REG", "REGULAR", "REGULAR_SEASON"}), "game_id"])
        target = target.loc[target["game_id"].isin(allowed)].copy()
    scheduled_games = int(target["game_id"].nunique())
    eligible, skips = _eligible_schedule(target, as_of)
    if eligible.empty:
        return QBPassingYardsWeeklyFeatureRows((), tuple(skips), scheduled_games, 0, 0, 0, 0, 0, 0)

    selected, unknown = _select_candidates(eligible, depth_chart_snapshots, as_of)
    history = _history_with_context(historical_quarterback_games, schedule)
    rows: list[QBPassingYardsWeeklyFeatureRow] = []
    for candidate in selected.itertuples(index=False):
        history_season, history_week = _history_cutoff(target_season, target_week)
        qb_prior = _prior(history, "player_id", candidate.player_id, history_season, history_week)
        defense_logs = _defense_logs(history)
        defense_prior = _prior(defense_logs, "defense", candidate.opponent, history_season, history_week)
        defense_window = defense_logs.loc[(defense_logs["season"] == history_season) & (defense_logs["week"] < history_week)]
        values = _qb_features(qb_prior) | _defense_features(defense_prior, defense_window, candidate.opponent)
        rows.append(QBPassingYardsWeeklyFeatureRow(
            int(candidate.season), int(candidate.week), candidate.game_id,
            pd.Timestamp(candidate.scheduled_kickoff).isoformat(), candidate.player_id,
            candidate.player_name if pd.notna(candidate.player_name) else None,
            candidate.team, candidate.opponent, candidate.home_away, SELECTION_STATUS,
            tuple(_safe_value(values[name]) for name in FEATURE_COLUMNS),
        ))
    rows.sort(key=lambda row: (row.season, row.week, str(row.game_id), str(row.player_id)))
    skips.extend(QBPassingYardsWeeklySkip(row.game_id, row.team, "no_usable_pregame_depth_chart_candidate") for row in unknown.itertuples(index=False))
    cold = sum(bool(row.feature_values[6]) for row in rows)
    defense_missing = sum(bool(row.feature_values[12]) for row in rows)
    return QBPassingYardsWeeklyFeatureRows(tuple(rows), tuple(sorted(skips, key=lambda item: (str(item.game_id), str(item.team), item.reason))), scheduled_games, int(eligible["game_id"].nunique()), len(eligible), len(rows), len(unknown), cold, defense_missing)


def generate_qb_passing_yards_weekly_projections(
    loaded_artifact: QBPassingYardsLoadedArtifact,
    historical_quarterback_games: pd.DataFrame,
    schedule_rows: pd.DataFrame,
    depth_chart_snapshots: pd.DataFrame,
    *,
    target_season: int,
    target_week: int,
    as_of_utc: object,
) -> QBPassingYardsWeeklyProjectionReport:
    """Generate unscored pregame projections using the trusted loaded artifact."""
    features = build_qb_passing_yards_weekly_feature_rows(historical_quarterback_games, schedule_rows, depth_chart_snapshots, target_season=target_season, target_week=target_week, as_of_utc=as_of_utc)
    if not features.rows:
        return QBPassingYardsWeeklyProjectionReport((), features)
    frame = _prediction_frame(features.rows)
    predicted = predict_qb_passing_yards_production_model(loaded_artifact, frame)
    expected = tuple(tuple(getattr(row, key) for key in TARGET_KEY) for row in features.rows)
    if predicted.keys != expected or len(predicted.predictions) != len(features.rows):
        raise ValueError("Production prediction keys do not align with weekly feature rows")
    as_of = _utc_timestamp(as_of_utc, "as_of_utc").isoformat()
    projections = tuple(
        QBPassingYardsWeeklyProjection(
            row.season, row.week, row.game_id, row.scheduled_kickoff_utc,
            row.player_id, row.player_name, row.team, row.opponent, row.home_away,
            float(value), bool(row.feature_values[6]), bool(row.feature_values[12]),
            loaded_artifact.metadata.schema_version, loaded_artifact.artifact_name, as_of,
        )
        for row, value in zip(features.rows, predicted.predictions, strict=True)
    )
    return QBPassingYardsWeeklyProjectionReport(projections, features)


def _utc_timestamp(value: object, label: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    if pd.isna(timestamp):
        raise ValueError(f"{label} must be a valid timestamp")
    return timestamp.tz_convert("UTC")


def _schedule_context(rows: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    required = [*SCHEDULE_KEY, "opponent", "home_away", "scheduled_kickoff"]
    if not isinstance(rows, pd.DataFrame):
        raise TypeError("schedule_rows must be a pandas DataFrame")
    missing = sorted(set(required).difference(rows.columns))
    if missing:
        raise ValueError(f"Schedule rows are missing required columns: {missing}")
    data = rows.loc[:, [*required, *(["status"] if "status" in rows else [])]].copy()
    if data.duplicated(list(SCHEDULE_KEY)).any() or data[list(SCHEDULE_KEY)].isna().any().any():
        raise ValueError("Schedule rows must have unique, nonmissing team-game keys")
    if data[["team", "opponent", "home_away"]].isna().any().any() or (~data["home_away"].isin(["home", "away"])).any() or (data["team"] == data["opponent"]).any():
        raise ValueError("Schedule rows have invalid team/opponent context")
    _validate_schedule_game_mappings(data)
    data["scheduled_kickoff"] = data["scheduled_kickoff"].map(_optional_utc_timestamp)
    data["_invalid_kickoff"] = data["scheduled_kickoff"].isna()
    return data.sort_values(list(SCHEDULE_KEY), kind="mergesort").reset_index(drop=True)


def _validate_schedule_game_mappings(data: pd.DataFrame) -> None:
    """Require the injected schedule's two reciprocal team perspectives."""
    for game_id, game in data.groupby(["season", "week", "game_id"], sort=False):
        if len(game) != 2:
            raise ValueError(f"Schedule game {game_id} must contain exactly two team rows")
        first, second = tuple(game.itertuples(index=False))
        if first.team != second.opponent or second.team != first.opponent or {first.home_away, second.home_away} != {"home", "away"}:
            raise ValueError(f"Schedule game {game_id} has conflicting team/opponent mappings")


def _optional_utc_timestamp(value: object) -> pd.Timestamp:
    try:
        return _utc_timestamp(value, "scheduled_kickoff")
    except (TypeError, ValueError):
        return pd.NaT


def _eligible_schedule(target: pd.DataFrame, as_of: pd.Timestamp) -> tuple[pd.DataFrame, list[QBPassingYardsWeeklySkip]]:
    skips: list[QBPassingYardsWeeklySkip] = []
    status = target["status"].astype(str).str.lower() if "status" in target else pd.Series("scheduled", index=target.index)
    terminal = status.isin({"final", "completed", "canceled", "cancelled", "postponed"})
    invalid_kickoff = target["_invalid_kickoff"]
    after = target["scheduled_kickoff"] > as_of
    for _, row in target.loc[terminal | invalid_kickoff | ~after].iterrows():
        reason = "terminal_game_status" if str(row.get("status", "")).lower() in {"final", "completed", "canceled", "cancelled", "postponed"} else ("invalid_or_missing_kickoff" if bool(row["_invalid_kickoff"]) else "kickoff_not_strictly_after_as_of")
        skips.append(QBPassingYardsWeeklySkip(row["game_id"], row["team"], reason))
    return target.loc[~terminal & ~invalid_kickoff & after].copy(), skips


def _select_candidates(schedule: pd.DataFrame, snapshots: pd.DataFrame, as_of: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = ["team", "player_id", "position", "snapshot_timestamp", "depth_rank"]
    if not isinstance(snapshots, pd.DataFrame):
        raise TypeError("depth_chart_snapshots must be a pandas DataFrame")
    missing = sorted(set(required).difference(snapshots.columns))
    if missing:
        raise ValueError(f"Depth-chart snapshots are missing required columns: {missing}")
    cols = [*required, *(["player_name"] if "player_name" in snapshots else [])]
    data = snapshots.loc[:, cols].copy()
    data["snapshot_timestamp"] = data["snapshot_timestamp"].map(lambda value: _utc_timestamp(value, "snapshot_timestamp"))
    data["depth_rank"] = pd.to_numeric(data["depth_rank"], errors="coerce")
    # Source snapshots can contain unranked QB rows.  They are not candidates;
    # an explicitly supplied numeric rank must nevertheless be a positive
    # integer so a malformed ranking cannot silently choose a player.
    qb = data["position"].eq("QB")
    invalid = qb & data["depth_rank"].notna() & (~np.isfinite(data["depth_rank"]) | data["depth_rank"].lt(1) | data["depth_rank"].mod(1).ne(0))
    if invalid.any():
        raise ValueError("QB depth-chart snapshots have invalid ranks or identifiers")
    data = data.loc[qb & data["player_id"].notna() & data["team"].notna() & data["snapshot_timestamp"].notna() & data["depth_rank"].notna() & np.isfinite(data["depth_rank"])].copy()
    joined = schedule.merge(data, on="team", how="left", validate="one_to_many")
    usable = joined.loc[joined["snapshot_timestamp"].notna() & (joined["snapshot_timestamp"] < joined["scheduled_kickoff"]) & (joined["snapshot_timestamp"] < as_of) & (joined["snapshot_timestamp"].dt.year == joined["season"])].copy()
    if usable.empty:
        return pd.DataFrame(columns=[*schedule.columns, "player_id", "player_name"]), schedule.copy()
    group = list(SCHEDULE_KEY)
    latest = usable.groupby(group, sort=False)["snapshot_timestamp"].transform("max")
    usable = usable.loc[usable["snapshot_timestamp"].eq(latest)].copy()
    best = usable.groupby(group, sort=False)["depth_rank"].transform("min")
    usable = usable.loc[usable["depth_rank"].eq(best)].copy()
    usable["_player"] = usable["player_id"].astype(str)
    selected = usable.sort_values([*group, "_player"], kind="mergesort").drop_duplicates(group).drop(columns=["_player"])
    selected = selected.loc[:, [*schedule.columns, "player_id", *(["player_name"] if "player_name" in selected else [])]]
    if "player_name" not in selected:
        selected["player_name"] = None
    unknown = schedule.merge(selected.loc[:, list(SCHEDULE_KEY)], on=list(SCHEDULE_KEY), how="left", indicator=True)
    unknown = unknown.loc[unknown["_merge"].eq("left_only"), list(schedule.columns)]
    return selected.sort_values([*SCHEDULE_KEY, "player_id"], kind="mergesort").reset_index(drop=True), unknown.reset_index(drop=True)


def _history_with_context(qb_games: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    if qb_games.duplicated(TARGET_KEY).any():
        raise ValueError("Historical quarterback games have duplicate target keys")
    normalized = build_quarterback_dataset(qb_games)
    context = schedules.loc[:, ["season", "week", "game_id", "team", "opponent"]].drop_duplicates()
    if context.duplicated(["season", "week", "game_id", "team"]).any():
        raise ValueError("Schedule rows have conflicting historical team mappings")
    data = normalized.merge(context, on=["season", "week", "game_id", "team", "opponent"], how="inner", validate="many_to_one")
    # An unmapped historic QB row cannot be safely assigned a defense.
    if len(data) != len(normalized):
        raise ValueError("Historical quarterback games are missing schedule opponent alignment")
    for column in ("passing_yards", "passing_attempts"):
        data[column] = pd.to_numeric(data[column], errors="coerce").where(lambda values: np.isfinite(values))
    return data.sort_values(TARGET_KEY, kind="mergesort").reset_index(drop=True)


def _defense_logs(history: pd.DataFrame) -> pd.DataFrame:
    return build_defense_vs_primary_quarterback_logs(identify_primary_quarterbacks(history))


def _history_cutoff(season: int, week: int) -> tuple[int, int]:
    return (season - 1, np.iinfo(np.int64).max) if week == 1 else (season, week)


def _prior(data: pd.DataFrame, entity: str, value: object, season: int, week: int) -> pd.DataFrame:
    return data.loc[(data[entity] == value) & (data["season"] == season) & (data["week"] < week)].sort_values(["season", "week", "game_id"], kind="mergesort")


def _finite(values: pd.Series) -> pd.Series:
    return pd.Series(np.isfinite(pd.to_numeric(values, errors="coerce")), index=values.index)


def _qb_features(prior: pd.DataFrame) -> dict[str, object]:
    last3 = prior.tail(3)
    usable = prior.loc[_finite(prior["passing_yards"])]
    usable_last3 = last3.loc[_finite(last3["passing_yards"])]
    return {
        "qb_season_passing_yards_avg": usable["passing_yards"].mean(),
        "qb_last3_passing_yards_avg": usable_last3["passing_yards"].mean(),
        "qb_season_passing_attempts_avg": prior.loc[_finite(prior["passing_attempts"]), "passing_attempts"].mean(),
        "qb_last3_passing_attempts_avg": last3.loc[_finite(last3["passing_attempts"]), "passing_attempts"].mean(),
        "qb_season_history_games": len(usable), "qb_last3_history_games": len(usable_last3), "qb_missing_history": len(usable) == 0,
    }


def _defense_features(prior: pd.DataFrame, window: pd.DataFrame, defense: object) -> dict[str, object]:
    last3 = prior.tail(3)
    usable = prior.loc[_finite(prior["passing_yards_allowed"])]
    usable_last3 = last3.loc[_finite(last3["passing_yards_allowed"])]
    ranks = window.loc[_finite(window["passing_yards_allowed"])].groupby("defense", sort=True)["passing_yards_allowed"].mean().rank(method="min", ascending=False)
    return {
        "defense_season_passing_yards_allowed_avg": usable["passing_yards_allowed"].mean(),
        "defense_last3_passing_yards_allowed_avg": usable_last3["passing_yards_allowed"].mean(),
        "defense_season_passing_attempts_allowed_avg": prior.loc[_finite(prior["passing_attempts_allowed"]), "passing_attempts_allowed"].mean(),
        "defense_season_history_games": len(usable), "defense_last3_history_games": len(usable_last3), "defense_missing_history": len(usable) == 0,
        "defense_matchup_rank": ranks.get(defense, np.nan),
    }


def _safe_value(value: object) -> float | bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return None
    return float(value)


def _prediction_frame(rows: Iterable[QBPassingYardsWeeklyFeatureRow]) -> pd.DataFrame:
    records = []
    for row in rows:
        record = {"season": row.season, "week": row.week, "game_id": row.game_id, "player_id": row.player_id}
        record.update({name: value for name, value in zip(FEATURE_COLUMNS, row.feature_values, strict=True)})
        records.append(record)
    return pd.DataFrame(records, columns=[*TARGET_KEY, *FEATURE_COLUMNS]).sort_values(TARGET_KEY, kind="mergesort").reset_index(drop=True)
