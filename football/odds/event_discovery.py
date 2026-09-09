"""Pure discovery transforms for lightweight NFL Odds API event lists."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import pandas as pd

from football.odds.event_matching import (
    EVENT_MATCH_COLUMNS,
    _positive_integer,
    _reconstruct_games,
    _resolve_event,
    _text,
    _utc_timestamp,
)


NFL_EVENT_COLUMNS = ["event_id", "commence_time", "home_team", "away_team"]
NFL_EVENT_MATCH_OUTPUT_COLUMNS = NFL_EVENT_COLUMNS + EVENT_MATCH_COLUMNS


def normalize_nfl_events(
    payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Normalize one event mapping or a non-string sequence of event mappings."""

    if isinstance(payload, Mapping):
        events = [payload]
    elif isinstance(payload, (str, bytes)) or not isinstance(payload, Sequence):
        raise TypeError("NFL event payload must be an event mapping or sequence of event mappings")
    else:
        events = list(payload)
    records = []
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise TypeError(f"event at index {index} must be a mapping")
        event_id = _text(event.get("id"), f"event at index {index}.id")
        home = _text(event.get("home_team"), f"event at index {index}.home_team")
        away = _text(event.get("away_team"), f"event at index {index}.away_team")
        if home == away:
            raise ValueError(f"event at index {index} has identical home and away teams")
        commence = _utc_timestamp(event.get("commence_time"), f"event at index {index}.commence_time")
        records.append({"event_id": event_id, "commence_time": commence, "home_team": home, "away_team": away})
    if not records:
        return pd.DataFrame(columns=NFL_EVENT_COLUMNS)
    data = pd.DataFrame(records, columns=NFL_EVENT_COLUMNS)
    conflicts = data.loc[data.duplicated("event_id", keep=False)].groupby("event_id", sort=True).filter(lambda group: len(group.drop_duplicates()) > 1)
    if not conflicts.empty:
        raise ValueError("Conflicting NFL event metadata for event_id values: " + str(sorted(conflicts["event_id"].unique())))
    return data.drop_duplicates().sort_values(["commence_time", "event_id"], kind="mergesort").reset_index(drop=True)


def match_nfl_events_to_schedule(events: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Attach existing schedule-match diagnostics to one normalized row per event."""

    _require_event_schema(events, NFL_EVENT_COLUMNS, "NFL event data")
    games = _reconstruct_games(schedule)
    if events.empty:
        return pd.DataFrame(columns=NFL_EVENT_MATCH_OUTPUT_COLUMNS)
    result = events.loc[:, NFL_EVENT_COLUMNS].copy().reset_index(drop=True)
    metadata = []
    for _, row in result.iterrows():
        event = {"event_id": _text(row["event_id"], "event_id"), "home": _text(row["home_team"], "home_team"), "away": _text(row["away_team"], "away_team"), "commence": _utc_timestamp(row["commence_time"].isoformat() if isinstance(row["commence_time"], pd.Timestamp) else row["commence_time"], "commence_time")}
        if event["home"] == event["away"]:
            raise ValueError("event_id has identical home and away teams")
        metadata.append(_resolve_event(event, games))
    return pd.concat([result, pd.DataFrame(metadata, columns=EVENT_MATCH_COLUMNS)], axis=1).reset_index(drop=True)


def select_matched_nfl_events(matched_events: pd.DataFrame, report_season: int, report_week: int) -> pd.DataFrame:
    """Return only uniquely matched events for the requested nflverse week."""

    _require_event_schema(matched_events, NFL_EVENT_MATCH_OUTPUT_COLUMNS, "Matched NFL event data")
    season = _positive_integer(report_season, "report_season")
    week = _positive_integer(report_week, "report_week")
    selected = matched_events.loc[(matched_events["event_match_status"].eq("matched")) & (matched_events["nflverse_season"].eq(season)) & (matched_events["nflverse_week"].eq(week))].copy()
    return selected.sort_values(["commence_time", "event_id"], kind="mergesort").reset_index(drop=True)


def _require_event_schema(data: Any, columns: list[str], label: str) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"{label} must be a pandas DataFrame")
    duplicates = data.columns[data.columns.duplicated()].tolist()
    if duplicates:
        raise ValueError(f"{label} contains duplicate columns: {duplicates}")
    if data.columns.tolist() != columns:
        raise ValueError(f"{label} columns must exactly match the canonical schema in order")
