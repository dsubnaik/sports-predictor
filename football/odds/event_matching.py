"""Deterministically match normalized Odds API events to nflverse schedules."""

from __future__ import annotations

import math
from datetime import datetime
from numbers import Integral
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from football.data.build_schedule_dataset import OUTPUT_COLUMNS as SCHEDULE_COLUMNS
from football.odds.player_props import PLAYER_PROP_ODDS_COLUMNS


ODDS_API_TEAM_TO_NFLVERSE = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}
KICKOFF_TIMEZONE = ZoneInfo("America/New_York")
KICKOFF_TOLERANCE = pd.Timedelta(0)
EVENT_MATCH_COLUMNS = [
    "nflverse_game_id", "nflverse_season", "nflverse_week",
    "nflverse_home_team", "nflverse_away_team", "nflverse_kickoff_time",
    "event_match_status", "event_match_method", "event_match_candidate_count",
    "event_match_note",
]
EVENT_MATCH_OUTPUT_COLUMNS = PLAYER_PROP_ODDS_COLUMNS + EVENT_MATCH_COLUMNS


def match_odds_events_to_schedule(odds: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Attach one nflverse regular-season game to each defensible odds event.

    Schedule ``game_time`` is interpreted as America/New_York wall-clock time;
    kickoff is converted to a timezone-aware UTC timestamp before exact matching.
    """

    events = _validate_odds(odds)
    games = _reconstruct_games(schedule)
    if odds.empty:
        return pd.DataFrame(columns=EVENT_MATCH_OUTPUT_COLUMNS)
    resolutions = {event_id: _resolve_event(event, games) for event_id, event in events.items()}
    result = odds.loc[:, PLAYER_PROP_ODDS_COLUMNS].copy().reset_index(drop=True)
    metadata = pd.DataFrame(
        [
            resolutions[_text(event_id, "odds event_id")]
            for event_id in result["event_id"]
        ],
        columns=EVENT_MATCH_COLUMNS,
    )
    return pd.concat([result, metadata], axis=1).reset_index(drop=True)


def _validate_odds(odds: pd.DataFrame) -> dict[str, dict[str, Any]]:
    _require_dataframe(odds, "odds")
    _reject_duplicate_columns(odds, "Odds data")
    if odds.columns.tolist() != PLAYER_PROP_ODDS_COLUMNS:
        raise ValueError("Odds data columns must exactly match PLAYER_PROP_ODDS_COLUMNS in order")
    events: dict[str, dict[str, Any]] = {}
    for index, row in odds.iterrows():
        event_id = _text(row["event_id"], f"odds event_id at index {index}")
        home = _text(row["home_team"], f"odds home_team at index {index}")
        away = _text(row["away_team"], f"odds away_team at index {index}")
        if home == away:
            raise ValueError(f"odds event_id {event_id!r} has identical home and away teams")
        commence = _utc_timestamp(row["commence_time"], f"odds commence_time at index {index}")
        event = {"event_id": event_id, "home": home, "away": away, "commence": commence}
        if event_id in events and event != events[event_id]:
            raise ValueError(f"Conflicting odds event metadata for event_id: {event_id!r}")
        events[event_id] = event
    return events


def _reconstruct_games(schedule: pd.DataFrame) -> list[dict[str, Any]]:
    _require_dataframe(schedule, "schedule")
    _reject_duplicate_columns(schedule, "Schedule data")
    if schedule.columns.tolist() != SCHEDULE_COLUMNS:
        raise ValueError("Schedule data columns must exactly match normalized schedule OUTPUT_COLUMNS in order")
    rows = schedule.loc[:, SCHEDULE_COLUMNS].drop_duplicates().copy()
    validated = []
    for index, row in rows.iterrows():
        season = _positive_integer(row["season"], f"schedule season at index {index}")
        week = _positive_integer(row["week"], f"schedule week at index {index}")
        game_id = _text(row["game_id"], f"schedule game_id at index {index}")
        team = _text(row["team"], f"schedule team at index {index}")
        opponent = _text(row["opponent"], f"schedule opponent at index {index}")
        if team == opponent:
            raise ValueError(f"schedule game_id {game_id!r} has identical team and opponent")
        home_away = row["home_away"]
        if not isinstance(home_away, str) or home_away not in {"home", "away"}:
            raise ValueError(f"schedule home_away at index {index} must be 'home' or 'away'")
        date = _date(row["game_date"], f"schedule game_date at index {index}")
        time = _time(row["game_time"], f"schedule game_time at index {index}")
        validated.append({"season": season, "week": week, "game_id": game_id, "game_date": date,
                          "game_time": time, "team": team, "opponent": opponent, "home_away": home_away})
    games = []
    for game_id, group in pd.DataFrame(validated).groupby("game_id", sort=True) if validated else []:
        if len(group) != 2 or set(group["home_away"]) != {"home", "away"}:
            raise ValueError(f"schedule game_id {game_id!r} must contain one home and one away row")
        home = group.loc[group["home_away"].eq("home")].iloc[0]
        away = group.loc[group["home_away"].eq("away")].iloc[0]
        shared = ["season", "week", "game_date", "game_time"]
        if any(home[field] != away[field] for field in shared):
            raise ValueError(f"Conflicting schedule rows for game_id: {game_id!r}")
        if home["team"] != away["opponent"] or away["team"] != home["opponent"]:
            raise ValueError(f"Nonreciprocal home/away schedule rows for game_id: {game_id!r}")
        local = pd.Timestamp(f"{home['game_date']} {home['game_time']}").tz_localize(KICKOFF_TIMEZONE, ambiguous="raise", nonexistent="raise")
        games.append({"game_id": game_id, "season": home["season"], "week": home["week"],
                      "home": home["team"], "away": away["team"], "kickoff": local.tz_convert("UTC")})
    return games


def _resolve_event(event: dict[str, Any], games: list[dict[str, Any]]) -> dict[str, Any]:
    home = ODDS_API_TEAM_TO_NFLVERSE.get(event["home"])
    away = ODDS_API_TEAM_TO_NFLVERSE.get(event["away"])
    base = {"nflverse_home_team": home if home else pd.NA, "nflverse_away_team": away if away else pd.NA}
    if not home or not away:
        return _unmatched(base, "Unknown Odds API team name")
    candidates = [game for game in games if game["home"] == home and game["away"] == away and abs(game["kickoff"] - event["commence"]) <= KICKOFF_TOLERANCE]
    if len(candidates) == 1:
        game = candidates[0]
        return {**base, "nflverse_game_id": game["game_id"], "nflverse_season": game["season"], "nflverse_week": game["week"], "nflverse_kickoff_time": game["kickoff"], "event_match_status": "matched", "event_match_method": "mapped_teams_and_exact_utc_kickoff", "event_match_candidate_count": 1, "event_match_note": "Unique mapped-team and kickoff match"}
    if not candidates:
        return _unmatched(base, "No mapped-team schedule game at kickoff")
    return {**base, "nflverse_game_id": pd.NA, "nflverse_season": pd.NA, "nflverse_week": pd.NA, "nflverse_kickoff_time": pd.NA, "event_match_status": "ambiguous", "event_match_method": pd.NA, "event_match_candidate_count": len(candidates), "event_match_note": "Multiple mapped-team schedule games at kickoff"}


def _unmatched(base: dict[str, Any], note: str) -> dict[str, Any]:
    return {**base, "nflverse_game_id": pd.NA, "nflverse_season": pd.NA, "nflverse_week": pd.NA, "nflverse_kickoff_time": pd.NA, "event_match_status": "unmatched", "event_match_method": pd.NA, "event_match_candidate_count": 0, "event_match_note": note}


def _utc_timestamp(value: Any, field: str) -> pd.Timestamp:
    if not isinstance(value, str): raise ValueError(f"{field} must be a timezone-aware ISO timestamp")
    try: timestamp = pd.Timestamp(value)
    except (TypeError, ValueError): raise ValueError(f"{field} must be a timezone-aware ISO timestamp") from None
    if timestamp.tzinfo is None: raise ValueError(f"{field} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _date(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must use YYYY-MM-DD")
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"{field} must use YYYY-MM-DD") from None
    return value


def _time(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must use HH:MM")
    try:
        datetime.strptime(value, "%H:%M")
    except ValueError:
        raise ValueError(f"{field} must use HH:MM") from None
    return value


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1: raise ValueError(f"{field} must be a positive integer")
    return int(value)


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip(): raise ValueError(f"{field} must be nonblank text")
    return value.strip()


def _require_dataframe(value: Any, name: str) -> None:
    if not isinstance(value, pd.DataFrame): raise TypeError(f"{name} must be a pandas DataFrame")


def _reject_duplicate_columns(data: pd.DataFrame, label: str) -> None:
    duplicates = data.columns[data.columns.duplicated()].tolist()
    if duplicates: raise ValueError(f"{label} contains duplicate columns: {duplicates}")
