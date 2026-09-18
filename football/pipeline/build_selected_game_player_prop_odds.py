"""Retrieve and match player props for one already selected nflverse game.

This module deliberately keeps event discovery (free) separate from the one
paid event-odds request.  It is a backend boundary only; callers decide when a
user has deliberately requested the paid snapshot.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Integral
from typing import Any

import pandas as pd

from football.data.build_schedule_dataset import OUTPUT_COLUMNS as SCHEDULE_COLUMNS
from football.odds.event_discovery import (
    NFL_EVENT_COLUMNS,
    NFL_EVENT_MATCH_OUTPUT_COLUMNS,
    match_nfl_events_to_schedule,
    normalize_nfl_events,
)
from football.odds.event_matching import EVENT_MATCH_OUTPUT_COLUMNS, match_odds_events_to_schedule
from football.odds.player_matching import match_player_prop_odds
from football.odds.player_props import (
    DEFAULT_FOOTBALL_PROP_BOOKMAKERS,
    EventPlayerPropsFetchResult,
    OddsApiQuotaMetadata,
    PLAYER_PROP_ODDS_COLUMNS,
    fetch_event_player_props,
    fetch_nfl_events,
    normalize_player_prop_odds,
)
from football.pipeline.build_weekly_player_prop_odds import (
    WEEKLY_PLAYER_PROP_ODDS_COLUMNS,
    _combine_match_results,
)


Fetcher = Callable[..., Any]
SELECTED_GAME_PROP_MARKETS = ("player_pass_yds", "player_rush_yds")


@dataclass(frozen=True)
class ImmutableOddsTable:
    """Immutable row records with a safe DataFrame reconstruction boundary."""

    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, columns: Sequence[str]) -> "ImmutableOddsTable":
        selected = frame.loc[:, list(columns)].copy(deep=True).reset_index(drop=True)
        return cls(tuple(columns), tuple(tuple(row) for row in selected.itertuples(index=False, name=None)))

    def to_frame(self) -> pd.DataFrame:
        """Return a new caller-owned DataFrame, never an internal mutable table."""

        return pd.DataFrame(list(self.rows), columns=list(self.columns)).copy(deep=True)


@dataclass(frozen=True)
class SelectedGamePlayerPropOddsResult:
    """Immutable selected-game odds snapshot and matching diagnostics."""

    game_id: str
    matched_event: tuple[tuple[str, Any], ...] | None
    normalized_events: ImmutableOddsTable
    event_matches: ImmutableOddsTable
    normalized_odds: ImmutableOddsTable
    event_matched_odds: ImmutableOddsTable
    player_matched_odds: ImmutableOddsTable
    retrieved_at: datetime
    markets: tuple[str, ...]
    bookmakers: tuple[str, ...]
    quota: OddsApiQuotaMetadata


def build_selected_game_player_prop_odds(
    game_id: str,
    schedule: pd.DataFrame,
    expected_qbs: pd.DataFrame,
    expected_rbs: pd.DataFrame,
    report_season: int,
    report_week: int,
    *,
    api_key: str | None = None,
    event_fetcher: Fetcher | None = None,
    event_props_fetcher: Fetcher | None = None,
    retrieved_at_factory: Callable[[], datetime] | None = None,
) -> SelectedGamePlayerPropOddsResult:
    """Fetch both supported player markets for exactly one matched game.

    ``expected_qbs`` accepts the established QB expected-player columns
    (``expected_player_id``/``expected_player_name``) or the generic player
    reference names.  ``expected_rbs`` uses generic player reference names.
    Both inputs must retain ``game_id`` and ``team`` so references are scoped
    to the selected game before player matching.
    """

    selected_game_id = _required_text(game_id, "game_id")
    _positive_integer(report_season, "report_season")
    _positive_integer(report_week, "report_week")
    selected_schedule = _select_schedule_game(schedule, selected_game_id, report_season, report_week)
    players = _selected_players(expected_qbs, expected_rbs, selected_game_id, selected_schedule)

    # Validate all local caller data before even the free discovery request.
    match_nfl_events_to_schedule(pd.DataFrame(columns=NFL_EVENT_COLUMNS), selected_schedule)
    match_player_prop_odds(pd.DataFrame(columns=PLAYER_PROP_ODDS_COLUMNS), players)

    discovered_payload = (event_fetcher or fetch_nfl_events)(api_key=api_key)
    normalized_events = normalize_nfl_events(discovered_payload)
    event_matches = match_nfl_events_to_schedule(normalized_events, selected_schedule)
    matched = event_matches.loc[
        event_matches["event_match_status"].eq("matched")
        & event_matches["nflverse_game_id"].eq(selected_game_id)
    ].copy()

    empty_odds = pd.DataFrame(columns=PLAYER_PROP_ODDS_COLUMNS)
    empty_event_odds = pd.DataFrame(columns=EVENT_MATCH_OUTPUT_COLUMNS)
    empty_player_odds = pd.DataFrame(columns=WEEKLY_PLAYER_PROP_ODDS_COLUMNS)
    now = (retrieved_at_factory or _utc_now)()
    if len(matched) != 1:
        return _result(
            selected_game_id, None, normalized_events, event_matches, empty_odds,
            empty_event_odds, empty_player_odds, now,
        )

    event_id = _required_text(matched.iloc[0]["event_id"], "matched event_id")
    fetched = (event_props_fetcher or fetch_event_player_props)(
        event_id,
        SELECTED_GAME_PROP_MARKETS,
        api_key=api_key,
        bookmakers=DEFAULT_FOOTBALL_PROP_BOOKMAKERS,
        include_quota_metadata=True,
    )
    if isinstance(fetched, EventPlayerPropsFetchResult):
        payload, quota = fetched.payload, fetched.quota
    else:
        # Injectable legacy-shaped fetchers remain supported, with explicit
        # unavailable quota values rather than invented zero values.
        payload, quota = fetched, OddsApiQuotaMetadata(None, None, None)
    normalized_odds = normalize_player_prop_odds(payload)
    event_matched_odds = match_odds_events_to_schedule(normalized_odds, selected_schedule)
    player_matches = match_player_prop_odds(normalized_odds, players)
    player_matched_odds = _combine_match_results(event_matched_odds, player_matches)
    event_identity = tuple((column, matched.iloc[0][column]) for column in NFL_EVENT_MATCH_OUTPUT_COLUMNS)
    return _result(
        selected_game_id, event_identity, normalized_events, event_matches,
        normalized_odds, event_matched_odds, player_matched_odds, now, quota,
    )


def _result(
    game_id: str,
    matched_event: tuple[tuple[str, Any], ...] | None,
    normalized_events: pd.DataFrame,
    event_matches: pd.DataFrame,
    normalized_odds: pd.DataFrame,
    event_matched_odds: pd.DataFrame,
    player_matched_odds: pd.DataFrame,
    retrieved_at: datetime,
    quota: OddsApiQuotaMetadata | None = None,
) -> SelectedGamePlayerPropOddsResult:
    return SelectedGamePlayerPropOddsResult(
        game_id=game_id,
        matched_event=matched_event,
        normalized_events=ImmutableOddsTable.from_frame(normalized_events, NFL_EVENT_COLUMNS),
        event_matches=ImmutableOddsTable.from_frame(event_matches, NFL_EVENT_MATCH_OUTPUT_COLUMNS),
        normalized_odds=ImmutableOddsTable.from_frame(normalized_odds, PLAYER_PROP_ODDS_COLUMNS),
        event_matched_odds=ImmutableOddsTable.from_frame(event_matched_odds, EVENT_MATCH_OUTPUT_COLUMNS),
        player_matched_odds=ImmutableOddsTable.from_frame(player_matched_odds, WEEKLY_PLAYER_PROP_ODDS_COLUMNS),
        retrieved_at=retrieved_at,
        markets=SELECTED_GAME_PROP_MARKETS,
        bookmakers=DEFAULT_FOOTBALL_PROP_BOOKMAKERS,
        quota=quota or OddsApiQuotaMetadata(None, None, None),
    )


def _select_schedule_game(
    schedule: pd.DataFrame,
    game_id: str,
    report_season: int,
    report_week: int,
) -> pd.DataFrame:
    if not isinstance(schedule, pd.DataFrame) or schedule.columns.tolist() != SCHEDULE_COLUMNS:
        raise ValueError("schedule must use normalized schedule OUTPUT_COLUMNS in order")
    selected = schedule.loc[schedule["game_id"].eq(game_id), SCHEDULE_COLUMNS].copy(deep=True)
    if selected.empty:
        raise ValueError(f"No normalized schedule game exists for game_id: {game_id!r}")
    if not selected["season"].eq(report_season).all() or not selected["week"].eq(report_week).all():
        raise ValueError("selected game does not belong to the requested report season and week")
    # The established matcher performs the authoritative reciprocal orientation
    # validation.  It will reject malformed or ambiguous team-game contexts.
    return selected.reset_index(drop=True)


def _selected_players(
    expected_qbs: pd.DataFrame,
    expected_rbs: pd.DataFrame,
    game_id: str,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    teams = set(schedule["team"].tolist())
    qb = _reference_rows(expected_qbs, game_id, teams, "QB", "expected_player_id", "expected_player_name")
    rb = _reference_rows(expected_rbs, game_id, teams, "RB", "player_id", "player_name")
    return pd.concat([qb, rb], ignore_index=True).sort_values(
        ["position", "team", "player_id", "player_name"], kind="mergesort", na_position="last"
    ).reset_index(drop=True)


def _reference_rows(
    source: pd.DataFrame,
    game_id: str,
    teams: set[Any],
    position: str,
    preferred_id: str,
    preferred_name: str,
) -> pd.DataFrame:
    if not isinstance(source, pd.DataFrame):
        raise TypeError(f"expected {position} participants must be a pandas DataFrame")
    id_column = preferred_id if preferred_id in source.columns else "player_id"
    name_column = preferred_name if preferred_name in source.columns else "player_name"
    needed = {"game_id", "team", id_column, name_column}
    missing = sorted(needed.difference(source.columns))
    if missing:
        raise ValueError(f"expected {position} participants are missing required columns: {missing}")
    rows = source.loc[
        source["game_id"].eq(game_id) & source["team"].isin(teams),
        [id_column, name_column, "team"],
    ].copy(deep=True).rename(columns={id_column: "player_id", name_column: "player_name"})
    rows["position"] = position
    return rows.loc[:, ["player_id", "player_name", "team", "position"]]


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonblank text")
    return value.strip()


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
