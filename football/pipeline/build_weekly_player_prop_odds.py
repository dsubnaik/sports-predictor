"""Build normalized NFL player-prop odds for one nflverse schedule week."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import pandas as pd

from football.odds.event_discovery import (
    NFL_EVENT_COLUMNS,
    NFL_EVENT_MATCH_OUTPUT_COLUMNS,
    match_nfl_events_to_schedule,
    normalize_nfl_events,
    select_matched_nfl_events,
)
from football.odds.event_matching import EVENT_MATCH_COLUMNS, EVENT_MATCH_OUTPUT_COLUMNS, match_odds_events_to_schedule
from football.odds.player_matching import (
    PLAYER_PROP_MATCH_COLUMNS,
    PLAYER_PROP_MATCH_OUTPUT_COLUMNS,
    match_player_prop_odds,
)
from football.odds.player_props import (
    PLAYER_PROP_ODDS_COLUMNS,
    SUPPORTED_PLAYER_PROP_MARKETS,
    fetch_event_player_props,
    fetch_nfl_events,
    normalize_player_prop_odds,
)


Fetcher = Callable[..., Any]

WEEKLY_PLAYER_PROP_ODDS_COLUMNS = (
    PLAYER_PROP_ODDS_COLUMNS + EVENT_MATCH_COLUMNS + PLAYER_PROP_MATCH_COLUMNS
)


@dataclass(frozen=True)
class WeeklyPlayerPropOddsResult:
    """Tables produced while collecting player props for a single NFL week.

    Count properties intentionally describe rows where applicable, so an Over and
    Under pair is not presented as a player count.
    """

    normalized_events: pd.DataFrame
    event_matches: pd.DataFrame
    selected_events: pd.DataFrame
    normalized_odds: pd.DataFrame
    event_matched_odds: pd.DataFrame
    player_matched_odds: pd.DataFrame

    @property
    def discovered_event_count(self) -> int:
        return len(self.normalized_events)

    @property
    def matched_selected_week_event_count(self) -> int:
        return len(self.selected_events)

    @property
    def unmatched_event_count(self) -> int:
        return int((self.event_matches["event_match_status"] == "unmatched").sum())

    @property
    def ambiguous_event_count(self) -> int:
        return int((self.event_matches["event_match_status"] == "ambiguous").sum())

    @property
    def prop_request_count(self) -> int:
        return len(self.selected_events)

    @property
    def normalized_odds_row_count(self) -> int:
        return len(self.normalized_odds)

    @property
    def matched_player_row_count(self) -> int:
        return int((self.player_matched_odds["match_status"] == "matched").sum())

    @property
    def unmatched_player_row_count(self) -> int:
        return int((self.player_matched_odds["match_status"] == "unmatched").sum())

    @property
    def ambiguous_player_row_count(self) -> int:
        return int((self.player_matched_odds["match_status"] == "ambiguous").sum())


def build_weekly_player_prop_odds(
    schedule: pd.DataFrame,
    players: pd.DataFrame,
    report_season: int,
    report_week: int,
    market_keys: Sequence[str],
    api_key: str | None = None,
    event_fetcher: Fetcher | None = None,
    event_props_fetcher: Fetcher | None = None,
) -> WeeklyPlayerPropOddsResult:
    """Collect and match only the requested player-prop markets for one week.

    ``event_fetcher`` is called once.  ``event_props_fetcher`` is called once for
    each uniquely selected, already schedule-matched event, in kickoff order.
    Both injected callables use the public fetch-function argument conventions.
    """

    _validate_positive_integer(report_season, "report_season")
    _validate_positive_integer(report_week, "report_week")
    requested_markets = _normalize_market_keys(market_keys)

    # Validate caller-owned tables before consuming an API credit.  The public
    # matchers own their schema and reference-data validation contracts.
    empty_events = pd.DataFrame(columns=NFL_EVENT_COLUMNS)
    match_nfl_events_to_schedule(empty_events, schedule)
    empty_odds = pd.DataFrame(columns=PLAYER_PROP_ODDS_COLUMNS)
    match_player_prop_odds(empty_odds, players)

    selected_event_fetcher = event_fetcher or fetch_nfl_events
    selected_props_fetcher = event_props_fetcher or fetch_event_player_props

    event_payload = selected_event_fetcher(api_key=api_key)
    normalized_events = normalize_nfl_events(event_payload)
    event_matches = match_nfl_events_to_schedule(normalized_events, schedule)
    selected_events = select_matched_nfl_events(
        event_matches,
        report_season,
        report_week,
    )

    prop_payloads: list[Any] = []
    for event_id in selected_events["event_id"].tolist():
        prop_payloads.append(
            selected_props_fetcher(
                event_id,
                requested_markets,
                api_key=api_key,
            )
        )

    normalized_odds = _normalize_prop_payloads(prop_payloads)
    event_matched_odds = match_odds_events_to_schedule(normalized_odds, schedule)
    player_matched_odds = match_player_prop_odds(normalized_odds, players)
    final_odds = _combine_match_results(event_matched_odds, player_matched_odds)

    return WeeklyPlayerPropOddsResult(
        normalized_events=normalized_events,
        event_matches=event_matches,
        selected_events=selected_events,
        normalized_odds=normalized_odds,
        event_matched_odds=event_matched_odds,
        player_matched_odds=final_odds,
    )


def _normalize_market_keys(market_keys: Sequence[str]) -> tuple[str, ...]:
    if isinstance(market_keys, (str, bytes)) or not isinstance(market_keys, Sequence):
        raise TypeError("market_keys must be a non-string collection of market keys")
    if not market_keys:
        raise ValueError("market_keys must contain at least one supported market key")

    normalized: list[str] = []
    for market_key in market_keys:
        if not isinstance(market_key, str) or not market_key.strip():
            raise ValueError("market_keys must contain nonblank text market keys")
        normalized.append(market_key.strip())

    unsupported = sorted(set(normalized) - SUPPORTED_PLAYER_PROP_MARKETS)
    if unsupported:
        raise ValueError(f"unsupported player-prop market keys: {', '.join(unsupported)}")
    return tuple(sorted(set(normalized)))


def _validate_positive_integer(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _normalize_prop_payloads(payloads: Sequence[Any]) -> pd.DataFrame:
    """Flatten supported single-event or event-collection fetch responses."""

    event_payloads: list[Any] = []
    for payload in payloads:
        if isinstance(payload, Mapping):
            event_payloads.append(payload)
        elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
            event_payloads.extend(payload)
        else:
            # Leave malformed data visible to the established normalizer.
            event_payloads.append(payload)
    return normalize_player_prop_odds(event_payloads)


def _combine_match_results(
    event_matched_odds: pd.DataFrame,
    player_matched_odds: pd.DataFrame,
) -> pd.DataFrame:
    """Combine independent enrichments after proving their odds rows align."""

    if list(event_matched_odds.columns) != EVENT_MATCH_OUTPUT_COLUMNS:
        raise ValueError("event matcher returned an unexpected schema")
    if list(player_matched_odds.columns) != PLAYER_PROP_MATCH_OUTPUT_COLUMNS:
        raise ValueError("player matcher returned an unexpected schema")

    event_odds = event_matched_odds.loc[:, PLAYER_PROP_ODDS_COLUMNS].reset_index(drop=True)
    player_odds = player_matched_odds.loc[:, PLAYER_PROP_ODDS_COLUMNS].reset_index(drop=True)
    if not event_odds.equals(player_odds):
        raise ValueError("event and player matching outputs do not align to the same odds rows")

    combined = pd.concat(
        [
            event_matched_odds.loc[:, PLAYER_PROP_ODDS_COLUMNS + EVENT_MATCH_COLUMNS].reset_index(drop=True),
            player_matched_odds.loc[:, PLAYER_PROP_MATCH_COLUMNS].reset_index(drop=True),
        ],
        axis=1,
    )
    return combined.loc[:, WEEKLY_PLAYER_PROP_ODDS_COLUMNS].reset_index(drop=True)
