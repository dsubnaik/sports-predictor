"""Fetch and normalize current NFL passing- and rushing-yard prop odds.

Normalized quotes use the natural identity ``(event_id, bookmaker_key,
market_key, player_name, outcome_name)``.  ``outcome_name`` is canonicalized
to ``Over`` or ``Under``; sportsbook player names are otherwise preserved
apart from surrounding whitespace.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping, Sequence
from numbers import Real
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


NFL_SPORT_KEY = "americanfootball_nfl"
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4/sports"
REQUEST_TIMEOUT_SECONDS = 10
SUPPORTED_PLAYER_PROP_MARKETS = frozenset({"player_pass_yds", "player_rush_yds"})
PLAYER_PROP_ODDS_COLUMNS = [
    "event_id",
    "commence_time",
    "home_team",
    "away_team",
    "bookmaker_key",
    "bookmaker_title",
    "bookmaker_last_update",
    "market_key",
    "market_last_update",
    "player_name",
    "outcome_name",
    "price",
    "point",
]

_NATURAL_KEY = [
    "event_id",
    "bookmaker_key",
    "market_key",
    "player_name",
    "outcome_name",
]


def fetch_nfl_events(
    api_key: str | None = None,
    *,
    request_get: Callable[..., Any] | None = None,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> Any:
    """Return the current NFL event payload from The Odds API."""

    key = _resolve_api_key(api_key)
    response = (request_get or requests.get)(
        f"{ODDS_API_BASE_URL}/{NFL_SPORT_KEY}/events",
        params={"apiKey": key},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def fetch_event_player_props(
    event_id: str,
    market_keys: Sequence[str],
    api_key: str | None = None,
    *,
    request_get: Callable[..., Any] | None = None,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> Any:
    """Return raw requested supported player-prop markets for one NFL event."""

    event_identifier = _required_text(event_id, "event_id")
    markets = _validate_requested_market_keys(market_keys)
    key = _resolve_api_key(api_key)
    response = (request_get or requests.get)(
        f"{ODDS_API_BASE_URL}/{NFL_SPORT_KEY}/events/{event_identifier}/odds",
        params={
            "apiKey": key,
            "regions": "us",
            "markets": ",".join(markets),
            "oddsFormat": "american",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def normalize_player_prop_odds(payload: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Return a deterministic quote table from raw Odds API event payloads.

    An empty list or events without bookmakers/outcomes returns an empty frame
    with ``PLAYER_PROP_ODDS_COLUMNS``. Malformed populated records are rejected.
    """

    events = _validate_event_payload(payload)
    rows: list[dict[str, Any]] = []
    for event_index, event in enumerate(events):
        context = f"event at index {event_index}"
        event_values = {
            "event_id": _required_mapping_text(event, "id", context),
            "commence_time": _required_mapping_text(event, "commence_time", context),
            "home_team": _required_mapping_text(event, "home_team", context),
            "away_team": _required_mapping_text(event, "away_team", context),
        }
        bookmakers = _required_list(event, "bookmakers", context)
        for bookmaker_index, bookmaker in enumerate(bookmakers):
            bookmaker_context = f"{context} bookmaker at index {bookmaker_index}"
            if not isinstance(bookmaker, Mapping):
                raise TypeError(f"{bookmaker_context} must be a mapping")
            bookmaker_values = {
                "bookmaker_key": _required_mapping_text(bookmaker, "key", bookmaker_context),
                "bookmaker_title": _required_mapping_text(bookmaker, "title", bookmaker_context),
                "bookmaker_last_update": _optional_mapping_text(
                    bookmaker, "last_update", bookmaker_context
                ),
            }
            markets = _required_list(bookmaker, "markets", bookmaker_context)
            for market_index, market in enumerate(markets):
                market_context = f"{bookmaker_context} market at index {market_index}"
                if not isinstance(market, Mapping):
                    raise TypeError(f"{market_context} must be a mapping")
                market_key = _required_mapping_text(market, "key", market_context)
                if market_key not in SUPPORTED_PLAYER_PROP_MARKETS:
                    raise ValueError(f"{market_context}.key is unsupported: {market_key!r}")
                outcomes = _required_list(market, "outcomes", market_context)
                market_last_update = _optional_mapping_text(
                    market, "last_update", market_context
                )
                for outcome_index, outcome in enumerate(outcomes):
                    outcome_context = f"{market_context} outcome at index {outcome_index}"
                    if not isinstance(outcome, Mapping):
                        raise TypeError(f"{outcome_context} must be a mapping")
                    outcome_name = _canonical_outcome_name(
                        _required_mapping_text(outcome, "name", outcome_context), outcome_context
                    )
                    rows.append({
                        **event_values,
                        **bookmaker_values,
                        "market_key": market_key,
                        "market_last_update": market_last_update,
                        "player_name": _required_mapping_text(outcome, "description", outcome_context),
                        "outcome_name": outcome_name,
                        "price": _numeric_value(outcome.get("price"), f"{outcome_context}.price"),
                        "point": _numeric_value(outcome.get("point"), f"{outcome_context}.point"),
                    })

    if not rows:
        return pd.DataFrame(columns=PLAYER_PROP_ODDS_COLUMNS)
    normalized = pd.DataFrame(rows, columns=PLAYER_PROP_ODDS_COLUMNS)
    _reject_conflicting_quotes(normalized)
    normalized = normalized.drop_duplicates().sort_values(
        _NATURAL_KEY, kind="mergesort"
    )
    return normalized.loc[:, PLAYER_PROP_ODDS_COLUMNS].reset_index(drop=True)


def _resolve_api_key(api_key: str | None) -> str:
    if api_key is None:
        load_dotenv()
        api_key = os.getenv("ODDS_API_KEY")
    return _required_text(api_key, "ODDS_API_KEY")


def _validate_requested_market_keys(market_keys: Sequence[str]) -> list[str]:
    if isinstance(market_keys, (str, bytes)) or not isinstance(market_keys, Sequence):
        raise TypeError("market_keys must be a non-string sequence of supported market keys")
    if not market_keys:
        raise ValueError("market_keys must contain at least one supported market key")
    normalized = [_required_text(key, "market_keys item") for key in market_keys]
    unsupported = sorted(set(normalized).difference(SUPPORTED_PLAYER_PROP_MARKETS))
    if unsupported:
        raise ValueError(f"Unsupported player-prop market keys: {unsupported}")
    return sorted(set(normalized))


def _validate_event_payload(payload: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        return [payload]
    if isinstance(payload, (str, bytes)) or not isinstance(payload, Sequence):
        raise TypeError("Player-prop payload must be an event mapping or a sequence of event mappings")
    events = list(payload)
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise TypeError(f"event at index {index} must be a mapping")
    return events


def _required_list(mapping: Mapping[str, Any], field: str, context: str) -> list[Any]:
    value = mapping.get(field)
    if not isinstance(value, list):
        raise TypeError(f"{context}.{field} must be a list")
    return value


def _required_mapping_text(mapping: Mapping[str, Any], field: str, context: str) -> str:
    return _required_text(mapping.get(field), f"{context}.{field}")


def _optional_mapping_text(
    mapping: Mapping[str, Any], field: str, context: str
) -> str | object:
    """Return optional API metadata text without manufacturing a string value."""

    value = mapping.get(field)
    if value is None:
        return pd.NA
    if not isinstance(value, str):
        raise ValueError(f"{context}.{field} must be text or missing")
    normalized = value.strip()
    return normalized if normalized else pd.NA


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be nonblank text")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must be nonblank text")
    return normalized


def _canonical_outcome_name(value: str, context: str) -> str:
    if value not in {"Over", "Under"}:
        raise ValueError(f"{context}.name must be exactly 'Over' or 'Under'")
    return value


def _numeric_value(value: Any, field: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite numeric value and not a boolean")
    return value


def _reject_conflicting_quotes(data: pd.DataFrame) -> None:
    duplicate_keys = data.duplicated(_NATURAL_KEY, keep=False)
    if not duplicate_keys.any():
        return
    duplicates = data.loc[duplicate_keys]
    conflicts = duplicates.groupby(_NATURAL_KEY, dropna=False, sort=True).filter(
        lambda group: len(group.drop_duplicates()) > 1
    )
    if conflicts.empty:
        return
    identities = (
        conflicts.loc[:, _NATURAL_KEY]
        .drop_duplicates()
        .sort_values(_NATURAL_KEY, kind="mergesort")
        .to_dict("records")
    )
    raise ValueError(f"Conflicting player-prop quotes for natural keys: {identities}")
