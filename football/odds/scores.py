"""Fetch raw NFL score events from The Odds API without interpreting finality.

This boundary validates only the provider response container.  It deliberately
preserves both completed and non-completed events for a later pure component to
interpret.  Returned event mappings are shallow copies; nested provider values
are otherwise preserved unchanged.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping
from numbers import Integral, Real
from typing import Any

import requests
from dotenv import load_dotenv

from football.odds.player_props import (
    NFL_SPORT_KEY,
    ODDS_API_BASE_URL,
    REQUEST_TIMEOUT_SECONDS,
)


class OddsScoresError(Exception):
    """Base exception for NFL scores-client errors."""


class OddsScoresValidationError(ValueError, OddsScoresError):
    """Raised when scores-client input or response structure is invalid."""


class OddsScoresRequestError(RuntimeError, OddsScoresError):
    """Raised for expected safe transport, HTTP, or JSON response failures."""


def fetch_nfl_scores(
    api_key: str | None = None,
    *,
    days_from: int = 3,
    request_get: Callable[..., Any] | None = None,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> list[dict[str, object]]:
    """Return the complete raw NFL scores event list from The Odds API.

    ``days_from`` requests recently completed events and is limited by the
    provider to one through three days. This function neither filters nor
    interprets an event's ``completed`` field.
    """

    key = _resolve_api_key(api_key)
    validated_days_from = _days_from(days_from)
    validated_timeout = _timeout(timeout)
    endpoint = f"{ODDS_API_BASE_URL}/{NFL_SPORT_KEY}/scores"

    try:
        response = (request_get or requests.get)(
            endpoint,
            params={"apiKey": key, "daysFrom": validated_days_from},
            timeout=validated_timeout,
        )
    except requests.RequestException as error:
        raise OddsScoresRequestError("NFL scores request failed") from error

    try:
        response.raise_for_status()
    except requests.RequestException as error:
        status_code = _safe_status_code(response)
        message = (
            f"NFL scores request failed with HTTP status {status_code}"
            if status_code is not None
            else "NFL scores request failed"
        )
        raise OddsScoresRequestError(message) from error

    try:
        payload = response.json()
    except ValueError as error:
        raise OddsScoresRequestError("NFL scores response contained invalid JSON") from error

    if not isinstance(payload, list):
        raise OddsScoresValidationError("NFL scores response must be a list")
    if not all(isinstance(event, Mapping) for event in payload):
        raise OddsScoresValidationError(
            "NFL scores response items must be objects"
        )
    return [dict(event) for event in payload]


def _resolve_api_key(api_key: str | None) -> str:
    if api_key is None:
        load_dotenv()
        api_key = os.getenv("ODDS_API_KEY")
    if not isinstance(api_key, str) or not api_key.strip():
        raise OddsScoresValidationError("ODDS_API_KEY must be nonblank text")
    return api_key.strip()


def _days_from(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value not in {1, 2, 3}:
        raise OddsScoresValidationError("days_from must be an integer from 1 through 3")
    return int(value)


def _timeout(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise OddsScoresValidationError(
            "timeout must be a finite numeric value greater than zero"
        )
    return float(value)


def _safe_status_code(response: object) -> int | None:
    value = getattr(response, "status_code", None)
    if isinstance(value, bool) or not isinstance(value, Integral):
        return None
    return int(value)
