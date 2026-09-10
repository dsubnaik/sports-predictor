"""Tests for the isolated NFL scores client without live provider access."""

from copy import deepcopy

import pytest
import requests

import football.odds.scores as scores
from football.odds.player_props import NFL_SPORT_KEY, ODDS_API_BASE_URL
from football.odds.scores import (
    OddsScoresRequestError,
    OddsScoresValidationError,
    fetch_nfl_scores,
)


class FakeResponse:
    def __init__(self, payload=None, *, status_code=None, status_error=None, json_error=None):
        self.payload = payload
        self.status_code = status_code
        self.status_error = status_error
        self.json_error = json_error
        self.json_calls = 0

    def raise_for_status(self):
        if self.status_error is not None:
            raise self.status_error

    def json(self):
        self.json_calls += 1
        if self.json_error is not None:
            raise self.json_error
        return self.payload


def test_fetches_expected_endpoint_params_and_complete_payload_once():
    calls = []
    payload = [
        {"id": "complete", "completed": True, "unknown": {"value": 1}},
        {"id": "live", "completed": False},
    ]
    response = FakeResponse(payload)

    def fake_get(*args, **kwargs):
        calls.append((args, kwargs))
        return response

    result = fetch_nfl_scores(" explicit-key ", request_get=fake_get, timeout=7.5)

    assert calls == [
        (
            (f"{ODDS_API_BASE_URL}/{NFL_SPORT_KEY}/scores",),
            {"params": {"apiKey": "explicit-key", "daysFrom": 3}, "timeout": 7.5},
        )
    ]
    assert response.json_calls == 1
    assert result == payload
    assert result is not payload
    assert result[0] is not payload[0]
    assert result[0]["unknown"] is payload[0]["unknown"]


@pytest.mark.parametrize("days_from", [1, 2, 3])
def test_accepts_provider_days_from_range(days_from):
    calls = []

    def fake_get(*args, **kwargs):
        calls.append(kwargs)
        return FakeResponse([])

    assert fetch_nfl_scores("key", days_from=days_from, request_get=fake_get) == []
    assert calls[0]["params"]["daysFrom"] == days_from


@pytest.mark.parametrize("days_from", [0, -1, 4, True, False, 1.0, "1"])
def test_rejects_invalid_days_from_without_request(days_from):
    with pytest.raises(OddsScoresValidationError, match="days_from"):
        fetch_nfl_scores("key", days_from=days_from, request_get=pytest.fail)


@pytest.mark.parametrize("timeout", [0, -1, True, False, float("nan"), float("inf"), float("-inf")])
def test_rejects_invalid_timeout_without_request(timeout):
    with pytest.raises(OddsScoresValidationError, match="timeout"):
        fetch_nfl_scores("key", timeout=timeout, request_get=pytest.fail)


def test_uses_environment_key_and_explicit_key_takes_precedence(monkeypatch):
    monkeypatch.setattr(scores, "load_dotenv", lambda: None)
    monkeypatch.setenv("ODDS_API_KEY", "environment-key")
    calls = []

    def fake_get(*args, **kwargs):
        calls.append(kwargs)
        return FakeResponse([])

    fetch_nfl_scores(request_get=fake_get)
    fetch_nfl_scores("explicit-key", request_get=fake_get)

    assert [call["params"]["apiKey"] for call in calls] == [
        "environment-key",
        "explicit-key",
    ]


@pytest.mark.parametrize("explicit, environment", [
    (None, None),
    ("", "environment-key"),
    ("   ", "environment-key"),
    (None, ""),
    (None, "  "),
])
def test_rejects_missing_or_blank_keys_without_leaking_credentials(
    monkeypatch, explicit, environment
):
    monkeypatch.setattr(scores, "load_dotenv", lambda: None)
    if environment is None:
        monkeypatch.delenv("ODDS_API_KEY", raising=False)
    else:
        monkeypatch.setenv("ODDS_API_KEY", environment)

    with pytest.raises(OddsScoresValidationError) as error:
        fetch_nfl_scores(explicit, request_get=pytest.fail)

    assert "environment-key" not in str(error.value)
    assert "ODDS_API_KEY" in str(error.value)


@pytest.mark.parametrize("payload", [{}, "not-a-list", ["not-an-object"], [1]])
def test_rejects_invalid_response_boundaries(payload):
    with pytest.raises(OddsScoresValidationError):
        fetch_nfl_scores("key", request_get=lambda *_args, **_kwargs: FakeResponse(payload))


def test_does_not_mutate_provider_payload_or_filter_events():
    payload = [
        {"id": "completed", "completed": True},
        {"id": "uncompleted", "completed": False},
        {"id": "unknown", "additional": "preserved"},
    ]
    before = deepcopy(payload)

    result = fetch_nfl_scores("key", request_get=lambda *_args, **_kwargs: FakeResponse(payload))

    assert result == payload
    assert [event["id"] for event in result] == ["completed", "uncompleted", "unknown"]
    assert payload == before


def test_safe_http_transport_and_json_errors_preserve_chaining_without_secrets():
    key = "credential-not-for-errors"
    transport = requests.ConnectionError("https://provider/?apiKey=credential-not-for-errors")
    with pytest.raises(OddsScoresRequestError) as transport_error:
        fetch_nfl_scores(key, request_get=lambda *_args, **_kwargs: (_ for _ in ()).throw(transport))
    assert str(transport_error.value) == "NFL scores request failed"
    assert key not in str(transport_error.value)
    assert transport_error.value.__cause__ is transport

    http_error = requests.HTTPError("credential-not-for-errors")
    with pytest.raises(OddsScoresRequestError) as http_error_result:
        fetch_nfl_scores(
            key,
            request_get=lambda *_args, **_kwargs: FakeResponse(
                [], status_code=429, status_error=http_error
            ),
        )
    assert str(http_error_result.value) == "NFL scores request failed with HTTP status 429"
    assert key not in str(http_error_result.value)
    assert http_error_result.value.__cause__ is http_error

    json_error = ValueError("credential-not-for-errors")
    with pytest.raises(OddsScoresRequestError) as json_error_result:
        fetch_nfl_scores(
            key,
            request_get=lambda *_args, **_kwargs: FakeResponse([], json_error=json_error),
        )
    assert str(json_error_result.value) == "NFL scores response contained invalid JSON"
    assert key not in str(json_error_result.value)
    assert json_error_result.value.__cause__ is json_error
