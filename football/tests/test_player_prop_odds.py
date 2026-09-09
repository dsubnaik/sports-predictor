"""Tests for isolated NFL player-prop odds ingestion and normalization."""

from copy import deepcopy

import pandas as pd
import pytest

import football.odds.player_props as player_props
from football.odds.player_props import (
    NFL_SPORT_KEY,
    ODDS_API_BASE_URL,
    PLAYER_PROP_ODDS_COLUMNS,
    fetch_event_player_props,
    fetch_nfl_events,
    normalize_player_prop_odds,
)


def make_outcome(name="Over", description="Josh Allen", price=-115, point=249.5):
    return {"name": name, "description": description, "price": price, "point": point}


def make_market(key="player_pass_yds", outcomes=None, last_update="2026-09-08T12:01:00Z"):
    return {"key": key, "last_update": last_update, "outcomes": outcomes if outcomes is not None else [make_outcome(), make_outcome("Under", price=-105)]}


def make_bookmaker(key="draftkings", markets=None, title="DraftKings", last_update="2026-09-08T12:00:00Z"):
    return {"key": key, "title": title, "last_update": last_update, "markets": markets if markets is not None else [make_market()]}


def make_event(bookmakers=None):
    return {"id": "event-1", "commence_time": "2026-09-10T00:20:00Z", "home_team": "Kansas City Chiefs", "away_team": "Buffalo Bills", "bookmakers": bookmakers if bookmakers is not None else [make_bookmaker()]}


class FakeResponse:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise self.error

    def json(self):
        return self.payload


def test_normalizes_both_markets_bookmakers_and_exact_column_order():
    payload = make_event(bookmakers=[
        make_bookmaker(markets=[make_market("player_pass_yds"), make_market("player_rush_yds", [make_outcome("Over", "James Cook", -110, 62.5), make_outcome("Under", "James Cook", -120, 62.5)], "2026-09-08T12:02:00Z")]),
        make_bookmaker("fanduel", title="FanDuel", last_update="2026-09-08T12:03:00Z"),
    ])
    result = normalize_player_prop_odds(payload)
    assert result.columns.tolist() == PLAYER_PROP_ODDS_COLUMNS
    assert set(result["market_key"]) == {"player_pass_yds", "player_rush_yds"}
    assert set(result["bookmaker_key"]) == {"draftkings", "fanduel"}
    assert result.loc[(result.bookmaker_key == "draftkings") & (result.market_key == "player_pass_yds") & (result.outcome_name == "Over"), "price"].item() == -115
    assert result.loc[(result.bookmaker_key == "draftkings") & (result.market_key == "player_pass_yds") & (result.outcome_name == "Under"), "price"].item() == -105
    assert result.loc[result.bookmaker_key == "fanduel", "bookmaker_last_update"].iloc[0] == "2026-09-08T12:03:00Z"
    assert result.loc[result.market_key == "player_rush_yds", "market_last_update"].iloc[0] == "2026-09-08T12:02:00Z"


def test_normalization_is_deterministic_and_does_not_mutate_payload():
    payload = [make_event(bookmakers=[make_bookmaker("fanduel"), make_bookmaker("draftkings")])]
    before = deepcopy(payload)
    forward = normalize_player_prop_odds(payload)
    reverse_payload = deepcopy(payload)
    reverse_payload[0]["bookmakers"].reverse()
    reverse = normalize_player_prop_odds(reverse_payload)
    pd.testing.assert_frame_equal(forward, reverse)
    assert payload == before


@pytest.mark.parametrize("payload", [[], make_event(bookmakers=[])])
def test_empty_valid_payloads_return_complete_empty_schema(payload):
    result = normalize_player_prop_odds(payload)
    assert result.empty
    assert result.columns.tolist() == PLAYER_PROP_ODDS_COLUMNS


def test_normalizer_keeps_missing_update_timestamps_as_missing_metadata():
    payload = make_event()
    bookmaker = payload["bookmakers"][0]
    bookmaker.pop("last_update")
    bookmaker["markets"][0]["last_update"] = None
    result = normalize_player_prop_odds(payload)
    assert result["bookmaker_last_update"].isna().all()
    assert result["market_last_update"].isna().all()


def test_fetch_events_uses_expected_endpoint_timeout_and_http_behavior():
    calls = []
    def fake_get(*args, **kwargs):
        calls.append((args, kwargs)); return FakeResponse([])
    assert fetch_nfl_events("key", request_get=fake_get) == []
    assert calls == [((f"{ODDS_API_BASE_URL}/{NFL_SPORT_KEY}/events",), {"params": {"apiKey": "key"}, "timeout": 10})]
    with pytest.raises(RuntimeError, match="bad response"):
        fetch_nfl_events("key", request_get=lambda *_, **__: FakeResponse(error=RuntimeError("bad response")))


def test_fetch_props_validates_and_uses_deterministic_parameters():
    calls = []
    def fake_get(*args, **kwargs):
        calls.append((args, kwargs)); return FakeResponse({"id": "event-1"})
    fetch_event_player_props(" event-1 ", ["player_rush_yds", "player_pass_yds", "player_pass_yds"], "key", request_get=fake_get, timeout=7)
    assert calls == [((f"{ODDS_API_BASE_URL}/{NFL_SPORT_KEY}/events/event-1/odds",), {"params": {"apiKey": "key", "regions": "us", "markets": "player_pass_yds,player_rush_yds", "oddsFormat": "american"}, "timeout": 7})]


@pytest.mark.parametrize("api_key", [None, "", "  "])
def test_missing_api_key_is_rejected_before_request(api_key, monkeypatch):
    called = False
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.setattr(player_props, "load_dotenv", lambda: None)
    def fake_get(**_):
        nonlocal called; called = True
    with pytest.raises(ValueError, match="ODDS_API_KEY"):
        fetch_nfl_events(api_key, request_get=fake_get)
    assert not called


@pytest.mark.parametrize("event_id", ["", "  ", None])
def test_blank_event_id_is_rejected(event_id):
    with pytest.raises(ValueError, match="event_id"):
        fetch_event_player_props(event_id, ["player_pass_yds"], "key")


@pytest.mark.parametrize("markets, error", [([], ValueError), ("player_pass_yds", TypeError), (["bad_market"], ValueError)])
def test_requested_market_validation(markets, error):
    with pytest.raises(error):
        fetch_event_player_props("event", markets, "key")


@pytest.mark.parametrize("payload", ["bad", 1, [{"id": "event"}]])
def test_normalizer_rejects_wrong_top_level_or_malformed_event(payload):
    with pytest.raises((TypeError, ValueError)):
        normalize_player_prop_odds(payload)


@pytest.mark.parametrize("container_path", [
    ("bookmakers", 0),
    ("bookmakers", 0, "markets", 0),
    ("bookmakers", 0, "markets", 0, "outcomes", 0),
])
def test_normalizer_rejects_non_mapping_nested_records(container_path):
    payload = make_event()
    target = payload
    for part in container_path[:-1]:
        target = target[part]
    target[container_path[-1]] = "not a mapping"
    with pytest.raises(TypeError, match="must be a mapping"):
        normalize_player_prop_odds(payload)


@pytest.mark.parametrize("path, value", [
    (("id",), ""), (("home_team",), None), (("away_team",), ""), (("bookmakers",), {}),
    (("bookmakers", 0, "key"), ""), (("bookmakers", 0, "markets"), {}),
    (("bookmakers", 0, "markets", 0, "key"), "player_reception_yds"),
    (("bookmakers", 0, "markets", 0, "outcomes"), {}), (("bookmakers", 0, "markets", 0, "outcomes", 0, "description"), ""),
    (("bookmakers", 0, "markets", 0, "outcomes", 0, "name"), "Yes"),
    (("bookmakers", 0, "markets", 0, "outcomes", 0, "price"), True), (("bookmakers", 0, "markets", 0, "outcomes", 0, "point"), "249.5"),
])
def test_normalizer_rejects_malformed_required_records_without_mutation(path, value):
    payload = make_event()
    target = payload
    for part in path[:-1]: target = target[part]
    target[path[-1]] = value
    before = deepcopy(payload)
    with pytest.raises((TypeError, ValueError)):
        normalize_player_prop_odds(payload)
    assert payload == before


def test_normalizer_does_not_mutate_invalid_input():
    payload = make_event(); payload["bookmakers"][0]["markets"][0]["outcomes"][0]["point"] = True
    before = deepcopy(payload)
    with pytest.raises(ValueError, match="point"):
        normalize_player_prop_odds(payload)
    assert payload == before


def test_exact_duplicates_collapse_and_conflicting_natural_keys_raise():
    duplicate = make_event(bookmakers=[make_bookmaker(markets=[make_market(outcomes=[make_outcome(), make_outcome()])])])
    assert len(normalize_player_prop_odds(duplicate)) == 1
    conflict = deepcopy(duplicate)
    conflict["bookmakers"][0]["markets"][0]["outcomes"][1]["price"] = -120
    with pytest.raises(ValueError, match="Conflicting player-prop quotes"):
        normalize_player_prop_odds(conflict)
