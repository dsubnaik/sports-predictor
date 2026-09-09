"""Tests for weekly player-prop odds orchestration without network access."""

from copy import deepcopy

import pandas as pd
import pytest

from football.data.build_schedule_dataset import OUTPUT_COLUMNS as SCHEDULE_COLUMNS
from football.odds.event_discovery import NFL_EVENT_COLUMNS, NFL_EVENT_MATCH_OUTPUT_COLUMNS
from football.odds.player_props import PLAYER_PROP_ODDS_COLUMNS
from football.pipeline.build_weekly_player_prop_odds import (
    WEEKLY_PLAYER_PROP_ODDS_COLUMNS,
    build_weekly_player_prop_odds,
)


def event(event_id="event-1", commence="2026-09-10T00:20:00Z", home="Seattle Seahawks", away="New England Patriots"):
    return {"id": event_id, "commence_time": commence, "home_team": home, "away_team": away}


def schedules(rows=None):
    rows = rows or [
        (2026, 1, "game-1", "2026-09-09", "20:20", "SEA", "NE", "home"),
        (2026, 1, "game-1", "2026-09-09", "20:20", "NE", "SEA", "away"),
    ]
    return pd.DataFrame(rows, columns=SCHEDULE_COLUMNS)


def players(rows=None):
    return pd.DataFrame(
        rows if rows is not None else [("qb-1", "Drake Maye", "NE", "QB"), ("rb-1", "Rhamondre Stevenson", "NE", "RB")],
        columns=["player_id", "player_name", "team", "position"],
    )


def prop_payload(event_id="event-1", markets=("player_pass_yds",), bookmakers=True):
    outcomes = {
        "player_pass_yds": [
            {"name": "Over", "description": "Drake Maye", "price": -110, "point": 220.5},
            {"name": "Under", "description": "Drake Maye", "price": -115, "point": 220.5},
        ],
        "player_rush_yds": [
            {"name": "Over", "description": "Rhamondre Stevenson", "price": -105, "point": 60.5},
            {"name": "Under", "description": "Rhamondre Stevenson", "price": -120, "point": 60.5},
        ],
    }
    return {
        "id": event_id,
        "commence_time": "2026-09-10T00:20:00Z",
        "home_team": "Seattle Seahawks",
        "away_team": "New England Patriots",
        "bookmakers": ([{
            "key": "draftkings", "title": "DraftKings", "last_update": "2026-09-09T00:00:00Z",
            "markets": [{"key": key, "last_update": "2026-09-09T00:01:00Z", "outcomes": outcomes[key]} for key in markets],
        }] if bookmakers else []),
    }


def test_collects_explicit_both_markets_and_combines_exact_schema():
    event_calls, prop_calls = [], []
    source_events = [event()]
    source_props = prop_payload(markets=("player_pass_yds", "player_rush_yds"))
    events_before, props_before = deepcopy(source_events), deepcopy(source_props)

    def fetch_events(*, api_key):
        event_calls.append(api_key)
        return source_events

    def fetch_props(event_id, market_keys, *, api_key):
        prop_calls.append((event_id, market_keys, api_key))
        return source_props

    result = build_weekly_player_prop_odds(
        schedules(), players(), 2026, 1,
        ["player_rush_yds", "player_pass_yds"], "test-key", fetch_events, fetch_props,
    )

    assert event_calls == ["test-key"]
    assert prop_calls == [("event-1", ("player_pass_yds", "player_rush_yds"), "test-key")]
    assert result.normalized_odds.columns.tolist() == PLAYER_PROP_ODDS_COLUMNS
    assert result.event_matches.columns.tolist() == NFL_EVENT_MATCH_OUTPUT_COLUMNS
    assert result.player_matched_odds.columns.tolist() == WEEKLY_PLAYER_PROP_ODDS_COLUMNS
    assert result.player_matched_odds["nflverse_game_id"].tolist() == ["game-1"] * 4
    assert result.player_matched_odds["player_id"].tolist() == ["qb-1", "qb-1", "rb-1", "rb-1"]
    assert result.discovered_event_count == 1
    assert result.matched_selected_week_event_count == result.prop_request_count == 1
    assert result.normalized_odds_row_count == 4
    assert result.matched_player_row_count == 4
    assert source_events == events_before
    assert source_props == props_before


@pytest.mark.parametrize(("market", "expected_name"), [
    ("player_pass_yds", "Drake Maye"),
    ("player_rush_yds", "Rhamondre Stevenson"),
])
def test_never_adds_an_unrequested_market(market, expected_name):
    calls = []
    result = build_weekly_player_prop_odds(
        schedules(), players(), 2026, 1, [market],
        event_fetcher=lambda **_: [event()],
        event_props_fetcher=lambda event_id, market_keys, **_: calls.append((event_id, market_keys)) or prop_payload(markets=market_keys),
    )
    assert calls == [("event-1", (market,))]
    assert result.normalized_odds["market_key"].tolist() == [market, market]
    assert result.player_matched_odds["player_name"].tolist() == [expected_name, expected_name]


def test_fetches_only_selected_week_in_deterministic_order_and_stops_on_failure():
    game_rows = [
        (2026, 1, "game-a", "2026-09-09", "20:20", "SEA", "NE", "home"),
        (2026, 1, "game-a", "2026-09-09", "20:20", "NE", "SEA", "away"),
        (2026, 1, "game-b", "2026-09-13", "13:00", "KC", "BUF", "home"),
        (2026, 1, "game-b", "2026-09-13", "13:00", "BUF", "KC", "away"),
    ]
    source_events = [
        event("later", "2026-09-13T17:00:00Z", "Kansas City Chiefs", "Buffalo Bills"),
        event("first"),
        event("other-week", "2026-09-17T00:20:00Z"),
        event("unknown", home="Unknown Team"),
    ]
    calls = []

    def props(event_id, *_args, **_kwargs):
        calls.append(event_id)
        if event_id == "later":
            raise RuntimeError("provider failure")
        return prop_payload(event_id="first")

    with pytest.raises(RuntimeError, match="provider failure"):
        build_weekly_player_prop_odds(
            schedules(game_rows), players(), 2026, 1, ["player_pass_yds"],
            event_fetcher=lambda **_: source_events, event_props_fetcher=props,
        )
    # kickoff ordering calls first, then later; no further requests can occur.
    assert calls == ["first", "later"]


def test_empty_and_unmatched_discovery_issue_no_prop_requests_and_keep_schemas():
    calls = []
    result = build_weekly_player_prop_odds(
        schedules(), players(), 2026, 1, ["player_pass_yds"],
        event_fetcher=lambda **_: [event(home="Unknown Team")],
        event_props_fetcher=lambda *_args, **_kwargs: calls.append(True),
    )
    assert calls == []
    assert result.event_matches.loc[0, "event_match_status"] == "unmatched"
    assert result.normalized_odds.empty
    assert result.event_matched_odds.columns.tolist() == PLAYER_PROP_ODDS_COLUMNS + [
        "nflverse_game_id", "nflverse_season", "nflverse_week", "nflverse_home_team", "nflverse_away_team", "nflverse_kickoff_time", "event_match_status", "event_match_method", "event_match_candidate_count", "event_match_note",
    ]
    assert result.player_matched_odds.columns.tolist() == WEEKLY_PLAYER_PROP_ODDS_COLUMNS


def test_ambiguous_events_are_retained_in_diagnostics_but_never_request_props():
    ambiguous_schedule = pd.concat([
        schedules(),
        schedules([
            (2026, 2, "game-2", "2026-09-09", "20:20", "SEA", "NE", "home"),
            (2026, 2, "game-2", "2026-09-09", "20:20", "NE", "SEA", "away"),
        ]),
    ], ignore_index=True)
    calls = []
    result = build_weekly_player_prop_odds(
        ambiguous_schedule, players(), 2026, 1, ["player_pass_yds"],
        event_fetcher=lambda **_: [event()],
        event_props_fetcher=lambda *_args, **_kwargs: calls.append(True),
    )
    assert result.event_matches.loc[0, ["event_match_status", "event_match_candidate_count"]].tolist() == ["ambiguous", 2]
    assert result.ambiguous_event_count == 1
    assert result.selected_events.empty
    assert calls == []


def test_no_bookmakers_preserves_selected_event_and_returns_empty_final_table():
    result = build_weekly_player_prop_odds(
        schedules(), players(), 2026, 1, ["player_pass_yds"],
        event_fetcher=lambda **_: [event()],
        event_props_fetcher=lambda *_args, **_kwargs: prop_payload(bookmakers=False),
    )
    assert result.selected_events["event_id"].tolist() == ["event-1"]
    assert result.prop_request_count == 1
    assert result.player_matched_odds.empty
    assert result.player_matched_odds.columns.tolist() == WEEKLY_PLAYER_PROP_ODDS_COLUMNS


@pytest.mark.parametrize("bad_markets", ["player_pass_yds", [], ["bad"]])
def test_invalid_arguments_and_input_schemas_fail_before_event_fetch(bad_markets):
    called = False

    def fetch_events(**_kwargs):
        nonlocal called
        called = True
        return []

    with pytest.raises((TypeError, ValueError)):
        build_weekly_player_prop_odds(
            schedules(), players(), 2026, 1, bad_markets, event_fetcher=fetch_events,
        )
    assert not called
    with pytest.raises(ValueError):
        build_weekly_player_prop_odds(schedules(), players(), True, 1, ["player_pass_yds"], event_fetcher=fetch_events)
    with pytest.raises(ValueError):
        build_weekly_player_prop_odds(schedules().assign(extra=1), players(), 2026, 1, ["player_pass_yds"], event_fetcher=fetch_events)
    assert not called


def test_empty_player_reference_preserves_odds_as_unmatched_and_inputs_are_unchanged():
    source_schedule, source_players = schedules(), players([])
    schedule_before, players_before = source_schedule.copy(deep=True), source_players.copy(deep=True)
    result = build_weekly_player_prop_odds(
        source_schedule, source_players, 2026, 1, ["player_pass_yds"],
        event_fetcher=lambda **_: [event()],
        event_props_fetcher=lambda *_args, **_kwargs: prop_payload(),
    )
    assert result.player_matched_odds["match_status"].tolist() == ["unmatched", "unmatched"]
    assert result.unmatched_player_row_count == 2
    pd.testing.assert_frame_equal(source_schedule, schedule_before)
    pd.testing.assert_frame_equal(source_players, players_before)


def test_ambiguous_players_are_preserved_without_row_multiplication():
    result = build_weekly_player_prop_odds(
        schedules(),
        players([("rb-a", "Rhamondre Stevenson", "NE", "RB"), ("rb-b", "Rhamondre Stevenson", "SEA", "RB")]),
        2026, 1, ["player_rush_yds"],
        event_fetcher=lambda **_: [event()],
        event_props_fetcher=lambda *_args, **_kwargs: prop_payload(markets=("player_rush_yds",)),
    )
    assert len(result.normalized_odds) == len(result.player_matched_odds) == 2
    assert result.player_matched_odds["match_status"].tolist() == ["ambiguous", "ambiguous"]
    assert result.player_matched_odds["match_candidate_count"].tolist() == [2, 2]
    assert result.ambiguous_player_row_count == 2


def test_fetch_and_malformed_payload_failures_propagate_without_extra_requests():
    with pytest.raises(RuntimeError, match="events failed"):
        build_weekly_player_prop_odds(
            schedules(), players(), 2026, 1, ["player_pass_yds"],
            event_fetcher=lambda **_: (_ for _ in ()).throw(RuntimeError("events failed")),
        )

    calls = []
    with pytest.raises((TypeError, ValueError)):
        build_weekly_player_prop_odds(
            schedules(), players(), 2026, 1, ["player_pass_yds"],
            event_fetcher=lambda **_: [event()],
            event_props_fetcher=lambda *_args, **_kwargs: calls.append(True) or {"bad": "payload"},
        )
    assert calls == [True]
