"""Selected-game player-prop orchestration tests without network access."""

from copy import deepcopy
from datetime import datetime, timezone

import pandas as pd
import pytest

from football.data.build_schedule_dataset import OUTPUT_COLUMNS as SCHEDULE_COLUMNS
from football.odds.player_props import (
    DEFAULT_FOOTBALL_PROP_BOOKMAKERS,
    EventPlayerPropsFetchResult,
    OddsApiQuotaMetadata,
)
from football.pipeline.build_selected_game_player_prop_odds import (
    SELECTED_GAME_PROP_MARKETS,
    build_selected_game_player_prop_odds,
)


def schedule(rows=None):
    rows = rows or [
        (2026, 2, "game-a", "2026-09-17", "20:15", "BUF", "DET", "home"),
        (2026, 2, "game-a", "2026-09-17", "20:15", "DET", "BUF", "away"),
        (2026, 2, "game-b", "2026-09-20", "13:00", "KC", "NE", "home"),
        (2026, 2, "game-b", "2026-09-20", "13:00", "NE", "KC", "away"),
    ]
    return pd.DataFrame([(*row, pd.NA, pd.NA) for row in rows], columns=SCHEDULE_COLUMNS)


def qbs():
    return pd.DataFrame([
        ("game-a", "BUF", "qb-allen", "Josh Allen"),
        ("game-a", "DET", "qb-goff", "Jared Goff"),
        ("game-b", "NE", "qb-other", "Josh Allen"),
    ], columns=["game_id", "team", "expected_player_id", "expected_player_name"])


def rbs():
    return pd.DataFrame([
        ("game-a", "BUF", "rb-cook", "James Cook III"),
        ("game-a", "DET", "rb-gibbs", "Jahmyr Gibbs"),
        ("game-b", "NE", "rb-other", "James Cook III"),
    ], columns=["game_id", "team", "player_id", "player_name"])


def events():
    return [
        {"id": "event-b", "commence_time": "2026-09-20T17:00:00Z", "home_team": "Kansas City Chiefs", "away_team": "New England Patriots"},
        {"id": "event-a", "commence_time": "2026-09-18T00:15:00Z", "home_team": "Buffalo Bills", "away_team": "Detroit Lions"},
    ]


def payload():
    def outcomes(name, point, over, under):
        return [
            {"name": "Over", "description": name, "point": point, "price": over},
            {"name": "Under", "description": name, "point": point, "price": under},
        ]
    return {
        "id": "event-a", "commence_time": "2026-09-18T00:15:00Z",
        "home_team": "Buffalo Bills", "away_team": "Detroit Lions",
        "bookmakers": [
            {"key": "draftkings", "title": "DraftKings", "last_update": "2026-09-17T12:00:00Z", "markets": [
                {"key": "player_pass_yds", "last_update": "2026-09-17T12:01:00Z", "outcomes": outcomes("Josh Allen", 246.5, -110, -110)},
                {"key": "player_rush_yds", "last_update": "2026-09-17T12:01:00Z", "outcomes": outcomes("James Cook", 62.5, -105, -115)},
                {"key": "player_rush_yds", "last_update": "2026-09-17T12:02:00Z", "outcomes": outcomes("James Cook", 72.5, 125, -150)},
            ]},
            {"key": "fanduel", "title": "FanDuel", "last_update": "2026-09-17T12:03:00Z", "markets": [
                {"key": "player_pass_yds", "last_update": "2026-09-17T12:03:00Z", "outcomes": outcomes("Jared Goff", 259.5, -112, -108)},
                {"key": "player_rush_yds", "last_update": "2026-09-17T12:03:00Z", "outcomes": outcomes("Jahmyr Gibbs", 67.5, -110, -110)},
            ]},
        ],
    }


def build(**kwargs):
    kwargs.setdefault("event_fetcher", lambda **_kwargs: events())
    kwargs.setdefault("retrieved_at_factory", lambda: datetime(2026, 9, 18, tzinfo=timezone.utc))
    return build_selected_game_player_prop_odds(
        "game-a", schedule(), qbs(), rbs(), 2026, 2,
        **kwargs,
    )


def test_selected_game_makes_one_paid_request_for_exact_markets_and_bookmakers():
    calls = []

    def fetch_props(event_id, markets, **kwargs):
        calls.append((event_id, markets, kwargs))
        return EventPlayerPropsFetchResult(payload(), 2, 12, 488)

    result = build(event_props_fetcher=fetch_props)

    assert calls == [("event-a", SELECTED_GAME_PROP_MARKETS, {
        "api_key": None, "bookmakers": DEFAULT_FOOTBALL_PROP_BOOKMAKERS,
        "include_quota_metadata": True,
    })]
    assert result.game_id == "game-a"
    assert result.markets == ("player_pass_yds", "player_rush_yds")
    assert result.bookmakers == DEFAULT_FOOTBALL_PROP_BOOKMAKERS
    assert result.quota == OddsApiQuotaMetadata(2, 12, 488)
    rows = result.player_matched_odds.to_frame()
    assert set(rows["nflverse_game_id"]) == {"game-a"}
    assert set(rows["bookmaker_key"]) <= set(DEFAULT_FOOTBALL_PROP_BOOKMAKERS)
    assert set(rows.loc[rows["match_status"].eq("matched"), "player_id"]) == {"qb-allen", "qb-goff", "rb-cook", "rb-gibbs"}
    assert rows.loc[(rows["player_id"] == "rb-cook") & (rows["outcome_name"] == "Over"), ["point", "price"]].values.tolist() == [[62.5, -105], [72.5, 125]]


def test_missing_or_unmatched_selected_game_never_makes_paid_request():
    calls = []
    with pytest.raises(ValueError, match="No normalized schedule game"):
        build_selected_game_player_prop_odds(
            "missing", schedule(), qbs(), rbs(), 2026, 2,
            event_fetcher=lambda **_kwargs: events(),
            event_props_fetcher=lambda *_args, **_kwargs: calls.append(True),
        )
    assert calls == []


def test_ambiguous_schedule_game_is_rejected_before_free_or_paid_fetches():
    calls = []
    conflicting_home = schedule().iloc[[0]].copy()
    conflicting_home.loc[:, "home_score"] = 99
    ambiguous_schedule = pd.concat([schedule().iloc[:2], conflicting_home], ignore_index=True)
    with pytest.raises(ValueError, match="must contain one home and one away row"):
        build_selected_game_player_prop_odds(
            "game-a", ambiguous_schedule, qbs(), rbs(), 2026, 2,
            event_fetcher=lambda **_kwargs: calls.append("free"),
            event_props_fetcher=lambda *_args, **_kwargs: calls.append("paid"),
        )
    assert calls == []
    result = build(
        event_fetcher=lambda **_kwargs: [{**events()[0], "home_team": "Unknown Team"}],
        event_props_fetcher=lambda *_args, **_kwargs: calls.append(True),
    )
    assert result.matched_event is None
    assert result.player_matched_odds.rows == ()
    assert calls == []


def test_ambiguous_event_match_never_makes_paid_request():
    calls = []
    duplicate = events()[1]
    result = build(
        event_fetcher=lambda **_kwargs: [duplicate, {**duplicate, "id": "event-a-copy"}],
        event_props_fetcher=lambda *_args, **_kwargs: calls.append(True),
    )
    assert result.matched_event is None
    assert calls == []


def test_snapshot_is_deterministic_and_does_not_mutate_inputs_or_expose_internal_frames():
    source_schedule, source_qbs, source_rbs, source_payload = schedule(), qbs(), rbs(), payload()
    before = (source_schedule.copy(deep=True), source_qbs.copy(deep=True), source_rbs.copy(deep=True), deepcopy(source_payload))
    first = build(event_props_fetcher=lambda *_args, **_kwargs: source_payload)
    reversed_payload = deepcopy(source_payload)
    reversed_payload["bookmakers"].reverse()
    second = build(event_props_fetcher=lambda *_args, **_kwargs: reversed_payload)
    pd.testing.assert_frame_equal(first.player_matched_odds.to_frame(), second.player_matched_odds.to_frame())
    copy = first.player_matched_odds.to_frame()
    copy.loc[:, "player_id"] = "changed"
    assert "changed" not in first.player_matched_odds.to_frame()["player_id"].tolist()
    pd.testing.assert_frame_equal(source_schedule, before[0])
    pd.testing.assert_frame_equal(source_qbs, before[1])
    pd.testing.assert_frame_equal(source_rbs, before[2])
    assert source_payload == before[3]
