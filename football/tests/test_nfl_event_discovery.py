"""Tests for pure lightweight NFL event discovery transforms."""

from copy import deepcopy

import pandas as pd
import pytest

from football.data.build_schedule_dataset import OUTPUT_COLUMNS as SCHEDULE_COLUMNS
from football.odds.event_discovery import (
    NFL_EVENT_COLUMNS, NFL_EVENT_MATCH_OUTPUT_COLUMNS, match_nfl_events_to_schedule,
    normalize_nfl_events, select_matched_nfl_events,
)


def payload(event_id="event", commence="2026-09-10T00:20:00Z", home="Seattle Seahawks", away="New England Patriots"):
    return {"id": event_id, "commence_time": commence, "home_team": home, "away_team": away}


def schedule(rows=None):
    rows = rows or [
        (2026, 1, "game", "2026-09-09", "20:20", "SEA", "NE", "home"),
        (2026, 1, "game", "2026-09-09", "20:20", "NE", "SEA", "away"),
    ]
    return pd.DataFrame(
        [(*row, pd.NA, pd.NA) if len(row) == 8 else row for row in rows],
        columns=SCHEDULE_COLUMNS,
    )


def test_normalizes_sorts_deduplicates_and_does_not_mutate_payload():
    source = [payload("b"), payload("a"), payload("a")]
    before = deepcopy(source)
    result = normalize_nfl_events(source)
    assert result.columns.tolist() == NFL_EVENT_COLUMNS
    assert result["event_id"].tolist() == ["a", "b"]
    assert str(result.loc[0, "commence_time"].tz) == "UTC"
    assert source == before


@pytest.mark.parametrize("bad", ["bad", 1, ["bad"], [payload("a"), payload("a", home="New England Patriots", away="Seattle Seahawks")], [payload(commence="bad")], [payload(commence="2026-09-09T20:20:00")], [payload(home="Seattle Seahawks", away="Seattle Seahawks")]])
def test_normalizer_rejects_bad_payloads(bad):
    with pytest.raises((TypeError, ValueError)):
        normalize_nfl_events(bad)


def test_event_matching_selection_and_shared_rules():
    events = normalize_nfl_events([payload(), payload("rams", home="Los Angeles Rams", away="Seattle Seahawks")])
    games = pd.concat([schedule(), schedule([(2026, 2, "rams", "2026-09-09", "20:20", "LA", "SEA", "home"), (2026, 2, "rams", "2026-09-09", "20:20", "SEA", "LA", "away")])], ignore_index=True)
    matched = match_nfl_events_to_schedule(events, games)
    assert matched.columns.tolist() == NFL_EVENT_MATCH_OUTPUT_COLUMNS
    assert matched["event_match_status"].tolist() == ["matched", "matched"]
    selected = select_matched_nfl_events(matched, 2026, 1)
    assert selected["event_id"].tolist() == ["event"]
    assert select_matched_nfl_events(matched, 2026, 3).columns.tolist() == NFL_EVENT_MATCH_OUTPUT_COLUMNS


def test_empty_unknown_schema_and_input_immutability():
    empty = normalize_nfl_events([])
    assert empty.columns.tolist() == NFL_EVENT_COLUMNS
    unmatched = match_nfl_events_to_schedule(normalize_nfl_events([payload(home="Unknown", away="Seattle Seahawks")]), pd.DataFrame(columns=SCHEDULE_COLUMNS))
    assert unmatched.loc[0, "event_match_status"] == "unmatched"
    events, games = normalize_nfl_events([payload()]), schedule()
    events_before, games_before = events.copy(deep=True), games.copy(deep=True)
    match_nfl_events_to_schedule(events, games)
    pd.testing.assert_frame_equal(events, events_before)
    pd.testing.assert_frame_equal(games, games_before)
    with pytest.raises(ValueError): match_nfl_events_to_schedule(events.assign(extra=1), games)
    with pytest.raises(ValueError): select_matched_nfl_events(events, 2026, 1)
    with pytest.raises(ValueError): select_matched_nfl_events(match_nfl_events_to_schedule(events, games), True, 1)
