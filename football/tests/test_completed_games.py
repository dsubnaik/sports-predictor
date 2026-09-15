"""Tests for pure completed Odds API event mapping."""

from copy import deepcopy

import pandas as pd
import pytest

from football.data.build_schedule_dataset import OUTPUT_COLUMNS
from football.results.completed_games import (
    CompletedGameConflictError,
    CompletedGameValidationError,
    identify_completed_nflverse_games,
)


def schedule(rows=None):
    rows = rows or [
        (2026, 1, "2026_01_NE_SEA", "2026-09-09", "20:20", "SEA", "NE", "home", 0, 0),
        (2026, 1, "2026_01_NE_SEA", "2026-09-09", "20:20", "NE", "SEA", "away", 0, 0),
    ]
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def score_event(**changes):
    event = {
        "id": "event-1",
        "completed": True,
        "commence_time": "2026-09-10T00:20:00Z",
        "home_team": "Seattle Seahawks",
        "away_team": "New England Patriots",
        "scores": [{"name": "Seattle Seahawks", "score": "0"}],
    }
    event.update(changes)
    return event


def test_completed_event_maps_to_one_game_without_interpreting_zero_or_tied_scores():
    report = identify_completed_nflverse_games([score_event()], schedule())

    assert report.completed_game_ids == ("2026_01_NE_SEA",)
    assert report.diagnostics == ()


@pytest.mark.parametrize("completed", [False, None, 1, "true"])
def test_only_literal_boolean_true_is_eligible(completed):
    report = identify_completed_nflverse_games(
        [score_event(completed=completed, id=None, commence_time=None)],
        schedule(),
    )

    assert report.completed_game_ids == ()
    assert report.diagnostics == ()


def test_malformed_incomplete_event_does_not_block_completed_mapping():
    report = identify_completed_nflverse_games(
        [score_event(), score_event(id=None, completed=False, home_team=None)],
        schedule(),
    )

    assert report.completed_game_ids == ("2026_01_NE_SEA",)


def test_completed_events_require_existing_exact_team_and_kickoff_match():
    report = identify_completed_nflverse_games(
        [
            score_event(id="bad-team", home_team="Unknown Team"),
            score_event(id="bad-kickoff", commence_time="2026-09-10T00:21:00Z"),
        ],
        schedule(),
    )

    assert report.completed_game_ids == ()
    assert [(item.provider_event_id, item.match_status) for item in report.diagnostics] == [
        ("bad-kickoff", "unmatched"),
        ("bad-team", "unmatched"),
    ]
    assert all(item.diagnostic for item in report.diagnostics)


def test_ambiguous_completed_event_is_diagnostic_not_a_game_id():
    ambiguous_schedule = pd.concat(
        [
            schedule(),
            schedule([
                (2026, 2, "2026_02_NE_SEA", "2026-09-09", "20:20", "SEA", "NE", "home", 1, 1),
                (2026, 2, "2026_02_NE_SEA", "2026-09-09", "20:20", "NE", "SEA", "away", 1, 1),
            ]),
        ],
        ignore_index=True,
    )

    report = identify_completed_nflverse_games([score_event()], ambiguous_schedule)

    assert report.completed_game_ids == ()
    assert [(item.provider_event_id, item.match_status) for item in report.diagnostics] == [
        ("event-1", "ambiguous")
    ]


def test_identical_duplicates_collapse_but_conflicting_provider_ids_raise():
    assert identify_completed_nflverse_games(
        [score_event(), score_event()], schedule()
    ).completed_game_ids == ("2026_01_NE_SEA",)

    with pytest.raises(CompletedGameConflictError, match="provider event IDs conflict"):
        identify_completed_nflverse_games(
            [score_event(), score_event(home_team="New England Patriots", away_team="Seattle Seahawks")],
            schedule(),
        )


def test_distinct_completed_provider_ids_for_one_game_raise_conflict():
    with pytest.raises(CompletedGameConflictError, match="distinct completed"):
        identify_completed_nflverse_games(
            [score_event(id="a"), score_event(id="b")], schedule()
        )


@pytest.mark.parametrize("field, value", [
    ("id", " "),
    ("home_team", None),
    ("away_team", " "),
    ("commence_time", "not-a-timestamp"),
])
def test_completed_events_require_public_event_matching_fields(field, value):
    with pytest.raises(CompletedGameValidationError, match="eligible completed"):
        identify_completed_nflverse_games([score_event(**{field: value})], schedule())


def test_empty_payload_and_empty_schedule_are_safe_but_schema_is_required():
    assert identify_completed_nflverse_games([], schedule()).completed_game_ids == ()
    report = identify_completed_nflverse_games([score_event()], pd.DataFrame(columns=OUTPUT_COLUMNS))
    assert report.completed_game_ids == ()
    assert report.diagnostics[0].match_status == "unmatched"
    with pytest.raises(CompletedGameValidationError, match="normalized schedule"):
        identify_completed_nflverse_games([], schedule().drop(columns=["home_score"]))


def test_ids_diagnostics_and_inputs_are_deterministic_and_immutable():
    other_game = [
        (2026, 2, "2026_02_BUF_MIA", "2026-09-16", "13:00", "MIA", "BUF", "home", 3, 3),
        (2026, 2, "2026_02_BUF_MIA", "2026-09-16", "13:00", "BUF", "MIA", "away", 3, 3),
    ]
    games = pd.concat([schedule(), schedule(other_game)], ignore_index=True)
    events = [
        score_event(id="z", commence_time="2026-09-16T17:00:00Z", home_team="Miami Dolphins", away_team="Buffalo Bills"),
        score_event(id="a"),
        score_event(id="unmatched", home_team="Unknown Team"),
    ]
    events_before = deepcopy(events)
    games_before = games.copy(deep=True)

    forward = identify_completed_nflverse_games(events, games)
    reverse = identify_completed_nflverse_games(list(reversed(events)), games.iloc[::-1])

    assert forward == reverse
    assert forward.completed_game_ids == ("2026_01_NE_SEA", "2026_02_BUF_MIA")
    assert [item.provider_event_id for item in forward.diagnostics] == ["unmatched"]
    assert events == events_before
    pd.testing.assert_frame_equal(games, games_before)


def test_payload_boundary_rejects_non_objects():
    with pytest.raises(CompletedGameValidationError, match="scores_payload"):
        identify_completed_nflverse_games(["bad"], schedule())
