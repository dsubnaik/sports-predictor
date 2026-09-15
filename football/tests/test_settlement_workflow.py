"""Integration tests for the deliberately invoked settlement workflow."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import sqlite3

import pandas as pd
import pytest

import football.results.settlement_workflow as workflow
from football.data.build_schedule_dataset import OUTPUT_COLUMNS
from football.decisions import (
    DecisionValidationError,
    PendingDecision,
    get_decision,
    initialize_decision_store,
    record_decision,
)
from football.results.batch_settlement import BatchSettlementReport
from football.results.completed_games import (
    CompletedGameConflictError,
    CompletedGameReport,
)


TIME = datetime(2026, 9, 10, 18, 0, tzinfo=timezone.utc)
QB_COLUMNS = ["season", "week", "game_id", "player_id", "passing_yards"]
RB_COLUMNS = ["season", "week", "game_id", "player_id", "rushing_yards"]


@pytest.fixture
def connection():
    database = sqlite3.connect(":memory:")
    initialize_decision_store(database)
    yield database
    database.close()


def decision(**changes):
    values = dict(
        decision_id="qb", position="QB", player_id="qb-1", player_name="QB",
        team="SEA", opponent="NE", season=2026, week=1,
        game_id="2026_01_NE_SEA", market_key="player_pass_yds",
        sportsbook="draftkings", line=Decimal("250.5"), selection="over",
        selected_price=-110, recorded_at=TIME, odds_retrieved_at=TIME,
        research_notes="notes",
    )
    values.update(changes)
    return PendingDecision(**values)


def schedule(rows=None):
    rows = rows or [
        (2026, 1, "2026_01_NE_SEA", "2026-09-09", "20:20", "SEA", "NE", "home", 0, 0),
        (2026, 1, "2026_01_NE_SEA", "2026-09-09", "20:20", "NE", "SEA", "away", 0, 0),
    ]
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def score_event(**changes):
    event = {
        "id": "event-1", "completed": True,
        "commence_time": "2026-09-10T00:20:00Z",
        "home_team": "Seattle Seahawks", "away_team": "New England Patriots",
    }
    event.update(changes)
    return event


def qbs(rows=None):
    rows = rows if rows is not None else [
        {"season": 2026, "week": 1, "game_id": "2026_01_NE_SEA", "player_id": "qb-1", "passing_yards": 275}
    ]
    return pd.DataFrame(rows, columns=QB_COLUMNS)


def rbs(rows=None):
    rows = rows if rows is not None else [
        {"season": 2026, "week": 1, "game_id": "2026_01_NE_SEA", "player_id": "rb-1", "rushing_yards": 75}
    ]
    return pd.DataFrame(rows, columns=RB_COLUMNS)


def run(connection, payload=None, **kwargs):
    return workflow.run_decision_settlement_workflow(
        connection,
        [score_event()] if payload is None else payload,
        schedule(),
        qbs(),
        rbs(),
        **kwargs,
    )


def test_completed_qb_and_rb_events_settle_through_existing_batch(connection):
    record_decision(connection, decision())
    record_decision(connection, decision(
        decision_id="rb", position="RB", player_id="rb-1",
        market_key="player_rush_yds", line=Decimal("80"), selection="under",
    ))

    report = run(connection, settled_at=TIME + timedelta(hours=1))

    assert report.completed_game_report.completed_game_ids == ("2026_01_NE_SEA",)
    assert [(entry.decision_id, entry.decision_status) for entry in report.batch_settlement_report.settled_entries] == [
        ("qb", "win"), ("rb", "win")
    ]


@pytest.mark.parametrize("completed", [False, None, 1])
def test_non_completed_values_leave_decisions_pending(completed, connection):
    original = record_decision(connection, decision())

    report = run(connection, [score_event(completed=completed)])

    assert report.completed_game_report.completed_game_ids == ()
    assert report.batch_settlement_report.entries[0].match_status == "game_pending"
    assert get_decision(connection, "qb") == original


def test_unmatched_and_ambiguous_completed_events_preserve_mapping_diagnostics(connection):
    original = record_decision(connection, decision())
    unmatched = run(connection, [score_event(home_team="Unknown Team")])
    assert unmatched.completed_game_report.diagnostics[0].match_status == "unmatched"
    assert get_decision(connection, "qb") == original

    ambiguous_schedule = pd.concat([
        schedule(),
        schedule([
            (2026, 2, "2026_02_NE_SEA", "2026-09-09", "20:20", "SEA", "NE", "home", 0, 0),
            (2026, 2, "2026_02_NE_SEA", "2026-09-09", "20:20", "NE", "SEA", "away", 0, 0),
        ]),
    ], ignore_index=True)
    report = workflow.run_decision_settlement_workflow(
        connection, [score_event()], ambiguous_schedule, qbs(), rbs()
    )
    assert report.completed_game_report.diagnostics[0].match_status == "ambiguous"
    assert report.batch_settlement_report.entries[0].match_status == "game_pending"


def test_mapping_conflict_prevents_batch_call_and_preserves_decisions(connection, monkeypatch):
    original = record_decision(connection, decision())
    called = False

    def forbidden_batch(*_args, **_kwargs):
        nonlocal called
        called = True
        pytest.fail("batch settlement must not run")

    monkeypatch.setattr(workflow, "settle_decision_batch", forbidden_batch)
    with pytest.raises(CompletedGameConflictError):
        run(connection, [score_event(), score_event(home_team="New England Patriots", away_team="Seattle Seahawks")])

    assert called is False
    assert get_decision(connection, "qb") == original


def test_empty_payload_and_missing_player_result_preserve_batch_diagnostics(connection):
    record_decision(connection, decision())
    empty = run(connection, [])
    assert empty.batch_settlement_report.entries[0].match_status == "game_pending"

    missing = workflow.run_decision_settlement_workflow(
        connection,
        [score_event()],
        schedule(),
        qbs([]),
        rbs(),
        settled_at=TIME + timedelta(hours=1),
    )
    assert missing.batch_settlement_report.entries[0].match_status == "player_result_missing"
    assert get_decision(connection, "qb").status == "pending"


def test_workflow_forwards_timestamp_clock_and_is_safe_to_rerun(connection):
    record_decision(connection, decision())
    calls = []

    def clock():
        calls.append(True)
        return TIME + timedelta(hours=1)

    first = run(connection, clock=clock)
    stored = get_decision(connection, "qb")
    assert calls == [True]
    assert stored.settled_at == TIME + timedelta(hours=1)

    second = run(connection, clock=lambda: pytest.fail("clock must not run"))
    assert second.batch_settlement_report.entries == ()
    assert get_decision(connection, "qb") == stored

    explicit_calls = []
    report = run(
        connection,
        [],
        settled_at=TIME + timedelta(hours=2),
        clock=lambda: explicit_calls.append(True),
    )
    assert report.batch_settlement_report.entries == ()
    assert explicit_calls == []


def test_batch_failure_propagates_with_its_existing_atomic_rollback(connection):
    first = record_decision(connection, decision())
    second = record_decision(connection, decision(
        decision_id="rb", position="RB", player_id="rb-1",
        market_key="player_rush_yds", recorded_at=TIME + timedelta(hours=2),
        odds_retrieved_at=TIME + timedelta(hours=2),
    ))

    with pytest.raises(DecisionValidationError, match="must not precede"):
        run(connection, settled_at=TIME + timedelta(hours=1))

    assert get_decision(connection, "qb") == first
    assert get_decision(connection, "rb") == second


def test_mapper_and_batch_are_called_once_in_order_with_exact_completed_ids(
    connection, monkeypatch
):
    mapping_report = CompletedGameReport(("exact-game",), ())
    batch_report = BatchSettlementReport(())
    calls = []

    def mapper(*_args, **_kwargs):
        calls.append("mapper")
        return mapping_report

    def batch(*args, **kwargs):
        calls.append(("batch", args[3], kwargs["settled_at"], kwargs["clock"]))
        return batch_report

    monkeypatch.setattr(workflow, "identify_completed_nflverse_games", mapper)
    monkeypatch.setattr(workflow, "settle_decision_batch", batch)
    marker_clock = lambda: TIME

    report = run(
        connection,
        settled_at=TIME + timedelta(hours=1),
        clock=marker_clock,
    )

    assert calls == [
        "mapper",
        ("batch", ("exact-game",), TIME + timedelta(hours=1), marker_clock),
    ]
    assert report == workflow.DecisionSettlementWorkflowReport(
        mapping_report,
        batch_report,
    )


def test_workflow_does_not_mutate_inputs_and_connection_remains_usable(connection):
    record_decision(connection, decision())
    payload = [score_event()]
    games, quarterback_games, running_back_games = schedule(), qbs(), rbs()
    payload_before = deepcopy(payload)
    frames_before = [frame.copy(deep=True) for frame in (games, quarterback_games, running_back_games)]

    workflow.run_decision_settlement_workflow(
        connection, payload, games, quarterback_games, running_back_games,
        settled_at=TIME + timedelta(hours=1),
    )

    assert payload == payload_before
    for actual, expected in zip((games, quarterback_games, running_back_games), frames_before):
        pd.testing.assert_frame_equal(actual, expected)
    assert connection.execute("SELECT decision_id FROM decisions").fetchone() == ("qb",)
