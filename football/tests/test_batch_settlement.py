"""Integration tests for atomic decision-result batch settlement."""

import sqlite3
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pandas as pd
import pytest

import football.results.batch_settlement as batch_settlement
from football.decisions import (
    DecisionValidationError,
    PendingDecision,
    SettlementConflictError,
    get_decision,
    initialize_decision_store,
    record_decision,
    settle_decision,
)
from football.results.decision_result_matching import (
    ConflictingResultDataError,
    DecisionResultValidationError,
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


def pending_decision(**changes):
    values = dict(
        decision_id="qb", position="QB", player_id="qb-1", player_name="QB", team="KC",
        opponent="LAC", season=2026, week=1, game_id="2026_01_KC_LAC",
        market_key="player_pass_yds", sportsbook="draftkings", line=Decimal("250.5"),
        selection="over", selected_price=-110, recorded_at=TIME, odds_retrieved_at=TIME,
        research_notes="notes",
    )
    values.update(changes)
    return PendingDecision(**values)


def qbs(rows=None):
    if rows is None:
        rows = [{"season": 2026, "week": 1, "game_id": "2026_01_KC_LAC", "player_id": "qb-1", "passing_yards": 275}]
    return pd.DataFrame(rows, columns=QB_COLUMNS)


def rbs(rows=None):
    if rows is None:
        rows = [{"season": 2026, "week": 1, "game_id": "2026_01_KC_LAC", "player_id": "rb-1", "rushing_yards": 75}]
    return pd.DataFrame(rows, columns=RB_COLUMNS)


def settle(connection, qbs_frame=None, rbs_frame=None, completed=("2026_01_KC_LAC",), **kwargs):
    return batch_settlement.settle_decision_batch(
        connection, qbs_frame if qbs_frame is not None else qbs(), rbs_frame if rbs_frame is not None else rbs(), completed, **kwargs
    )


def test_empty_database_returns_empty_report_without_calling_clock(connection):
    called = False
    def clock():
        nonlocal called
        called = True
        return TIME
    report = settle(connection, clock=clock)
    assert report.entries == ()
    assert called is False


def test_unresolved_entries_preserve_matcher_diagnostics_and_remain_pending(connection):
    record_decision(connection, pending_decision())
    report = settle(connection, completed=())
    entry = report.entries[0]
    assert (entry.batch_status, entry.match_status, entry.actual_result, entry.decision_status) == ("unresolved", "game_pending", None, "pending")
    assert entry.diagnostic == "Game is not in completed_game_ids."
    assert get_decision(connection, "qb").status == "pending"


def test_completed_qb_rb_and_push_are_settled_by_existing_settlement_logic(connection):
    record_decision(connection, pending_decision())
    record_decision(connection, pending_decision(decision_id="rb", position="RB", player_id="rb-1", market_key="player_rush_yds", selection="under", line=Decimal("80")))
    record_decision(connection, pending_decision(decision_id="push", player_id="qb-2", line=Decimal("250"), game_id="2026_01_BUF_MIA"))
    qb_rows = qbs([
        {"season": 2026, "week": 1, "game_id": "2026_01_KC_LAC", "player_id": "qb-1", "passing_yards": 275},
        {"season": 2026, "week": 1, "game_id": "2026_01_BUF_MIA", "player_id": "qb-2", "passing_yards": 250},
    ])
    report = settle(connection, qbs_frame=qb_rows, completed=("2026_01_KC_LAC", "2026_01_BUF_MIA"), settled_at=TIME + timedelta(hours=1))
    assert [(entry.decision_id, entry.decision_status) for entry in report.settled_entries] == [("push", "push"), ("qb", "win"), ("rb", "win")]
    assert all(entry.actual_result is not None for entry in report.settled_entries)


def test_only_matched_results_are_sent_to_settlement_and_snapshot_is_preserved(connection, monkeypatch):
    original = record_decision(connection, pending_decision())
    calls = []
    real_settle = batch_settlement.settle_decision
    def recording_settle(*args, **kwargs):
        calls.append((args[1], args[2], kwargs["settled_at"]))
        return real_settle(*args, **kwargs)
    monkeypatch.setattr(batch_settlement, "settle_decision", recording_settle)
    report = settle(connection, qbs_frame=qbs(), rbs_frame=rbs([]), settled_at=TIME + timedelta(hours=1))
    stored = get_decision(connection, "qb")
    assert calls == [("qb", Decimal("275"), TIME + timedelta(hours=1))]
    assert report.entries[0].batch_status == "settled"
    assert (stored.line, stored.selected_price, stored.sportsbook, stored.selection, stored.research_notes, stored.recorded_at, stored.odds_retrieved_at) == (original.line, original.selected_price, original.sportsbook, original.selection, original.research_notes, original.recorded_at, original.odds_retrieved_at)


def test_shared_timestamp_clock_behavior_and_rerun_safety(connection):
    record_decision(connection, pending_decision())
    record_decision(connection, pending_decision(decision_id="rb", position="RB", player_id="rb-1", market_key="player_rush_yds"))
    calls = []
    def clock():
        calls.append(True)
        return TIME + timedelta(hours=1)
    first = settle(connection, clock=clock)
    assert calls == [True]
    assert {entry.decision_id for entry in first.settled_entries} == {"qb", "rb"}
    assert {get_decision(connection, item.decision_id).settled_at for item in first.settled_entries} == {TIME + timedelta(hours=1)}
    second = settle(connection, clock=lambda: pytest.fail("clock should not run"))
    assert second.entries == ()
    assert get_decision(connection, "qb").settled_at == TIME + timedelta(hours=1)


def test_explicit_timestamp_skips_clock_and_invalid_shared_timestamp_rolls_back(connection):
    first = record_decision(connection, pending_decision())
    second = record_decision(connection, pending_decision(decision_id="rb", position="RB", player_id="rb-1", market_key="player_rush_yds", recorded_at=TIME + timedelta(hours=2), odds_retrieved_at=TIME + timedelta(hours=2)))
    with pytest.raises(DecisionValidationError, match="must not precede"):
        settle(connection, settled_at=TIME + timedelta(hours=1), clock=lambda: pytest.fail("clock should not run"))
    assert get_decision(connection, "qb") == first
    assert get_decision(connection, "rb") == second


def test_matching_errors_happen_before_writes_and_conflicts_rollback_batch(connection, monkeypatch):
    original = record_decision(connection, pending_decision())
    with pytest.raises(DecisionResultValidationError):
        settle(connection, qbs_frame=qbs().drop(columns=["passing_yards"]))
    assert get_decision(connection, "qb") == original

    record_decision(connection, pending_decision(decision_id="rb", position="RB", player_id="rb-1", market_key="player_rush_yds"))
    real_settle = batch_settlement.settle_decision
    calls = 0
    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise SettlementConflictError("simulated conflict")
        return real_settle(*args, **kwargs)
    monkeypatch.setattr(batch_settlement, "settle_decision", fail_second)
    with pytest.raises(SettlementConflictError, match="simulated"):
        settle(connection, settled_at=TIME + timedelta(hours=3))
    assert get_decision(connection, "qb") == original
    assert get_decision(connection, "rb").status == "pending"


def test_conflicting_source_results_fail_before_any_settlement_write(connection):
    original = record_decision(connection, pending_decision())
    conflict = qbs([
        {"season": 2026, "week": 1, "game_id": "2026_01_KC_LAC", "player_id": "qb-1", "passing_yards": 275},
        {"season": 2026, "week": 1, "game_id": "2026_01_KC_LAC", "player_id": "qb-1", "passing_yards": 276},
    ])
    with pytest.raises(ConflictingResultDataError):
        settle(connection, qbs_frame=conflict, settled_at=TIME + timedelta(hours=1))
    assert get_decision(connection, "qb") == original


def test_orchestrator_calls_list_and_match_once_and_settles_in_match_order(connection, monkeypatch):
    record_decision(connection, pending_decision(decision_id="z", player_id="z", game_id="2026_02_Z", week=2))
    record_decision(connection, pending_decision(decision_id="a", player_id="a", game_id="2026_01_A"))
    source = qbs([
        {"season": 2026, "week": 2, "game_id": "2026_02_Z", "player_id": "z", "passing_yards": 275},
        {"season": 2026, "week": 1, "game_id": "2026_01_A", "player_id": "a", "passing_yards": 275},
    ])
    calls = {"list": 0, "match": 0, "settle": []}
    real_list = batch_settlement.list_decisions
    real_match = batch_settlement.match_decision_results
    real_settle = batch_settlement.settle_decision
    def listed(*args, **kwargs):
        calls["list"] += 1
        return real_list(*args, **kwargs)
    def matched(*args, **kwargs):
        calls["match"] += 1
        return real_match(*args, **kwargs)
    def settled(*args, **kwargs):
        calls["settle"].append(args[1])
        return real_settle(*args, **kwargs)
    monkeypatch.setattr(batch_settlement, "list_decisions", listed)
    monkeypatch.setattr(batch_settlement, "match_decision_results", matched)
    monkeypatch.setattr(batch_settlement, "settle_decision", settled)
    report = settle(connection, qbs_frame=source, completed=("2026_02_Z", "2026_01_A"), settled_at=TIME + timedelta(hours=1))
    assert calls == {"list": 1, "match": 1, "settle": ["a", "z"]}
    assert [entry.decision_id for entry in report.entries] == ["a", "z"]


def test_generator_order_inputs_no_mutation_and_caller_connection_transaction_survives(connection):
    original = record_decision(connection, pending_decision())
    qbs_frame = qbs()
    rbs_frame = rbs()
    before_qbs = qbs_frame.copy(deep=True)
    before_rbs = rbs_frame.copy(deep=True)
    connection.execute("CREATE TABLE caller_work (value TEXT)")
    connection.execute("INSERT INTO caller_work VALUES ('keep')")
    report = settle(connection, qbs_frame=qbs_frame, rbs_frame=rbs_frame, completed=(item for item in ["2026_01_KC_LAC"]), settled_at=TIME + timedelta(hours=1))
    assert [entry.decision_id for entry in report.entries] == ["qb"]
    pd.testing.assert_frame_equal(qbs_frame, before_qbs)
    pd.testing.assert_frame_equal(rbs_frame, before_rbs)
    assert connection.execute("SELECT value FROM caller_work").fetchone() == ("keep",)
    connection.rollback()
    assert get_decision(connection, "qb") == original
    assert connection.execute("SELECT value FROM caller_work").fetchone() is None
