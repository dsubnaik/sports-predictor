"""Tests for deliberate local settlement-runner composition."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import sqlite3

import pandas as pd
import pytest

import football.results.local_settlement_runner as runner_module
from football.data.build_schedule_dataset import OUTPUT_COLUMNS as SCHEDULE_COLUMNS
from football.decision_database import open_local_decision_database
from football.decisions import (
    PendingDecision,
    SettlementConflictError,
    get_decision,
    initialize_decision_store,
    record_decision,
    settle_decision,
)
from football.results.batch_settlement import BatchSettlementReport
from football.results.completed_games import CompletedGameReport, CompletedGameValidationError
from football.results.live_settlement_inputs import LiveSettlementInputs
from football.results.local_settlement_runner import (
    LocalSettlementRunnerValidationError,
    run_local_decision_settlement,
)
from football.results.settlement_workflow import DecisionSettlementWorkflowReport


TIME = datetime(2026, 9, 10, 18, tzinfo=timezone.utc)
QB_COLUMNS = ["season", "week", "game_id", "player_id", "passing_yards"]
RB_COLUMNS = ["season", "week", "game_id", "player_id", "rushing_yards"]


def pending(decision_id="qb", *, season=2026, position="QB", player_id="qb-1", game_id="2026_01_NE_SEA", selection="over", line=Decimal("250.5")):
    return PendingDecision(
        decision_id=decision_id, position=position, player_id=player_id,
        player_name="Player", team="SEA", opponent="NE", season=season, week=1,
        game_id=game_id, market_key="player_pass_yds" if position == "QB" else "player_rush_yds",
        sportsbook="draftkings", line=line, selection=selection, selected_price=-110,
        recorded_at=TIME, odds_retrieved_at=TIME, research_notes=None,
    )


def memory_connection(*decisions):
    connection = sqlite3.connect(":memory:")
    initialize_decision_store(connection)
    for value in decisions:
        record_decision(connection, value)
    return connection


def inputs() -> LiveSettlementInputs:
    schedule = pd.DataFrame([
        (2026, 1, "2026_01_NE_SEA", "2026-09-09", "20:20", "SEA", "NE", "home", 0, 0),
        (2026, 1, "2026_01_NE_SEA", "2026-09-09", "20:20", "NE", "SEA", "away", 0, 0),
    ], columns=SCHEDULE_COLUMNS)
    return LiveSettlementInputs(
        season=2026,
        scores_payload=[{
            "id": "event-1", "completed": True,
            "commence_time": "2026-09-10T00:20:00Z",
            "home_team": "Seattle Seahawks", "away_team": "New England Patriots",
        }],
        normalized_schedule=schedule,
        quarterback_games=pd.DataFrame([
            {"season": 2026, "week": 1, "game_id": "2026_01_NE_SEA", "player_id": "qb-1", "passing_yards": 275}
        ], columns=QB_COLUMNS),
        running_back_games=pd.DataFrame([], columns=RB_COLUMNS),
    )


def test_module_exposes_runner_without_eager_calls():
    assert callable(runner_module.run_local_decision_settlement)


@pytest.mark.parametrize("season", [0, -1, True, False, 2026.0, "2026"])
def test_invalid_season_rejects_before_database_open(season):
    with pytest.raises(LocalSettlementRunnerValidationError, match="season"):
        run_local_decision_settlement(season, database_opener=pytest.fail)


@pytest.mark.parametrize("days_from", [0, -1, 4, True, False, 1.0, "1"])
def test_invalid_days_from_rejects_before_database_open(days_from):
    with pytest.raises(LocalSettlementRunnerValidationError, match="days_from"):
        run_local_decision_settlement(2026, database_opener=pytest.fail, days_from=days_from)


@pytest.mark.parametrize("decisions", [(), (pending("settled"),)])
def test_empty_or_settled_only_database_returns_no_work_and_closes_connection(decisions):
    connection = memory_connection(*decisions)
    if decisions:
        settle_decision(connection, "settled", Decimal("251"), settled_at=TIME + timedelta(hours=1))
    report = run_local_decision_settlement(
        2026,
        database_path="explicit-test-path",
        database_opener=lambda _path: connection,
        inputs_loader=pytest.fail,
        workflow_runner=pytest.fail,
    )
    assert report.outcome == "no_pending_decisions" and report.workflow_report is None
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")


def test_other_season_pending_decision_returns_no_work_and_is_untouched(tmp_path):
    path = tmp_path / "other-season.sqlite3"
    connection = open_local_decision_database(path)
    try:
        record_decision(connection, pending("older", season=2025, game_id="2025_01_NE_SEA"))
        connection.commit()
    finally:
        connection.close()
    report = run_local_decision_settlement(
        2026,
        database_path=path,
        inputs_loader=pytest.fail,
        workflow_runner=pytest.fail,
    )
    assert report.outcome == "no_pending_decisions"
    reopened = open_local_decision_database(path)
    try:
        assert get_decision(reopened, "older").status == "pending"
    finally:
        reopened.close()


def test_target_pending_decision_forwards_inputs_timestamp_and_clock_once():
    connection = memory_connection(pending())
    prepared = inputs()
    workflow_report = DecisionSettlementWorkflowReport(CompletedGameReport((), ()), BatchSettlementReport(()))
    calls = []
    clock = lambda: TIME

    def load(season, *, days_from):
        calls.append(("inputs", season, days_from))
        return prepared

    def workflow(connection_arg, scores_payload, schedule, qbs, rbs, *, settled_at, clock):
        calls.append(("workflow", connection_arg, scores_payload, schedule, qbs, rbs, settled_at, clock))
        return workflow_report

    report = run_local_decision_settlement(
        2026,
        database_path="explicit-test-path",
        database_opener=lambda _path: connection,
        inputs_loader=load,
        workflow_runner=workflow,
        days_from=1,
        settled_at=TIME + timedelta(hours=1),
        clock=clock,
    )
    assert calls[0] == ("inputs", 2026, 1)
    assert calls[1] == (
        "workflow", connection, prepared.scores_payload, prepared.normalized_schedule,
        prepared.quarterback_games, prepared.running_back_games, TIME + timedelta(hours=1), clock,
    )
    assert report.outcome == "executed" and report.workflow_report is workflow_report
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")


def test_loader_and_workflow_failures_propagate_without_partial_success_and_close_connection():
    for failure, loader, workflow in (
        (RuntimeError("input failure"), lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("input failure")), pytest.fail),
        (CompletedGameValidationError("mapping failure"), lambda *_args, **_kwargs: inputs(), lambda *_args, **_kwargs: (_ for _ in ()).throw(CompletedGameValidationError("mapping failure"))),
        (SettlementConflictError("batch failure"), lambda *_args, **_kwargs: inputs(), lambda *_args, **_kwargs: (_ for _ in ()).throw(SettlementConflictError("batch failure"))),
    ):
        connection = memory_connection(pending())
        with pytest.raises(type(failure), match=str(failure)):
            run_local_decision_settlement(
                2026,
                database_path="explicit-test-path",
                database_opener=lambda _path, current=connection: current,
                inputs_loader=loader,
                workflow_runner=workflow,
            )
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")


def test_input_loading_failure_leaves_temporary_database_decision_pending(tmp_path):
    path = tmp_path / "failure.sqlite3"
    connection = open_local_decision_database(path)
    try:
        original = record_decision(connection, pending())
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="input failure"):
        run_local_decision_settlement(
            2026,
            database_path=path,
            inputs_loader=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("input failure")),
            workflow_runner=pytest.fail,
        )
    reopened = open_local_decision_database(path)
    try:
        assert get_decision(reopened, "qb") == original
    finally:
        reopened.close()


def test_real_workflow_settles_target_using_explicit_temporary_database(tmp_path):
    path = tmp_path / "runner.sqlite3"
    connection = open_local_decision_database(path)
    try:
        record_decision(connection, pending())
        record_decision(
            connection,
            pending("other", season=2025, game_id="2025_01_NE_SEA"),
        )
        connection.commit()
    finally:
        connection.close()

    calls = []
    report = run_local_decision_settlement(
        2026,
        database_path=path,
        inputs_loader=lambda season, *, days_from: calls.append((season, days_from)) or inputs(),
        settled_at=TIME + timedelta(hours=1),
    )
    assert report.outcome == "executed"
    assert report.workflow_report is not None
    assert calls == [(2026, 3)]

    reopened = open_local_decision_database(path)
    try:
        assert get_decision(reopened, "qb").status == "win"
        assert get_decision(reopened, "other").status == "pending"
    finally:
        reopened.close()

    no_work = run_local_decision_settlement(
        2026,
        database_path=path,
        inputs_loader=pytest.fail,
        workflow_runner=pytest.fail,
    )
    assert no_work.outcome == "no_pending_decisions"
