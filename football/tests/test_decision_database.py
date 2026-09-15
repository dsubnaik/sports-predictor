"""Tests for local file-backed decision-store connection ownership."""

from datetime import datetime, timezone
from decimal import Decimal
import sqlite3
from pathlib import Path

import pytest

import football.decision_database as decision_database
from football.decisions import PendingDecision, get_decision, record_decision


TIME = datetime(2026, 9, 10, 18, 0, tzinfo=timezone.utc)


def pending_decision():
    return PendingDecision(
        decision_id="decision-1", position="QB", player_id="qb-1",
        player_name="Quarterback", team="SEA", opponent="NE", season=2026,
        week=1, game_id="2026_01_NE_SEA", market_key="player_pass_yds",
        sportsbook="draftkings", line=Decimal("250.5"), selection="over",
        selected_price=-110, recorded_at=TIME, odds_retrieved_at=TIME,
        research_notes="snapshot",
    )


def test_default_path_is_absolute_deterministic_and_side_effect_free(monkeypatch, tmp_path):
    before = decision_database.get_local_decision_database_path()
    monkeypatch.chdir(tmp_path)
    after = decision_database.get_local_decision_database_path()

    assert before == after
    assert before == decision_database.FOOTBALL_DIR / "data" / "local" / "decisions.sqlite3"
    assert before.is_absolute()


def test_explicit_file_path_creates_parent_initializes_schema_and_keeps_connection_open(tmp_path):
    path = tmp_path / "nested" / "decisions.sqlite3"
    connection = decision_database.open_local_decision_database(str(path))
    try:
        assert path.parent.is_dir()
        assert connection.execute("SELECT 1").fetchone() == (1,)
        stored = record_decision(connection, pending_decision())
        assert get_decision(connection, stored.decision_id) == stored
    finally:
        connection.close()


def test_closing_and_reopening_preserves_existing_snapshot_without_replacement(tmp_path):
    path = tmp_path / "decisions.sqlite3"
    first = decision_database.open_local_decision_database(path)
    try:
        stored = record_decision(first, pending_decision())
        first.commit()
    finally:
        first.close()

    second = decision_database.open_local_decision_database(path)
    try:
        assert get_decision(second, "decision-1") == stored
        decision_database.initialize_decision_store(second)
        assert get_decision(second, "decision-1") == stored
    finally:
        second.close()


def test_each_open_call_returns_a_distinct_caller_owned_connection(tmp_path):
    path = tmp_path / "decisions.sqlite3"
    first = decision_database.open_local_decision_database(path)
    second = decision_database.open_local_decision_database(path)
    try:
        assert first is not second
        first.execute("SELECT 1").fetchone()
        second.execute("SELECT 1").fetchone()
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("path", ["", "   ", ":memory:", Path(":memory:"), 1, object()])
def test_invalid_or_memory_paths_are_rejected_deterministically(path):
    with pytest.raises(decision_database.DecisionDatabaseValidationError, match="path"):
        decision_database.open_local_decision_database(path)


def test_schema_initialization_failure_closes_only_created_connection(tmp_path, monkeypatch):
    created = sqlite3.connect(":memory:")
    monkeypatch.setattr(decision_database.sqlite3, "connect", lambda _path: created)
    monkeypatch.setattr(
        decision_database,
        "initialize_decision_store",
        lambda _connection: (_ for _ in ()).throw(RuntimeError("schema failure")),
    )

    with pytest.raises(RuntimeError, match="schema failure"):
        decision_database.open_local_decision_database(tmp_path / "failed.sqlite3")

    with pytest.raises(sqlite3.ProgrammingError):
        created.execute("SELECT 1")


def test_open_failure_propagates_without_in_memory_fallback(tmp_path, monkeypatch):
    requested = tmp_path / "unavailable.sqlite3"
    failure = sqlite3.OperationalError("open failed")
    calls = []

    def failing_connect(path):
        calls.append(path)
        raise failure

    monkeypatch.setattr(decision_database.sqlite3, "connect", failing_connect)
    with pytest.raises(sqlite3.OperationalError) as error:
        decision_database.open_local_decision_database(requested)

    assert error.value is failure
    assert calls == [str(requested)]


def test_gitignore_covers_only_the_local_database_and_sqlite_sidecars():
    rules = (Path(__file__).parents[2] / ".gitignore").read_text().splitlines()
    expected = {
        "/football/data/local/decisions.sqlite3",
        "/football/data/local/decisions.sqlite3-journal",
        "/football/data/local/decisions.sqlite3-wal",
        "/football/data/local/decisions.sqlite3-shm",
    }

    assert expected.issubset(rules)
    assert "*.sqlite" not in rules
    assert "*.sqlite3" not in rules
    assert "*.db" not in rules
