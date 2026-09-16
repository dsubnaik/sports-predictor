"""Tests for the read-only Streamlit decision-history page."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import sqlite3

import pytest

import football.ui.decision_history_page as history_page
from football.decision_database import open_local_decision_database
from football.decisions import DecisionStoreError, PendingDecision, record_decision, settle_decision
from football.ui.decision_history_view import (
    ALL_FILTER,
    decision_history_filter_options,
    decision_history_rows,
    filter_decision_history,
)


TIME = datetime(2026, 9, 10, 12, tzinfo=timezone.utc)


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeStreamlit:
    def __init__(self, *, load=False, state=None):
        self.load = load
        self.session_state = {} if state is None else state
        self.messages = []
        self.dataframes = []

    def title(self, value): self.messages.append(("title", value))
    def info(self, value): self.messages.append(("info", value))
    def caption(self, value): self.messages.append(("caption", value))
    def error(self, value): self.messages.append(("error", value))
    def button(self, label, **kwargs): return self.load
    def columns(self, count): return [_Context() for _ in range(count)]
    def selectbox(self, label, options, key): return self.session_state.get(key, options[0])
    def dataframe(self, value, **kwargs): self.dataframes.append(value)


def _pending(decision_id, *, position="QB", season=2026, week=1, sportsbook="draft", line=Decimal("250.5"), selection="over", price=-110):
    return PendingDecision(
        decision_id=decision_id, position=position,
        player_id=f"{position.lower()}-{decision_id}", player_name=f"{position} Player",
        team="KC", opponent="LAC", season=season, week=week,
        game_id=f"{season}_{week:02d}_KC_LAC", market_key="player_pass_yds" if position == "QB" else "player_rush_yds",
        sportsbook=sportsbook, line=line, selection=selection, selected_price=price,
        recorded_at=TIME + timedelta(hours=1), odds_retrieved_at=TIME, research_notes="note",
    )


def _write(path):
    connection = open_local_decision_database(path)
    try:
        qb = record_decision(connection, _pending("qb"))
        rb = record_decision(connection, _pending("rb", position="RB", season=2025, week=2, sportsbook="fan", line=Decimal("55.5"), selection="under", price=-115))
        settled = settle_decision(connection, rb.decision_id, Decimal("50"), settled_at=TIME + timedelta(hours=2))
        connection.commit()
        return qb, settled
    finally:
        connection.close()


def test_history_does_not_open_before_deliberate_load_and_not_loaded_is_not_empty():
    ui = FakeStreamlit()
    history_page.render_decision_history_page(
        streamlit_module=ui,
        database_opener=lambda: pytest.fail("database opened before Load"),
    )
    assert ("caption", "Load Decision History to view locally recorded decisions.") in ui.messages
    assert not any("No decisions recorded" in message for _, message in ui.messages)


def test_load_reads_orders_and_closes_connection(tmp_path):
    path = tmp_path / "decisions.sqlite3"
    _write(path)
    opened = []

    def opener():
        connection = open_local_decision_database(path)
        opened.append(connection)
        return connection

    ui = FakeStreamlit(load=True)
    history_page.render_decision_history_page(streamlit_module=ui, database_opener=opener)
    assert [row["Decision ID"] for row in ui.dataframes[0]] == ["qb", "rb"]
    for connection in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")


def test_filter_rerun_uses_loaded_history_without_reopening_database(tmp_path):
    path = tmp_path / "decisions.sqlite3"
    _write(path)
    state = {"football_history_position": "RB"}
    history_page.render_decision_history_page(
        streamlit_module=FakeStreamlit(load=True, state=state),
        database_opener=lambda: open_local_decision_database(path),
    )
    ui = FakeStreamlit(load=False, state=state)
    history_page.render_decision_history_page(
        streamlit_module=ui,
        database_opener=lambda: pytest.fail("filter rerun opened database"),
    )
    assert [row["Decision ID"] for row in ui.dataframes[0]] == ["rb"]


def test_empty_database_and_zero_filtered_rows_have_distinct_messages(tmp_path):
    empty_path = tmp_path / "empty.sqlite3"
    empty = FakeStreamlit(load=True)
    history_page.render_decision_history_page(
        streamlit_module=empty,
        database_opener=lambda: open_local_decision_database(empty_path),
    )
    assert ("info", "No decisions recorded yet.") in empty.messages

    path = tmp_path / "decisions.sqlite3"
    _write(path)
    state = {"football_history_position": "RB", "football_history_status": "pending"}
    filtered = FakeStreamlit(load=True, state=state)
    history_page.render_decision_history_page(
        streamlit_module=filtered,
        database_opener=lambda: open_local_decision_database(path),
    )
    assert ("info", "No decisions match these filters.") in filtered.messages


def test_filter_helpers_and_display_are_deterministic_and_preserve_snapshot_values(tmp_path):
    path = tmp_path / "decisions.sqlite3"
    qb, rb = _write(path)
    decisions = (qb, rb)
    options = decision_history_filter_options(decisions)
    assert options["position"] == (ALL_FILTER, "QB", "RB")
    assert options["season"] == (ALL_FILTER, 2025, 2026)
    assert options["sportsbook"] == (ALL_FILTER, "draft", "fan")
    assert filter_decision_history(decisions, position="RB", market_key="player_rush_yds", season=2025, week=2, sportsbook="fan", status="win") == (rb,)
    rows = decision_history_rows(decisions)
    assert rows[0]["Line"] == "250.5"
    assert rows[0]["Selected Price"] == -110
    assert rows[0]["Actual Result"] == "Unsettled"
    assert rows[0]["Settled At"] == "Unsettled"
    assert rows[0]["Sportsbook"] == "draft"
    assert rows[1]["Actual Result"] == "50"
    assert rows[1]["Status"] == "win"
    assert rows[1]["Settled At"].endswith("Z")


def test_failed_refresh_preserves_loaded_history_and_closes_open_connection(tmp_path, monkeypatch):
    path = tmp_path / "decisions.sqlite3"
    _write(path)
    state = {"football_qb_research_result": "qb", "football_rb_research_result": "rb"}
    history_page.render_decision_history_page(
        streamlit_module=FakeStreamlit(load=True, state=state),
        database_opener=lambda: open_local_decision_database(path),
    )
    loaded = state["football_decision_history_decisions"]
    created = sqlite3.connect(":memory:")
    monkeypatch.setattr(history_page, "list_decisions", lambda _connection: (_ for _ in ()).throw(DecisionStoreError("read failure")))
    failed = FakeStreamlit(load=True, state=state)
    history_page.render_decision_history_page(streamlit_module=failed, database_opener=lambda: created)
    assert state["football_decision_history_decisions"] is loaded
    assert state["football_qb_research_result"] == "qb"
    assert state["football_rb_research_result"] == "rb"
    assert ("error", "Could not load the local decision history.") in failed.messages
    with pytest.raises(sqlite3.ProgrammingError):
        created.execute("SELECT 1")


def test_navigation_adds_decision_history_without_page_database_access():
    app = open("app.py", encoding="utf-8").read()
    assert "render_decision_history_page" in app
    assert 'title="Decision History"' in app
