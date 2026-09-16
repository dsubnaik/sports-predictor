"""Tests for the read-only Streamlit decision-history page."""

from dataclasses import replace
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
    def subheader(self, value): self.messages.append(("subheader", value))
    def info(self, value): self.messages.append(("info", value))
    def caption(self, value): self.messages.append(("caption", value))
    def error(self, value): self.messages.append(("error", value))
    def button(self, label, **kwargs): return self.load
    def columns(self, count): return [_Context() for _ in range(count)]
    def selectbox(self, label, options, key): return self.session_state.get(key, options[0])
    def dataframe(self, value, **kwargs): self.dataframes.append(value)


def _snapshot_rows(ui):
    return next(rows for rows in ui.dataframes if rows and "Decision ID" in rows[0])


def _summary_rows(ui):
    return next(rows for rows in ui.dataframes if rows and rows[0].get("Scope") == "Current filters")


def _breakdown_rows(ui, first_key):
    return next(
        rows for rows in ui.dataframes
        if rows and rows[0].get("Group") == first_key
    )


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
    assert not ui.dataframes


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
    assert [row["Decision ID"] for row in _snapshot_rows(ui)] == ["qb", "rb"]
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
    assert [row["Decision ID"] for row in _snapshot_rows(ui)] == ["rb"]
    assert _summary_rows(ui)[0]["Decisions"] == 1


def test_empty_database_and_zero_filtered_rows_have_distinct_messages(tmp_path):
    empty_path = tmp_path / "empty.sqlite3"
    empty = FakeStreamlit(load=True)
    history_page.render_decision_history_page(
        streamlit_module=empty,
        database_opener=lambda: open_local_decision_database(empty_path),
    )
    assert ("info", "No decisions recorded yet.") in empty.messages
    assert not empty.dataframes

    path = tmp_path / "decisions.sqlite3"
    _write(path)
    state = {"football_history_position": "RB", "football_history_status": "pending"}
    filtered = FakeStreamlit(load=True, state=state)
    history_page.render_decision_history_page(
        streamlit_module=filtered,
        database_opener=lambda: open_local_decision_database(path),
    )
    assert ("info", "No decisions match these filters.") in filtered.messages
    assert not filtered.dataframes


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
    assert _summary_rows(failed)[0]["Decisions"] == 2
    with pytest.raises(sqlite3.ProgrammingError):
        created.execute("SELECT 1")


def test_navigation_adds_decision_history_without_page_database_access():
    app = open("app.py", encoding="utf-8").read()
    assert "render_decision_history_page" in app
    assert 'title="Decision History"' in app


def test_performance_display_uses_current_filtered_decisions_and_backend_breakdowns(tmp_path):
    path = tmp_path / "performance.sqlite3"
    connection = open_local_decision_database(path)
    try:
        positive = record_decision(connection, _pending("positive", price=150))
        negative = record_decision(connection, _pending("negative", position="RB", price=-150, selection="under", sportsbook="fan", line=Decimal("55.5")))
        loss = record_decision(connection, _pending("loss", price=-110, sportsbook="betmgm"))
        push = record_decision(connection, _pending("push", position="RB", price=100, selection="under", sportsbook="fan", line=Decimal("70")))
        settle_decision(connection, positive.decision_id, Decimal("251"), settled_at=TIME + timedelta(hours=2))
        settle_decision(connection, negative.decision_id, Decimal("50"), settled_at=TIME + timedelta(hours=2))
        settle_decision(connection, loss.decision_id, Decimal("249"), settled_at=TIME + timedelta(hours=2))
        settle_decision(connection, push.decision_id, Decimal("70"), settled_at=TIME + timedelta(hours=2))
        connection.commit()
    finally:
        connection.close()

    ui = FakeStreamlit(load=True)
    history_page.render_decision_history_page(
        streamlit_module=ui, database_opener=lambda: open_local_decision_database(path)
    )
    summary = _summary_rows(ui)[0]
    assert summary["Scope"] == "Current filters"
    assert summary["Hit Rate"] == "66.7%"
    assert summary["Hypothetical Flat-Stake Units"] == "1.17"
    assert ("subheader", "Performance — Current filters") in ui.messages
    assert any("not verified placed wagers" in message for kind, message in ui.messages if kind == "info")
    assert [row["Group"] for row in _breakdown_rows(ui, "QB")] == ["QB", "RB"]
    assert [row["Group"] for row in _breakdown_rows(ui, "player_pass_yds")] == ["player_pass_yds", "player_rush_yds"]
    assert [row["Group"] for row in _breakdown_rows(ui, "2026 Week 1")] == ["2026 Week 1"]
    assert [row["Group"] for row in _breakdown_rows(ui, "betmgm")] == ["betmgm", "draft", "fan"]

    filtered = FakeStreamlit(load=False, state={**ui.session_state, "football_history_position": "RB"})
    history_page.render_decision_history_page(
        streamlit_module=filtered, database_opener=lambda: pytest.fail("filter rerun opened database")
    )
    assert _summary_rows(filtered)[0]["Decisions"] == 2
    assert [row["Decision ID"] for row in _snapshot_rows(filtered)] == ["negative", "push"]

    combined = FakeStreamlit(
        load=False,
        state={
            **ui.session_state,
            "football_history_position": "RB",
            "football_history_sportsbook": "fan",
            "football_history_status": "win",
        },
    )
    history_page.render_decision_history_page(
        streamlit_module=combined, database_opener=lambda: pytest.fail("filter rerun opened database")
    )
    assert _summary_rows(combined)[0]["Decisions"] == 1
    assert [row["Decision ID"] for row in _snapshot_rows(combined)] == ["negative"]


def test_pending_and_zero_net_performance_formatting_and_validation_error_are_safe(tmp_path):
    _write_pending(tmp_path)
    pending_ui = FakeStreamlit(load=True)
    history_page.render_decision_history_page(
        streamlit_module=pending_ui,
        database_opener=lambda: open_local_decision_database(tmp_path / "pending.sqlite3"),
    )
    pending_summary = _summary_rows(pending_ui)[0]
    assert pending_summary["Hit Rate"] == "N/A (no wins or losses)"
    assert pending_summary["Hypothetical Flat-Stake Units"] == "0.00"

    path = tmp_path / "zero.sqlite3"
    connection = open_local_decision_database(path)
    try:
        winner = record_decision(connection, _pending("winner", price=100))
        loser = record_decision(connection, _pending("loser", price=-150))
        settle_decision(connection, winner.decision_id, Decimal("251"), settled_at=TIME + timedelta(hours=2))
        settle_decision(connection, loser.decision_id, Decimal("249"), settled_at=TIME + timedelta(hours=2))
        connection.commit()
    finally:
        connection.close()
    ui = FakeStreamlit(load=True)
    history_page.render_decision_history_page(streamlit_module=ui, database_opener=lambda: open_local_decision_database(path))
    assert _summary_rows(ui)[0]["Hypothetical Flat-Stake Units"] == "0.00"

    loaded = _write_pending(tmp_path, decision_id="invalid")
    invalid = replace(
        loaded,
        status="win",
        actual_result=Decimal("249"),
        settled_at=TIME + timedelta(hours=2),
    )
    state = {"football_decision_history_decisions": (invalid,), "football_qb_research_result": "qb"}
    failed = FakeStreamlit(state=state)
    history_page.render_decision_history_page(
        streamlit_module=failed, database_opener=lambda: pytest.fail("invalid display opened database")
    )
    assert ("error", "Could not calculate performance for the loaded decision history.") in failed.messages
    assert not any(rows and rows[0].get("Scope") == "Current filters" for rows in failed.dataframes)
    assert state["football_qb_research_result"] == "qb"


def _write_pending(tmp_path, decision_id="pending"):
    path = tmp_path / f"{decision_id}.sqlite3"
    connection = open_local_decision_database(path)
    try:
        stored = record_decision(connection, _pending(decision_id))
        connection.commit()
        return stored
    finally:
        connection.close()
