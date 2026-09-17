"""Tests for the read-only Streamlit decision-history page."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import sqlite3

import pytest

import football.ui.decision_history_page as history_page
from football.decision_database import open_local_decision_database
from football.decisions import (
    DecisionStoreError,
    PendingDecision,
    get_decision,
    record_decision,
    settle_decision,
)
from football.results.batch_settlement import BatchSettlementEntry, BatchSettlementReport
from football.results.completed_games import CompletedGameDiagnostic, CompletedGameReport
from football.results.local_settlement_runner import LocalSettlementRunReport
from football.results.settlement_workflow import DecisionSettlementWorkflowReport
from football.ui.decision_history_view import (
    ALL_FILTER,
    completed_game_diagnostic_rows,
    decision_history_filter_options,
    decision_history_rows,
    filter_decision_history,
    ManualSettlementInputError,
    manual_pending_decision_options,
    pending_decision_seasons,
    parse_manual_actual_result,
    settled_settlement_rows,
    unresolved_settlement_rows,
)


TIME = datetime(2026, 9, 10, 12, tzinfo=timezone.utc)


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeStreamlit:
    def __init__(
        self,
        *,
        load=False,
        settle=False,
        acknowledged=False,
        manual=False,
        manual_acknowledged=False,
        manual_result="",
        state=None,
    ):
        self.load = load
        self.settle = settle
        self.acknowledged = acknowledged
        self.manual = manual
        self.manual_acknowledged = manual_acknowledged
        self.manual_result = manual_result
        self.session_state = {} if state is None else state
        self.messages = []
        self.dataframes = []
        self.forms = []

    def title(self, value): self.messages.append(("title", value))
    def subheader(self, value): self.messages.append(("subheader", value))
    def info(self, value): self.messages.append(("info", value))
    def caption(self, value): self.messages.append(("caption", value))
    def warning(self, value): self.messages.append(("warning", value))
    def error(self, value): self.messages.append(("error", value))
    def success(self, value): self.messages.append(("success", value))
    def button(self, label, **kwargs): return self.load if label == "Load Decision History" else False
    def form(self, key): self.forms.append(key); return _Context()
    def checkbox(self, label, key):
        default = self.manual_acknowledged if "I verified" in label else self.acknowledged
        return self.session_state.get(key, default)
    def form_submit_button(self, label, **kwargs):
        if label == "Settle Pending Decisions": return self.settle
        if label == "Settle Manually Verified Result": return self.manual
        return False
    def columns(self, count): return [_Context() for _ in range(count)]
    def selectbox(self, label, options, key, **kwargs): return self.session_state.get(key, options[0])
    def text_input(self, _label, key): return self.session_state.get(key, self.manual_result)
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


def _executed_settlement_report(*, mapping=True, unresolved=True):
    diagnostics = (
        CompletedGameDiagnostic("event-unmatched", "unmatched", "No schedule game matched."),
    ) if mapping else ()
    entries = (
        BatchSettlementEntry(
            decision_id="pending-qb", position="QB", market_key="player_pass_yds",
            season=2026, week=1, game_id="2026_01_KC_LAC", player_id="qb-pending-qb",
            batch_status="settled", match_status="matched", actual_result=Decimal("275"),
            decision_status="win", diagnostic=None,
        ),
        BatchSettlementEntry(
            decision_id="missing-rb", position="RB", market_key="player_rush_yds",
            season=2026, week=1, game_id="2026_01_KC_LAC", player_id="rb-missing-rb",
            batch_status="unresolved", match_status="player_result_missing", actual_result=None,
            decision_status="pending", diagnostic="No normalized player result exists for the completed game.",
        ),
    ) if unresolved else ()
    return LocalSettlementRunReport(
        season=2026,
        outcome="executed",
        workflow_report=DecisionSettlementWorkflowReport(
            CompletedGameReport(("2026_01_KC_LAC",), diagnostics),
            BatchSettlementReport(entries),
        ),
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


def test_settlement_controls_do_not_run_before_history_load_or_on_history_load(tmp_path):
    calls = []
    before_load = FakeStreamlit()
    history_page.render_decision_history_page(
        streamlit_module=before_load,
        database_opener=lambda: pytest.fail("database opened before Load"),
        settlement_runner=lambda season: calls.append(season),
    )
    assert not calls

    path = tmp_path / "history.sqlite3"
    _write_pending_to_path(path)
    loaded = FakeStreamlit(load=True)
    history_page.render_decision_history_page(
        streamlit_module=loaded,
        database_opener=lambda: open_local_decision_database(path),
        settlement_runner=lambda season: calls.append(season),
    )
    assert not calls


def test_settlement_requires_acknowledgement_and_filter_reruns_do_not_call_runner(tmp_path):
    path = tmp_path / "history.sqlite3"
    _write_pending_to_path(path)
    state = {"football_history_position": "QB"}
    history_page.render_decision_history_page(
        streamlit_module=FakeStreamlit(load=True, state=state),
        database_opener=lambda: open_local_decision_database(path),
        settlement_runner=pytest.fail,
    )

    calls = []
    unacknowledged = FakeStreamlit(settle=True, acknowledged=False, state=state)
    history_page.render_decision_history_page(
        streamlit_module=unacknowledged,
        database_opener=lambda: pytest.fail("settlement form opened history database"),
        settlement_runner=lambda season: calls.append(season),
    )
    assert not calls
    assert any("Acknowledge" in message for kind, message in unacknowledged.messages if kind == "error")

    rerun = FakeStreamlit(state=state)
    history_page.render_decision_history_page(
        streamlit_module=rerun,
        database_opener=lambda: pytest.fail("filter rerun opened history database"),
        settlement_runner=lambda season: calls.append(season),
    )
    assert not calls


def test_confirmed_settlement_runs_once_refreshes_history_and_renders_distinct_diagnostics(tmp_path):
    path = tmp_path / "history.sqlite3"
    _write_pending_to_path(path, decision_id="pending-qb")
    opened = []
    calls = []

    def opener():
        connection = open_local_decision_database(path)
        opened.append(connection)
        return connection

    def runner(season):
        calls.append(season)
        return _executed_settlement_report()

    state = {"football_history_settlement_acknowledged": True}
    ui = FakeStreamlit(load=True, settle=True, state=state)
    history_page.render_decision_history_page(
        streamlit_module=ui,
        database_opener=opener,
        settlement_runner=runner,
    )
    assert calls == [2026]
    assert len(opened) == 2  # Initial load and one local post-settlement refresh.
    assert (
        "success",
        "Settlement run completed: 1 settled or confirmed; 1 unresolved.",
    ) in ui.messages
    assert ("caption", "Completed-event mapping diagnostics") in ui.messages
    assert ("caption", "Settled or confirmed decisions") in ui.messages
    assert ("caption", "Unresolved decision diagnostics") in ui.messages
    assert any("never assumed to be zero" in message for kind, message in ui.messages if kind == "info")
    assert any(rows and "Provider Event ID" in rows[0] for rows in ui.dataframes)
    assert any(rows and rows[0].get("Decision ID") == "pending-qb" and "Decision Status" in rows[0] for rows in ui.dataframes)
    assert any(rows and rows[0].get("Decision ID") == "missing-rb" for rows in ui.dataframes)
    for connection in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")


def test_no_work_message_never_claims_a_live_scores_request(tmp_path):
    path = tmp_path / "history.sqlite3"
    _write_pending_to_path(path)
    calls = []
    report = LocalSettlementRunReport(2026, "no_pending_decisions", None)
    ui = FakeStreamlit(load=True, settle=True, acknowledged=True)
    history_page.render_decision_history_page(
        streamlit_module=ui,
        database_opener=lambda: open_local_decision_database(path),
        settlement_runner=lambda season: calls.append(season) or report,
    )
    assert calls == [2026]
    assert any(
        "no live inputs or Odds API scores request was made" in message
        for kind, message in ui.messages if kind == "info"
    )


def test_prior_executed_report_does_not_trigger_settlement_on_ordinary_rerun(tmp_path):
    path = tmp_path / "history.sqlite3"
    stored = _write_pending_to_path(path)
    state = {
        "football_decision_history_decisions": (stored,),
        "football_decision_history_settlement_report": _executed_settlement_report(),
    }
    ui = FakeStreamlit(state=state)
    history_page.render_decision_history_page(
        streamlit_module=ui,
        database_opener=lambda: pytest.fail("ordinary rerun opened history database"),
        settlement_runner=pytest.fail,
    )
    assert ("caption", "Most recent deliberate settlement run (not a new settlement).") in ui.messages


def test_repeated_deliberate_submissions_are_separate_runner_invocations(tmp_path):
    path = tmp_path / "history.sqlite3"
    stored = _write_pending_to_path(path)
    state = {"football_decision_history_decisions": (stored,)}
    calls = []
    no_work = LocalSettlementRunReport(2026, "no_pending_decisions", None)
    for _ in range(2):
        history_page.render_decision_history_page(
            streamlit_module=FakeStreamlit(settle=True, acknowledged=True, state=state),
            database_opener=lambda: pytest.fail("no-work settlement should not refresh history"),
            settlement_runner=lambda season: calls.append(season) or no_work,
        )
    assert calls == [2026, 2026]


def test_settlement_failure_preserves_loaded_history_state_and_reports_no_partial_success(tmp_path):
    path = tmp_path / "history.sqlite3"
    stored = _write_pending_to_path(path)
    state = {
        "football_decision_history_decisions": (stored,),
        "football_qb_research_result": "qb",
        "football_rb_research_result": "rb",
    }
    ui = FakeStreamlit(settle=True, acknowledged=True, state=state)
    history_page.render_decision_history_page(
        streamlit_module=ui,
        database_opener=lambda: pytest.fail("failed settlement opened history database"),
        settlement_runner=lambda _season: (_ for _ in ()).throw(DecisionStoreError("conflict")),
    )
    assert state["football_decision_history_decisions"] == (stored,)
    assert state["football_qb_research_result"] == "qb"
    assert state["football_rb_research_result"] == "rb"
    assert ("error", "Could not settle pending decisions. History was left unchanged.") in ui.messages


def test_failed_post_settlement_refresh_preserves_history_and_marks_it_stale(tmp_path, monkeypatch):
    stored = _write_pending(tmp_path)
    state = {"football_decision_history_decisions": (stored,)}
    connection = sqlite3.connect(":memory:")
    monkeypatch.setattr(
        history_page,
        "list_decisions",
        lambda _connection: (_ for _ in ()).throw(DecisionStoreError("read failure")),
    )
    ui = FakeStreamlit(settle=True, acknowledged=True, state=state)
    history_page.render_decision_history_page(
        streamlit_module=ui,
        database_opener=lambda: connection,
        settlement_runner=lambda _season: _executed_settlement_report(mapping=False, unresolved=False),
    )
    assert state["football_decision_history_decisions"] == (stored,)
    assert state["football_decision_history_may_be_stale"] is True
    assert any("may be stale" in message for kind, message in ui.messages if kind == "warning")
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")


def test_pending_season_and_diagnostic_view_helpers_are_deterministic():
    qb = _pending("qb", season=2026)
    rb = _pending("rb", position="RB", season=2025)
    settled = replace(qb, decision_id="settled", status="win", actual_result=Decimal("251"), settled_at=TIME + timedelta(hours=2))
    assert pending_decision_seasons((qb, rb, settled)) == (2025, 2026)
    report = _executed_settlement_report()
    assert completed_game_diagnostic_rows(report.workflow_report.completed_game_report.diagnostics) == [{
        "Provider Event ID": "event-unmatched", "Match Status": "unmatched", "Diagnostic": "No schedule game matched.",
    }]
    unresolved = unresolved_settlement_rows(report.workflow_report.batch_settlement_report.unresolved_entries)
    assert unresolved[0]["Decision ID"] == "missing-rb"
    assert unresolved[0]["Diagnostic"].startswith("No normalized player result")
    settled_rows = settled_settlement_rows(report.workflow_report.batch_settlement_report.settled_entries)
    assert settled_rows == [{
        "Decision ID": "pending-qb", "Position": "QB", "Market": "player_pass_yds",
        "Game ID": "2026_01_KC_LAC", "Player ID": "qb-pending-qb",
        "Actual Result": "275", "Decision Status": "win",
    }]


def test_manual_historical_settlement_is_unavailable_before_load_or_without_pending_decisions(tmp_path):
    before_load = FakeStreamlit()
    history_page.render_decision_history_page(
        streamlit_module=before_load,
        database_opener=lambda: pytest.fail("manual form opened database before History load"),
        settlement_runner=pytest.fail,
    )
    assert "football_history_manual_settlement_form" not in before_load.forms

    path = tmp_path / "settled.sqlite3"
    connection = open_local_decision_database(path)
    try:
        stored = record_decision(connection, _pending("settled"))
        settle_decision(connection, stored.decision_id, Decimal("251"), settled_at=TIME + timedelta(hours=2))
        connection.commit()
    finally:
        connection.close()
    no_pending = FakeStreamlit(load=True)
    history_page.render_decision_history_page(
        streamlit_module=no_pending,
        database_opener=lambda: open_local_decision_database(path),
        settlement_runner=pytest.fail,
    )
    assert "football_history_manual_settlement_form" not in no_pending.forms
    assert any("No loaded pending decisions" in message for kind, message in no_pending.messages if kind == "info")


def test_manual_pending_options_and_decimal_parser_are_deterministic_and_safe():
    later = _pending("later", season=2026)
    earlier = _pending("earlier", season=2025)
    options = manual_pending_decision_options((later, earlier))
    assert [option.decision_id for option in options] == ["earlier", "later"]
    assert "QB/player_pass_yds" in options[0].label
    assert "KC vs LAC" in options[0].label
    assert "draft over 250.5" in options[0].label
    assert options[0].label.endswith("earlier")
    assert parse_manual_actual_result(" 250.125 ") == Decimal("250.125")
    assert parse_manual_actual_result("-1") == Decimal("-1")
    for value in ("", "NaN", "Infinity", "-Infinity", True, 12):
        with pytest.raises(ManualSettlementInputError):
            parse_manual_actual_result(value)


def test_manual_unacknowledged_or_invalid_result_never_opens_database(tmp_path):
    stored = _write_pending(tmp_path)
    state = _manual_state(stored)
    unacknowledged = FakeStreamlit(
        manual=True,
        manual_acknowledged=False,
        manual_result="251",
        state=state,
    )
    history_page.render_decision_history_page(
        streamlit_module=unacknowledged,
        database_opener=lambda: pytest.fail("unacknowledged manual submission opened database"),
        settlement_runner=pytest.fail,
    )
    assert any("Acknowledge" in message for kind, message in unacknowledged.messages if kind == "error")

    for result in ("", "NaN", "Infinity", "-Infinity"):
        invalid = FakeStreamlit(manual=True, manual_acknowledged=True, manual_result=result, state=_manual_state(stored))
        history_page.render_decision_history_page(
            streamlit_module=invalid,
            database_opener=lambda: pytest.fail("invalid result opened database"),
            settlement_runner=pytest.fail,
        )
        assert any("finite" in message or "nonblank" in message for kind, message in invalid.messages if kind == "error")


@pytest.mark.parametrize(
    ("selection", "line", "result_text", "expected_status"),
    [
        ("over", Decimal("250"), "251", "win"),
        ("over", Decimal("250"), "249", "loss"),
        ("over", Decimal("250"), "250", "push"),
        ("under", Decimal("250"), "249", "win"),
        ("under", Decimal("250"), "251", "loss"),
        ("under", Decimal("250"), "250", "push"),
    ],
)
def test_manual_settlement_uses_store_outcomes_and_refreshes_history(
    tmp_path,
    selection,
    line,
    result_text,
    expected_status,
):
    path = tmp_path / f"{selection}-{result_text}.sqlite3"
    connection = open_local_decision_database(path)
    try:
        stored = record_decision(
            connection,
            _pending("manual", selection=selection, line=line),
        )
        connection.commit()
    finally:
        connection.close()
    opened = []

    def opener():
        current = open_local_decision_database(path)
        opened.append(current)
        return current

    ui = FakeStreamlit(
        manual=True,
        manual_acknowledged=True,
        manual_result=result_text,
        state=_manual_state(stored),
    )
    history_page.render_decision_history_page(
        streamlit_module=ui,
        database_opener=opener,
        settlement_runner=pytest.fail,
        manual_settlement_clock=lambda: TIME + timedelta(hours=3),
    )
    check = open_local_decision_database(path)
    try:
        settled = get_decision(check, "manual")
        assert settled.status == expected_status
        assert settled.actual_result == Decimal(result_text)
        assert settled.settled_at == TIME + timedelta(hours=3)
    finally:
        check.close()
    assert len(opened) == 2
    assert _summary_rows(ui)[0]["Pending"] == 0
    assert any("No Odds API request was made" in message for kind, message in ui.messages if kind == "success")
    for current in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            current.execute("SELECT 1")


def test_manual_decimal_and_negative_results_round_trip_without_float_conversion(tmp_path):
    path = tmp_path / "manual-decimal.sqlite3"
    connection = open_local_decision_database(path)
    try:
        decimal_decision = record_decision(connection, _pending("decimal", line=Decimal("250")))
        negative_decision = record_decision(
            connection,
            _pending("negative", line=Decimal("0"), selection="under"),
        )
        connection.commit()
    finally:
        connection.close()
    for stored, value in ((decimal_decision, "250.125"), (negative_decision, "-1")):
        ui = FakeStreamlit(
            manual=True,
            manual_acknowledged=True,
            manual_result=value,
            state=_manual_state(stored),
        )
        history_page.render_decision_history_page(
            streamlit_module=ui,
            database_opener=lambda: open_local_decision_database(path),
            settlement_runner=pytest.fail,
            manual_settlement_clock=lambda: TIME + timedelta(hours=3),
        )
    check = open_local_decision_database(path)
    try:
        assert get_decision(check, "decimal").actual_result == Decimal("250.125")
        assert get_decision(check, "negative").actual_result == Decimal("-1")
    finally:
        check.close()


def test_manual_settlement_rereads_selected_id_and_handles_missing_id_without_settling(tmp_path, monkeypatch):
    stored = _write_pending(tmp_path)
    calls = []
    original_get = history_page.get_decision

    def counted_get(connection, decision_id):
        calls.append(decision_id)
        return original_get(connection, decision_id)

    monkeypatch.setattr(history_page, "get_decision", counted_get)
    empty_path = tmp_path / "missing.sqlite3"
    ui = FakeStreamlit(
        manual=True,
        manual_acknowledged=True,
        manual_result="251",
        state=_manual_state(stored),
    )
    monkeypatch.setattr(history_page, "settle_decision", pytest.fail)
    history_page.render_decision_history_page(
        streamlit_module=ui,
        database_opener=lambda: open_local_decision_database(empty_path),
        settlement_runner=pytest.fail,
        manual_settlement_clock=lambda: TIME + timedelta(hours=3),
    )
    assert calls == [stored.decision_id]
    assert any("no longer exists" in message for kind, message in ui.messages if kind == "error")


def test_manual_idempotent_confirmation_and_conflict_preserve_existing_result(tmp_path):
    path = tmp_path / "concurrent.sqlite3"
    connection = open_local_decision_database(path)
    try:
        pending = record_decision(connection, _pending("concurrent"))
        settled = settle_decision(connection, pending.decision_id, Decimal("251"), settled_at=TIME + timedelta(hours=2))
        connection.commit()
    finally:
        connection.close()

    confirmed = FakeStreamlit(
        manual=True,
        manual_acknowledged=True,
        manual_result="251",
        state=_manual_state(pending),
    )
    history_page.render_decision_history_page(
        streamlit_module=confirmed,
        database_opener=lambda: open_local_decision_database(path),
        settlement_runner=pytest.fail,
        manual_settlement_clock=lambda: TIME + timedelta(hours=3),
    )
    assert any("Manual settlement concurrent" in message for kind, message in confirmed.messages if kind == "success")

    conflict = FakeStreamlit(
        manual=True,
        manual_acknowledged=True,
        manual_result="249",
        state=_manual_state(pending),
    )
    history_page.render_decision_history_page(
        streamlit_module=conflict,
        database_opener=lambda: open_local_decision_database(path),
        settlement_runner=pytest.fail,
        manual_settlement_clock=lambda: TIME + timedelta(hours=3),
    )
    check = open_local_decision_database(path)
    try:
        assert get_decision(check, "concurrent") == settled
    finally:
        check.close()
    assert any("nothing was overwritten" in message for kind, message in conflict.messages if kind == "error")


def test_manual_refresh_failure_preserves_history_and_marks_it_stale(tmp_path, monkeypatch):
    path = tmp_path / "refresh.sqlite3"
    stored = _write_pending_to_path(path)
    state = {"football_decision_history_decisions": (stored,), "football_qb_research_result": "qb", "football_rb_research_result": "rb"}
    first = open_local_decision_database(path)
    refresh = sqlite3.connect(":memory:")
    opened = iter((first, refresh))
    monkeypatch.setattr(
        history_page,
        "list_decisions",
        lambda _connection: (_ for _ in ()).throw(DecisionStoreError("refresh failure")),
    )
    ui = FakeStreamlit(manual=True, manual_acknowledged=True, manual_result="251", state=_manual_state(stored, state))
    history_page.render_decision_history_page(
        streamlit_module=ui,
        database_opener=lambda: next(opened),
        settlement_runner=pytest.fail,
        manual_settlement_clock=lambda: TIME + timedelta(hours=3),
    )
    assert state["football_decision_history_decisions"] == (stored,)
    assert state["football_qb_research_result"] == "qb"
    assert state["football_rb_research_result"] == "rb"
    assert state["football_decision_history_may_be_stale"] is True
    assert any("may be stale" in message for kind, message in ui.messages if kind == "warning")
    for current in (first, refresh):
        with pytest.raises(sqlite3.ProgrammingError):
            current.execute("SELECT 1")


def _write_pending(tmp_path, decision_id="pending"):
    path = tmp_path / f"{decision_id}.sqlite3"
    connection = open_local_decision_database(path)
    try:
        stored = record_decision(connection, _pending(decision_id))
        connection.commit()
        return stored
    finally:
        connection.close()


def _write_pending_to_path(path, decision_id="pending-qb"):
    connection = open_local_decision_database(path)
    try:
        stored = record_decision(connection, _pending(decision_id))
        connection.commit()
        return stored
    finally:
        connection.close()


def _manual_state(stored, state=None):
    values = {} if state is None else state
    values.update({"football_decision_history_decisions": (stored,)})
    return values
