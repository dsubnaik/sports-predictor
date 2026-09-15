"""Small non-interactive tests for the QB page's Generate boundary."""

from datetime import date, datetime, timezone
from decimal import Decimal
import sqlite3

import pandas as pd
import pytest

import football.ui.qb_research_page as qb_page
from football.decision_database import open_local_decision_database
from football.decisions import get_decision, list_decisions
from football.pipeline import WeeklyQBResearchResult
from football.pipeline.build_weekly_player_prop_odds import WEEKLY_PLAYER_PROP_ODDS_COLUMNS


def test_generate_loader_requests_passing_props_on_every_deliberate_call(monkeypatch):
    calls = []
    result = WeeklyQBResearchResult(
        summary=pd.DataFrame(),
        qb_game_logs=pd.DataFrame(),
        defense_game_logs=pd.DataFrame(),
    )

    def fake_builder(**kwargs):
        calls.append(kwargs)
        return result

    monkeypatch.setattr(qb_page, "build_weekly_qb_research", fake_builder)

    assert qb_page.load_qb_research(2026, 1, date(2026, 9, 10), 2025) is result
    assert qb_page.load_qb_research(2026, 1, date(2026, 9, 10), 2025) is result
    assert len(calls) == 2
    assert all(call["include_player_props"] is True for call in calls)


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class DecisionEntryStreamlit:
    """Focused fake Streamlit surface for QB report-state and entry behavior."""

    def __init__(self, state, *, record_submitted=False, outcome_text="Over", notes=""):
        self.session_state = state
        self.record_submitted = record_submitted
        self.outcome_text = outcome_text
        self.notes = notes
        self.messages = []
        self.dataframes = []

    def title(self, value): self.messages.append(("title", value))
    def info(self, value): self.messages.append(("info", value))
    def caption(self, value): self.messages.append(("caption", value))
    def warning(self, value): self.messages.append(("warning", value))
    def error(self, value): self.messages.append(("error", value))
    def success(self, value): self.messages.append(("success", value))
    def subheader(self, value): self.messages.append(("subheader", value))
    def markdown(self, value): self.messages.append(("markdown", value))
    def write(self, value): pass
    def form(self, value): return _Context()
    def columns(self, values): return [_Context() for _ in values] if isinstance(values, list) else [_Context(), _Context()]
    def spinner(self, value): return _Context()
    def number_input(self, label, **kwargs): return kwargs["value"]
    def date_input(self, label, **kwargs): return kwargs["value"]
    def form_submit_button(self, label, **kwargs): return self.record_submitted if label == "Record Decision" else False
    def selectbox(self, label, options, **kwargs):
        if label == "Exact displayed outcome":
            formatter = kwargs["format_func"]
            return next(value for value in options if self.outcome_text in formatter(value))
        return options[0]
    def text_area(self, label, **kwargs): return self.notes
    def dataframe(self, value, **kwargs): self.dataframes.append(value)
    def stop(self): raise AssertionError("unexpected page stop")


def _summary():
    return pd.DataFrame([{
        "season": 2026, "report_week": 1, "game_id": "2026_01_KC_LAC",
        "team": "KC", "opponent": "LAC", "expected_player_id": "qb-1",
        "expected_player_name": "Quarterback", "selection_source": "depth_chart",
        "depth_chart_date": "2026-09-01", "starter_uncertain": False,
        "qb_history_season": 2025, "qb_history_cutoff_week": 19,
        "defense_history_season": 2025, "defense_history_cutoff_week": 19,
        "missing_qb_history": False, "missing_defense_history": False,
    }])


def _props(rows=None):
    base = {
        "event_id": "event-1", "commence_time": "2026-09-11T00:20:00Z",
        "home_team": "Kansas City Chiefs", "away_team": "Los Angeles Chargers",
        "bookmaker_key": "draftkings", "bookmaker_title": "DraftKings",
        "bookmaker_last_update": "2026-09-10T12:00:00Z",
        "market_key": "player_pass_yds", "market_last_update": "2026-09-10T12:01:00Z",
        "player_name": "Quarterback", "outcome_name": "Over", "price": -110,
        "point": 250.5, "nflverse_game_id": "2026_01_KC_LAC",
        "nflverse_season": 2026, "nflverse_week": 1, "nflverse_home_team": "KC",
        "nflverse_away_team": "LAC", "nflverse_kickoff_time": "2026-09-11T00:20:00Z",
        "event_match_status": "matched", "event_match_method": "team_and_kickoff",
        "event_match_candidate_count": 1, "event_match_note": "Matched", "player_id": "qb-1",
        "nflverse_player_name": "Quarterback", "nflverse_team": "KC", "expected_position": "QB",
        "match_status": "matched", "match_method": "canonical_name_and_position",
        "match_candidate_count": 1, "match_note": "Matched",
    }
    return pd.DataFrame(
        [{**base, **row} for row in (rows or [{}, {"outcome_name": "Under", "price": -115}])],
        columns=WEEKLY_PLAYER_PROP_ODDS_COLUMNS,
    )


def _report(props=None):
    return WeeklyQBResearchResult(
        summary=_summary(), qb_game_logs=pd.DataFrame(), defense_game_logs=pd.DataFrame(),
        player_prop_odds=type("Odds", (), {"player_matched_odds": _props(props)})(),
    )


def _state(report):
    return {
        "football_qb_research_inputs": {"report_season": 2026, "report_week": 1, "as_of_date": date(2026, 9, 10), "history_season": 2025},
        "football_qb_research_result": report,
        "football_qb_research_odds_retrieved_at": datetime(2026, 9, 10, 12, tzinfo=timezone.utc),
        "football_rb_research_result": "rb-state",
    }


def test_qb_selector_rerun_never_opens_database_or_reruns_report_loader():
    report = _report()
    state = _state(report)
    ui = DecisionEntryStreamlit(state)

    qb_page.render_football_qb_research_page(
        streamlit_module=ui,
        report_loader=lambda **_kwargs: pytest.fail("QB pipeline reran"),
        database_opener=lambda: pytest.fail("SQLite opened without submission"),
    )

    assert state["football_qb_research_result"] is report
    assert state["football_rb_research_result"] == "rb-state"


@pytest.mark.parametrize(
    ("outcome_text", "expected_selection", "expected_price", "notes"),
    [("Over", "over", -110, "  over note  "), ("Under", "under", -115, "   ")],
)
def test_deliberate_qb_submission_records_exact_displayed_snapshot(
    tmp_path, outcome_text, expected_selection, expected_price, notes,
):
    report = _report()
    state = _state(report)
    opened = []
    path = tmp_path / "decisions.sqlite3"

    def opener():
        connection = open_local_decision_database(path)
        opened.append(connection)
        return connection

    qb_page.render_football_qb_research_page(
        streamlit_module=DecisionEntryStreamlit(state, record_submitted=True, outcome_text=outcome_text, notes=notes),
        report_loader=lambda **_kwargs: pytest.fail("QB pipeline reran"),
        database_opener=opener,
        clock=lambda: datetime(2026, 9, 10, 13, tzinfo=timezone.utc),
        id_factory=lambda: f"decision-{expected_selection}",
    )

    check = open_local_decision_database(path)
    try:
        decision = get_decision(check, f"decision-{expected_selection}")
    finally:
        check.close()
    assert decision is not None
    assert decision.line == Decimal("250.5")
    assert decision.selected_price == expected_price
    assert decision.selection == expected_selection
    assert decision.sportsbook == "draftkings"
    assert (decision.game_id, decision.player_id, decision.season, decision.week) == ("2026_01_KC_LAC", "qb-1", 2026, 1)
    assert (decision.team, decision.opponent) == ("KC", "LAC")
    assert decision.odds_retrieved_at == datetime(2026, 9, 10, 12, tzinfo=timezone.utc)
    assert decision.recorded_at == datetime(2026, 9, 10, 13, tzinfo=timezone.utc)
    assert decision.research_notes == ("over note" if expected_selection == "over" else None)
    for connection in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")
    assert state["football_qb_research_result"] is report
    assert state["football_rb_research_result"] == "rb-state"


def test_exact_qb_duplicate_is_non_destructive_and_connections_close(tmp_path):
    path = tmp_path / "decisions.sqlite3"
    ids = iter(("first", "second"))
    messages = []
    for _ in range(2):
        state = _state(_report())
        ui = DecisionEntryStreamlit(state, record_submitted=True)
        qb_page.render_football_qb_research_page(
            streamlit_module=ui,
            database_opener=lambda: open_local_decision_database(path),
            clock=lambda: datetime(2026, 9, 10, 13, tzinfo=timezone.utc),
            id_factory=lambda: next(ids),
        )
        messages.extend(ui.messages)

    check = open_local_decision_database(path)
    try:
        assert [decision.decision_id for decision in list_decisions(check)] == ["first"]
    finally:
        check.close()
    assert any(kind == "info" and "already recorded" in text for kind, text in messages)


def test_invalid_displayed_price_closes_connection_without_changing_research_state(tmp_path):
    report = _report([{"price": 0}])
    state = _state(report)
    opened = []

    def opener():
        connection = open_local_decision_database(tmp_path / "decisions.sqlite3")
        opened.append(connection)
        return connection

    ui = DecisionEntryStreamlit(state, record_submitted=True)
    qb_page.render_football_qb_research_page(
        streamlit_module=ui, database_opener=opener,
        clock=lambda: datetime(2026, 9, 10, 13, tzinfo=timezone.utc), id_factory=lambda: "invalid",
    )

    assert any(kind == "error" and "selected_price" in text for kind, text in ui.messages)
    assert state["football_qb_research_result"] is report
    for connection in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")
