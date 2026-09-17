from datetime import datetime, timezone

import pandas as pd
from types import SimpleNamespace

from football.pipeline import WeeklyQBResearchResult, WeeklyRBResearchResult
from football.ui.football_research_page import render_football_research_page
from football.ui.football_research_page import _render_decision
from football.decisions import DuplicateDecisionError
from football.pipeline.build_weekly_player_prop_odds import WEEKLY_PLAYER_PROP_ODDS_COLUMNS
from football.ui.football_research_view import FootballResearchResult


class _Context:
    def __enter__(self): return self
    def __exit__(self, *_): return False


class FakeUI:
    def __init__(self, state, submitted=False): self.session_state, self.submitted, self.messages = state, submitted, []
    def title(self, value): self.messages.append(("title", value))
    def info(self, value): self.messages.append(("info", value))
    def caption(self, value): self.messages.append(("caption", value))
    def error(self, value): self.messages.append(("error", value))
    def warning(self, value): self.messages.append(("warning", value))
    def markdown(self, value): self.messages.append(("markdown", value))
    def write(self, *_): pass
    def form(self, *_): return _Context()
    def columns(self, values): return [_Context() for _ in values]
    def spinner(self, *_): return _Context()
    def expander(self, *_): return _Context()
    def container(self, **_): return _Context()
    def number_input(self, _, **kwargs): return kwargs["value"]
    def date_input(self, _, **kwargs): return kwargs["value"]
    def form_submit_button(self, *_args, **_kwargs): return self.submitted
    def selectbox(self, _label, options, **_kwargs): return options[-1]
    def success(self, value): self.messages.append(("success", value))


def _combined():
    qb = WeeklyQBResearchResult(pd.DataFrame(columns=["game_id", "team", "home_away", "expected_player_id", "expected_player_name"]), pd.DataFrame(), pd.DataFrame())
    rb = WeeklyRBResearchResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    return qb, rb


def test_loader_boundary_and_failed_refresh_preserves_combined_state():
    state, calls = {"football_qb_research_result": "legacy-qb", "football_rb_research_result": "legacy-rb"}, []
    ui = FakeUI(state)
    render_football_research_page(streamlit_module=ui, qb_report_loader=lambda **_: calls.append("qb"), rb_report_loader=lambda **_: calls.append("rb"))
    assert calls == []
    ui = FakeUI(state, submitted=True)
    render_football_research_page(streamlit_module=ui, qb_report_loader=lambda **_: calls.append("qb") or _combined()[0], rb_report_loader=lambda **_: calls.append("rb") or _combined()[1], clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc))
    saved = state["football_research_result"]
    assert calls == ["qb", "rb"] and isinstance(saved, FootballResearchResult)
    render_football_research_page(streamlit_module=FakeUI(state), qb_report_loader=lambda **_: (_ for _ in ()).throw(AssertionError()), rb_report_loader=lambda **_: (_ for _ in ()).throw(AssertionError()))
    assert state["football_research_result"] is saved
    render_football_research_page(streamlit_module=FakeUI(state, submitted=True), qb_report_loader=lambda **_: (_ for _ in ()).throw(RuntimeError("offline")), rb_report_loader=lambda **_: None)
    assert state["football_research_result"] is saved
    assert state["football_qb_research_result"] == "legacy-qb"
    assert state["football_rb_research_result"] == "legacy-rb"


def _decision_row(position="QB", outcome="Under", price=-115, point=250.5, book="book"):
    base = {column: pd.NA for column in WEEKLY_PLAYER_PROP_ODDS_COLUMNS}
    return {**base, "event_id": "e", "bookmaker_key": book, "bookmaker_title": "Book", "market_key": "player_pass_yds" if position == "QB" else "player_rush_yds", "player_name": position, "outcome_name": outcome, "price": price, "point": point, "nflverse_game_id": "g", "nflverse_season": 2026, "nflverse_week": 1, "nflverse_home_team": "H", "nflverse_away_team": "A", "event_match_status": "matched", "player_id": position.lower(), "nflverse_player_name": position, "nflverse_team": "A", "expected_position": position, "match_status": "matched"}


def test_decision_submit_is_injected_exact_and_connection_closes_without_loaders():
    row = _decision_row()
    person = SimpleNamespace(position="QB", player_id="qb", decision_rows=(tuple(row[column] for column in WEEKLY_PLAYER_PROP_ODDS_COLUMNS),))
    game = SimpleNamespace(game_id="g", away_team=SimpleNamespace(team="A", participants=(person,)), home_team=SimpleNamespace(team="H", participants=()))
    ui, recorded, opened = FakeUI({}, submitted=True), [], []
    class Connection:
        def close(self): opened.append("closed")
    _render_decision(ui, person, game, {"report_season": 2026, "report_week": 1}, datetime(2026, 1, 1, tzinfo=timezone.utc), lambda: opened.append("opened") or Connection(), lambda _c, decision: recorded.append(decision), lambda: datetime(2026, 1, 2, tzinfo=timezone.utc))
    assert len(recorded) == 1
    assert (recorded[0].game_id, recorded[0].player_id, recorded[0].market_key, recorded[0].line, recorded[0].selection, recorded[0].selected_price) == ("g", "qb", "player_pass_yds", 250.5, "under", -115)
    assert opened == ["opened", "closed"]


def test_rb_decision_and_duplicate_or_failure_are_local_and_close_connections():
    row = _decision_row("RB", "Over", -110, 60.5, "alternate")
    person = SimpleNamespace(position="RB", player_id="rb", decision_rows=(tuple(row[column] for column in WEEKLY_PLAYER_PROP_ODDS_COLUMNS),))
    game = SimpleNamespace(game_id="g", away_team=SimpleNamespace(team="A", participants=(person,)), home_team=SimpleNamespace(team="H", participants=()))
    for recorder, expected_kind in [(lambda _c, _d: (_ for _ in ()).throw(DuplicateDecisionError("duplicate")), "info"), (lambda _c, _d: (_ for _ in ()).throw(OSError("disk")), "error")]:
        ui, closed = FakeUI({}, submitted=True), []
        class Connection:
            def close(self): closed.append(True)
        _render_decision(ui, person, game, {"report_season": 2026, "report_week": 1}, datetime(2026, 1, 1, tzinfo=timezone.utc), Connection, recorder, lambda: datetime(2026, 1, 2, tzinfo=timezone.utc))
        assert closed == [True]
        assert any(kind == expected_kind for kind, _ in ui.messages)
