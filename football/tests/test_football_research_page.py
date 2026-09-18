from datetime import datetime, timezone

import pandas as pd
import pytest
from types import SimpleNamespace

from football.pipeline import WeeklyQBResearchResult, WeeklyRBResearchResult
from football.ui.football_research_page import format_kickoff_central, render_football_research_page
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
    def expander(self, *args, **kwargs): self.messages.append(("expander", (args, kwargs))); return _Context()
    def container(self, **_): return _Context()
    def number_input(self, _, **kwargs): return kwargs["value"]
    def date_input(self, _, **kwargs): return kwargs["value"]
    def form_submit_button(self, *_args, **_kwargs): return self.submitted
    def selectbox(self, _label, options, **_kwargs): return options[-1]
    def success(self, value): self.messages.append(("success", value))
    def button(self, _label, **_kwargs): return False


class StrictState(dict):
    """Raises if page code writes to a widget-owned key after creation."""
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs); self.instantiated = set()
    def __setitem__(self, key, value):
        if key in self.instantiated:
            raise AssertionError(f"post-widget mutation: {key}")
        super().__setitem__(key, value)
    def widget_value(self, key, default):
        if key not in self: super().__setitem__(key, default)
        self.instantiated.add(key)
        return self[key]


def _two_game_combined():
    qb = WeeklyQBResearchResult(pd.DataFrame([
        {"game_id": "g1", "team": "A", "opponent": "H", "home_away": "away", "game_date": "2026-09-13", "game_time": "13:00", "expected_player_id": "q1", "expected_player_name": "QB One"},
        {"game_id": "g2", "team": "B", "opponent": "I", "home_away": "away", "game_date": "2026-09-14", "game_time": "13:00", "expected_player_id": "q2", "expected_player_name": "QB Two"},
    ]), pd.DataFrame(), pd.DataFrame())
    return FootballResearchResult(qb, WeeklyRBResearchResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))


def test_selected_game_widget_state_is_initialized_before_pills_and_never_written_afterward():
    state = StrictState({"football_research_inputs": {"report_season": 2026, "report_week": 1, "as_of_date": datetime(2026, 9, 1).date(), "history_season": 2025}, "football_research_result": _two_game_combined()})
    class PillsUI(FakeUI):
        def pills(self, _label, options, *, key, **kwargs):
            return self.session_state.widget_value(key, kwargs["default"])
    ui = PillsUI(state)
    render_football_research_page(streamlit_module=ui, qb_report_loader=lambda **_: (_ for _ in ()).throw(AssertionError()), rb_report_loader=lambda **_: (_ for _ in ()).throw(AssertionError()), selected_game_odds_loader=lambda *_: (_ for _ in ()).throw(AssertionError()))
    assert state["football_research_selected_game_id"] == "g1"
    assert any("QB One" in message for kind, message in ui.messages if kind == "markdown")


def test_valid_and_invalid_selection_are_resolved_before_selectbox_without_loaders():
    inputs = {"report_season": 2026, "report_week": 1, "as_of_date": datetime(2026, 9, 1).date(), "history_season": 2025}
    for initial, expected in (("g2", "g2"), ("missing", "g1")):
        state = StrictState({"football_research_inputs": inputs, "football_research_result": _two_game_combined(), "football_research_selected_game_id": initial})
        class SelectUI(FakeUI):
            def selectbox(self, _label, options, *, key=None, **kwargs):
                return self.session_state.widget_value(key, options[0]) if key == "football_research_selected_game_id" else options[0]
        ui = SelectUI(state)
        render_football_research_page(streamlit_module=ui, qb_report_loader=lambda **_: (_ for _ in ()).throw(AssertionError()), rb_report_loader=lambda **_: (_ for _ in ()).throw(AssertionError()), selected_game_odds_loader=lambda *_: (_ for _ in ()).throw(AssertionError()))
        assert state["football_research_selected_game_id"] == expected


@pytest.mark.parametrize(("value", "expected"), [
    ("2026-09-18T00:15:00Z", "Thu 7:15 PM CT"),
    ("2026-12-07T02:15:00Z", "Sun 8:15 PM CT"),
    ("2026-09-20T17:00:00Z", "Sun 12:00 PM CT"),
    ("2026-09-20T05:00:00Z", "Sun 12:00 AM CT"),
])
def test_kickoff_formatter_uses_twelve_hour_dst_aware_central_time(value, expected):
    assert format_kickoff_central(value) == expected


def test_kickoff_formatter_handles_missing_invalid_and_naive_schedule_values():
    assert format_kickoff_central(None) is None
    assert format_kickoff_central("not-a-time") is None
    assert format_kickoff_central("2026-09-17 20:15") == "Thu 7:15 PM CT"


def test_compact_header_uses_caption_and_settings_disclosure_state():
    first = FakeUI({})
    render_football_research_page(streamlit_module=first)
    assert first.messages[1][0] == "caption"
    assert first.messages[1][1].startswith("Research context only ")
    assert not any(kind == "info" and "prediction" in value for kind, value in first.messages)
    assert next(value for kind, value in first.messages if kind == "expander")[1]["expanded"] is True
    state = {"football_research_inputs": {"report_season": 2026, "report_week": 2, "as_of_date": datetime(2026, 9, 18).date(), "history_season": 2026}, "football_research_result": _two_game_combined()}
    loaded = FakeUI(state)
    render_football_research_page(streamlit_module=loaded)
    assert next(value for kind, value in loaded.messages if kind == "expander")[1]["expanded"] is False
    assert any(kind == "caption" and "2026" in value and "Week 2" in value and "History 2026" in value for kind, value in loaded.messages)


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


def test_research_generation_disables_props_and_selected_game_load_is_explicit_and_cached():
    calls, odds_calls = [], []
    qb = WeeklyQBResearchResult(pd.DataFrame([{"game_id": "g", "team": "A", "opponent": "H", "home_away": "away", "game_date": "2026-09-13", "game_time": "13:00", "expected_player_id": "q", "expected_player_name": "QB"}]), pd.DataFrame(), pd.DataFrame())
    rb = WeeklyRBResearchResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    state = {}
    render_football_research_page(streamlit_module=FakeUI(state, submitted=True), qb_report_loader=lambda **kwargs: calls.append(kwargs) or qb, rb_report_loader=lambda **kwargs: calls.append(kwargs) or rb)
    assert [call["include_player_props"] for call in calls] == [False, False]
    class LoadUI(FakeUI):
        def button(self, label, **_kwargs): return label.startswith("Load")
    snapshot = SimpleNamespace(retrieved_at="now", quota=SimpleNamespace(requests_last=2, requests_remaining=98, requests_used=2), markets=("player_pass_yds", "player_rush_yds"), bookmakers=("draftkings", "fanduel", "betrivers"), player_matched_odds=SimpleNamespace(to_frame=lambda: pd.DataFrame(columns=WEEKLY_PLAYER_PROP_ODDS_COLUMNS)), event_matches=SimpleNamespace(to_frame=lambda: pd.DataFrame(columns=["event_id", "commence_time", "home_team", "away_team", "nflverse_game_id", "nflverse_season", "nflverse_week", "nflverse_home_team", "nflverse_away_team", "nflverse_kickoff_time", "event_match_status", "event_match_method", "event_match_candidate_count", "event_match_note"])))
    render_football_research_page(streamlit_module=LoadUI(state), selected_game_odds_loader=lambda *args: odds_calls.append(args) or snapshot)
    assert len(odds_calls) == 1 and odds_calls[0][0] == "g"
    render_football_research_page(streamlit_module=FakeUI(state), selected_game_odds_loader=lambda *_: (_ for _ in ()).throw(AssertionError("unexpected odds reload")))
    assert len(odds_calls) == 1


def test_failed_selected_game_refresh_retains_prior_snapshot():
    state = {"football_research_inputs": {"report_season": 2026, "report_week": 1, "as_of_date": datetime(2026, 9, 1).date(), "history_season": 2025}}
    qb = WeeklyQBResearchResult(pd.DataFrame([{"game_id": "g", "team": "A", "opponent": "H", "home_away": "away", "game_date": "2026-09-13", "game_time": "13:00", "expected_player_id": "q", "expected_player_name": "QB"}]), pd.DataFrame(), pd.DataFrame())
    state["football_research_result"] = FootballResearchResult(qb, WeeklyRBResearchResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))
    old = SimpleNamespace(retrieved_at="old", quota=SimpleNamespace(requests_last=None, requests_remaining=None, requests_used=None), markets=("player_pass_yds", "player_rush_yds"), bookmakers=("draftkings", "fanduel", "betrivers"), player_matched_odds=SimpleNamespace(to_frame=lambda: pd.DataFrame(columns=WEEKLY_PLAYER_PROP_ODDS_COLUMNS)), event_matches=SimpleNamespace(to_frame=lambda: pd.DataFrame(columns=["event_id", "commence_time", "home_team", "away_team", "nflverse_game_id", "nflverse_season", "nflverse_week", "nflverse_home_team", "nflverse_away_team", "nflverse_kickoff_time", "event_match_status", "event_match_method", "event_match_candidate_count", "event_match_note"])))
    state["football_research_game_odds"] = {"g": old}
    class RefreshUI(FakeUI):
        def button(self, label, **_kwargs): return label.startswith("Refresh")
    ui = RefreshUI(state)
    render_football_research_page(streamlit_module=ui, selected_game_odds_loader=lambda *_: (_ for _ in ()).throw(RuntimeError("quota exhausted")))
    assert state["football_research_game_odds"]["g"] is old
    assert any(kind == "error" and "previously retrieved" in value for kind, value in ui.messages)


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
