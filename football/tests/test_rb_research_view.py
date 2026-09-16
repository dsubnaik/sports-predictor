"""Tests for pure RB Streamlit display preparation helpers."""

from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
import sqlite3

import pandas as pd
import pytest

from football.pipeline import WeeklyRBResearchResult
from football.decision_database import open_local_decision_database
from football.decisions import get_decision, list_decisions
from football.pipeline.build_weekly_player_prop_odds import (
    WEEKLY_PLAYER_PROP_ODDS_COLUMNS,
)
from football.ui.rb_research_page import render_football_rb_research_page
from football.ui.rb_research_view import (
    DEFENSIVE_MATCHUP_RANK_HELP,
    build_participant_options,
    build_rb_decision_outcome_options,
    build_rb_pending_decision,
    build_selected_rb_prop_warnings,
    build_warning_counts,
    default_history_season,
    filter_defense_game_log,
    filter_rb_game_log,
    filter_selected_rb_rushing_props,
    find_participant,
    format_odds_retrieval_time,
    prepare_defense_log_display,
    prepare_rb_log_display,
    prepare_rushing_prop_display,
    prepare_summary_display,
    RUSHING_PROP_DISPLAY_COLUMNS,
    RBDecisionEntryValidationError,
)


def make_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            summary_row("id_one", "Alex Runner", "KC", "LAC", 1, 1),
            summary_row("id_two", "Alex Runner", "KC", "LAC", 2, 1),
            summary_row(pd.NA, "Unknown Back", "LAC", "KC", 1, 2, unresolved=True),
        ]
    )


def summary_row(
    player_id: object,
    player_name: str,
    team: str,
    opponent: str,
    order: int,
    rank: int,
    *,
    unresolved: bool = False,
) -> dict[str, object]:
    return {
        "report_season": 2026,
        "report_week": 1,
        "game_id": "2026_01_KC_LAC",
        "team": team,
        "opponent": opponent,
        "player_id": player_id,
        "player_name": player_name,
        "participant_order": order,
        "depth_position": "RB",
        "selection_source": "unresolved" if unresolved else "depth_chart",
        "historical_season": 2025,
        "matchup_rank": rank,
        "rb_season_rushing_yards_avg": 82.26 if not unresolved else pd.NA,
        "rb_last3_rushing_yards_avg": 92.04 if not unresolved else pd.NA,
        "rb_season_rushing_attempts_avg": 16.66 if not unresolved else pd.NA,
        "rb_last3_rushing_attempts_avg": 18.04 if not unresolved else pd.NA,
        "rb_season_opportunities_avg": 20.15 if not unresolved else pd.NA,
        "rb_last3_opportunities_avg": 23.01 if not unresolved else pd.NA,
        "rb_season_yards_per_carry": 4.94 if not unresolved else pd.NA,
        "defense_season_rb_rushing_yards_avg_allowed": 112.88,
        "defense_last3_rb_rushing_yards_avg_allowed": 119.15,
        "defense_season_rb_yards_per_carry_allowed": 4.8,
        "rb_season_games": 17 if not unresolved else pd.NA,
        "defense_season_games": 17,
        "rb_history_missing": unresolved,
        "defense_history_missing": False,
        "participant_resolution_missing": unresolved,
        "participant_team_mismatch": False,
        "multiple_expected_rbs": not unresolved,
        "limited_rb_sample": False,
        "limited_defense_sample": False,
    }


def make_rb_logs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            rb_log("id_one", 2, "2025_02_KC_LAC"),
            rb_log("id_two", 1, "2025_01_KC_DEN"),
            rb_log("id_one", 1, "2025_01_KC_DEN"),
        ]
    )


def rb_log(player_id: str, week: int, game_id: str) -> dict[str, object]:
    return {
        "season": 2025,
        "week": week,
        "game_id": game_id,
        "player_id": player_id,
        "team": "KC",
        "opponent": "LAC",
        "rushing_attempts": 10,
        "rushing_yards": 0,
        "rushing_touchdowns": 1,
        "receptions": 2,
        "targets": 3,
        "receiving_yards": 14,
        "receiving_touchdowns": 0,
        "opportunities": 13,
        "carry_share": 0.5,
        "target_share": 0.2,
        "opportunity_share": 0.4,
        "low_volume_rb": False,
        "shared_backfield": True,
    }


def make_defense_logs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"season": 2025, "week": 2, "game_id": "g2", "defense": "LAC", "offense_team": "KC", "rb_rushing_yards_allowed": 80},
            {"season": 2025, "week": 1, "game_id": "g1", "defense": "LAC", "offense_team": "DEN", "rb_rushing_yards_allowed": 90},
            {"season": 2025, "week": 1, "game_id": "g3", "defense": "KC", "offense_team": "LAC", "rb_rushing_yards_allowed": 70},
        ]
    )


def test_summary_display_preserves_multiple_expected_backfield_participants():
    display = prepare_summary_display(make_summary())

    assert display["Expected Backfield Participant"].tolist() == ["Alex Runner", "Alex Runner", "Unknown Back"]
    assert display.loc[0, "RB Season Rush Yds Avg"] == 82.3
    assert "Defensive Matchup Rank" in display


def test_resolved_selector_uses_stable_player_id_and_excludes_unresolved_rows():
    summary = make_summary()
    options = build_participant_options(summary)

    assert len(options) == 2
    assert options[0].option_id != options[1].option_id
    assert all("Unknown Back" not in option.label for option in options)
    assert find_participant(summary, options[1].option_id)["player_id"] == "id_two"


def test_rb_log_filters_by_player_id_not_name_and_keeps_real_zeroes():
    summary = make_summary()
    selected = summary.iloc[0]

    result = filter_rb_game_log(make_rb_logs(), selected)
    display = prepare_rb_log_display(result)

    assert result["player_id"].tolist() == ["id_one", "id_one"]
    assert result["week"].tolist() == [1, 2]
    assert display.loc[0, "Rushing Yards"] == 0


def test_defense_log_filters_by_selected_scheduled_opponent():
    result = filter_defense_game_log(make_defense_logs(), make_summary().iloc[0])

    assert result["defense"].tolist() == ["LAC", "LAC"]
    assert result["week"].tolist() == [1, 2]
    assert prepare_defense_log_display(result).columns.tolist()[:3] == ["Season", "Week", "Offense Faced"]


def test_warning_counts_distinguish_all_report_warning_types():
    summary = make_summary()
    summary.loc[0, "participant_team_mismatch"] = True
    summary.loc[1, "limited_rb_sample"] = True
    summary.loc[2, "limited_defense_sample"] = True

    counts = build_warning_counts(summary)

    assert counts == {
        "Missing RB history": 1,
        "Missing defense history": 0,
        "Unresolved participants": 1,
        "Historical-team mismatches": 1,
        "Shared backfields": 2,
        "Limited RB samples": 1,
        "Limited defense samples": 1,
    }


def test_default_history_season_matches_pipeline_rule():
    assert default_history_season(2026, 1) == 2025
    assert default_history_season(2026, 2) == 2026


def test_display_helpers_do_not_mutate_pipeline_dataframes():
    summary = make_summary()
    rb_logs = make_rb_logs()
    defense_logs = make_defense_logs()
    original_summary = summary.copy(deep=True)
    original_rb_logs = rb_logs.copy(deep=True)
    original_defense_logs = defense_logs.copy(deep=True)

    prepare_summary_display(summary)
    build_participant_options(summary)
    filter_rb_game_log(rb_logs, summary.iloc[0])
    filter_defense_game_log(defense_logs, summary.iloc[0])

    pd.testing.assert_frame_equal(summary, original_summary)
    pd.testing.assert_frame_equal(rb_logs, original_rb_logs)
    pd.testing.assert_frame_equal(defense_logs, original_defense_logs)


def test_rank_help_is_not_a_prediction_or_talent_rank():
    assert "Rank 1" in DEFENSIVE_MATCHUP_RANK_HELP
    assert "most RB rushing yards per game" in DEFENSIVE_MATCHUP_RANK_HELP
    assert "not an RB talent ranking, projection, or composite score" in DEFENSIVE_MATCHUP_RANK_HELP


def test_app_declares_grouped_top_navigation_without_analytics():
    app = Path("app.py").read_text(encoding="utf-8")

    assert '"Baseball": [' in app
    assert '"Football": [' in app
    assert 'title="QB Research"' in app
    assert 'title="RB Research"' in app
    assert "default=True" in app
    assert "position=\"top\"" in app
    assert "sidebar" not in app.lower()
    assert "build_weekly_rb_research" not in app


@pytest.mark.parametrize("player_id", ["", "   ", pd.NA])
def test_blank_or_missing_id_is_not_selectable(player_id: object):
    summary = make_summary()
    summary.loc[0, "player_id"] = player_id

    assert len(build_participant_options(summary)) == 1


class _StopRendering(Exception):
    pass


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeStreamlit:
    """Minimal injected Streamlit surface for page-controller behavior."""

    def __init__(self, *, submitted: bool, session_state: dict[str, object] | None = None):
        self.submitted = submitted
        self.session_state = {} if session_state is None else session_state
        self.messages: list[tuple[str, str]] = []
        self.dataframes: list[pd.DataFrame] = []

    def title(self, value): self.messages.append(("title", value))
    def info(self, value): self.messages.append(("info", value))
    def caption(self, value): self.messages.append(("caption", value))
    def warning(self, value): self.messages.append(("warning", value))
    def error(self, value): self.messages.append(("error", value))
    def subheader(self, value): self.messages.append(("subheader", value))
    def markdown(self, value): self.messages.append(("markdown", value))
    def write(self, value): pass
    def form(self, value): return _Context()
    def columns(self, values): return [_Context() for _ in values] if isinstance(values, list) else [_Context(), _Context()]
    def spinner(self, value): return _Context()
    def number_input(self, label, **kwargs): return kwargs["value"]
    def date_input(self, label, **kwargs): return kwargs["value"]
    def form_submit_button(self, label, **kwargs): return self.submitted
    def selectbox(self, label, options, key):
        return self.session_state.get(key, options[0])
    def dataframe(self, value, **kwargs): self.dataframes.append(value)
    def stop(self): raise _StopRendering()


def make_result() -> WeeklyRBResearchResult:
    return WeeklyRBResearchResult(
        summary=make_summary(),
        rb_game_logs=make_rb_logs(),
        defense_game_logs=make_defense_logs(),
    )


def test_page_does_not_run_pipeline_until_form_submission():
    calls: list[dict[str, object]] = []
    ui = FakeStreamlit(submitted=False)

    with pytest.raises(_StopRendering):
        render_football_rb_research_page(
            streamlit_module=ui,
            report_loader=lambda **kwargs: calls.append(kwargs),
        )

    assert calls == []
    assert "football_rb_research_result" not in ui.session_state


def test_page_generates_once_and_preserves_qb_session_state():
    calls: list[dict[str, object]] = []
    ui = FakeStreamlit(submitted=True, session_state={"football_qb_research_result": "qb"})

    def loader(**kwargs):
        calls.append(kwargs)
        return make_result()

    render_football_rb_research_page(streamlit_module=ui, report_loader=loader)

    assert len(calls) == 1
    assert calls[0]["report_week"] == 1
    assert calls[0]["history_season"] == calls[0]["report_season"] - 1
    assert calls[0]["include_player_props"] is True
    assert ui.session_state["football_qb_research_result"] == "qb"
    assert isinstance(ui.session_state["football_rb_research_result"], WeeklyRBResearchResult)
    assert ui.session_state["football_rb_research_odds_retrieved_at"].tzinfo is not None


def test_each_deliberate_generate_calls_the_rb_loader_again():
    calls: list[dict[str, object]] = []
    state: dict[str, object] = {}

    def loader(**kwargs):
        calls.append(kwargs)
        return make_result()

    render_football_rb_research_page(
        streamlit_module=FakeStreamlit(submitted=True, session_state=state),
        report_loader=loader,
    )
    render_football_rb_research_page(
        streamlit_module=FakeStreamlit(submitted=True, session_state=state),
        report_loader=loader,
    )

    assert len(calls) == 2
    assert all(call["include_player_props"] is True for call in calls)


def test_page_rerun_with_existing_result_does_not_reload_pipeline():
    result = make_result()
    ui = FakeStreamlit(
        submitted=False,
        session_state={
            "football_rb_research_inputs": {
                "report_season": 2026,
                "report_week": 1,
                "as_of_date": pd.Timestamp("2026-09-01").date(),
                "history_season": 2025,
            },
            "football_rb_research_result": result,
        },
    )

    render_football_rb_research_page(
        streamlit_module=ui,
        report_loader=lambda **kwargs: pytest.fail("pipeline reran during detail interaction"),
    )

    assert len(ui.dataframes) == 3


def test_page_does_not_swallow_unexpected_type_error():
    ui = FakeStreamlit(submitted=True)

    with pytest.raises(TypeError, match="developer defect"):
        render_football_rb_research_page(
            streamlit_module=ui,
            report_loader=lambda **kwargs: (_ for _ in ()).throw(TypeError("developer defect")),
        )


def test_failed_generation_preserves_existing_rb_state_and_qb_state():
    prior_result = make_result()
    prior_inputs = {
        "report_season": 2026,
        "report_week": 1,
        "as_of_date": pd.Timestamp("2026-09-01").date(),
        "history_season": 2025,
    }
    prior_time = pd.Timestamp("2026-09-10T12:00:00Z")
    state = {
        "football_rb_research_inputs": prior_inputs,
        "football_rb_research_result": prior_result,
        "football_rb_research_odds_retrieved_at": prior_time,
        "football_rb_research_selected_participant": "prior-option",
        "football_qb_research_result": "qb",
    }
    ui = FakeStreamlit(submitted=True, session_state=state)

    with pytest.raises(_StopRendering):
        render_football_rb_research_page(
            streamlit_module=ui,
            report_loader=lambda **_kwargs: (_ for _ in ()).throw(ValueError("bad odds payload")),
        )

    assert ui.session_state["football_rb_research_inputs"] is prior_inputs
    assert ui.session_state["football_rb_research_result"] is prior_result
    assert ui.session_state["football_rb_research_odds_retrieved_at"] == prior_time
    assert ui.session_state["football_rb_research_selected_participant"] == "prior-option"
    assert ui.session_state["football_qb_research_result"] == "qb"


def make_player_matched_odds(rows: list[dict[str, object]] | None = None) -> pd.DataFrame:
    default = {
        "event_id": "event-kc", "commence_time": "2026-09-11T00:20:00Z",
        "home_team": "Kansas City Chiefs", "away_team": "Los Angeles Chargers",
        "bookmaker_key": "draftkings", "bookmaker_title": "DraftKings",
        "bookmaker_last_update": "2026-09-10T12:00:00Z",
        "market_key": "player_rush_yds", "market_last_update": "2026-09-10T12:01:00Z",
        "player_name": "Alex Runner", "outcome_name": "Over", "price": -110,
        "point": 55.5, "nflverse_game_id": "2026_01_KC_LAC",
        "nflverse_season": 2026, "nflverse_week": 1, "nflverse_home_team": "KC",
        "nflverse_away_team": "LAC", "nflverse_kickoff_time": "2026-09-11T00:20:00Z",
        "event_match_status": "matched", "event_match_method": "team_and_kickoff",
        "event_match_candidate_count": 1, "event_match_note": "Matched",
        "player_id": "id_one", "nflverse_player_name": "Alex Runner",
        "nflverse_team": "KC", "expected_position": "RB", "match_status": "matched",
        "match_method": "canonical_name_and_position", "match_candidate_count": 1,
        "match_note": "Matched",
    }
    return pd.DataFrame(
        [{**default, **row} for row in (rows or [{}, {"outcome_name": "Under", "price": -115}])],
        columns=WEEKLY_PLAYER_PROP_ODDS_COLUMNS,
    )


def test_selected_rushing_props_require_game_and_player_identity_and_keep_backups_distinct():
    summary = make_summary()
    participant = summary.loc[summary["player_id"].eq("id_two")].iloc[0]
    props = make_player_matched_odds([
        {"player_id": "id_two", "player_name": "Alex Runner"},
        {"player_id": "id_two", "player_name": "Alex Runner", "outcome_name": "Under", "price": -115},
        {"player_id": "id_one"},
        {"nflverse_game_id": "2026_01_OTHER", "player_id": "id_two"},
        {"market_key": "player_pass_yds", "player_id": "id_two"},
        {"match_status": "unmatched", "player_id": "id_two"},
        {"event_match_status": "unmatched", "player_id": "id_two"},
    ])
    before = props.copy(deep=True)

    result = filter_selected_rb_rushing_props(props, participant)

    assert result["player_id"].tolist() == ["id_two", "id_two"]
    assert result["outcome_name"].tolist() == ["Over", "Under"]
    pd.testing.assert_frame_equal(props, before)
    unresolved = participant.copy()
    unresolved["player_id"] = pd.NA
    assert filter_selected_rb_rushing_props(props, unresolved).empty
    no_game = participant.copy()
    no_game["game_id"] = " "
    assert filter_selected_rb_rushing_props(props, no_game).empty


def test_rushing_prop_display_pairs_sides_keeps_points_and_rejects_duplicate_sides():
    props = make_player_matched_odds([
        {"bookmaker_key": "fan", "bookmaker_title": "FanDuel", "point": 56.5, "outcome_name": "Over", "price": 105},
        {"bookmaker_key": "fan", "bookmaker_title": "FanDuel", "point": 56.5, "outcome_name": "Under", "price": -125},
        {"bookmaker_key": "draft", "bookmaker_title": "DraftKings", "point": 55.5, "outcome_name": "Over", "price": -110},
        {"bookmaker_key": "draft", "bookmaker_title": "DraftKings", "point": 55.5, "outcome_name": "Under", "price": -115},
        {"bookmaker_key": "draft", "bookmaker_title": "DraftKings", "point": 57.5, "outcome_name": "Over", "price": 100},
    ])
    before = props.copy(deep=True)
    display = prepare_rushing_prop_display(props)

    assert display.columns.tolist() == RUSHING_PROP_DISPLAY_COLUMNS
    assert display["Sportsbook"].tolist() == ["DraftKings", "DraftKings", "FanDuel"]
    assert display["Rushing Yards Line"].tolist() == [55.5, 57.5, 56.5]
    assert display.loc[0, ["Over Price", "Under Price"]].tolist() == [-110, -115]
    assert pd.isna(display.loc[1, "Under Price"])
    assert display.loc[0, "Market Updated"] == "2026-09-10T12:01:00Z"
    pd.testing.assert_frame_equal(props, before)
    with pytest.raises(ValueError, match="Duplicate sportsbook outcome"):
        prepare_rushing_prop_display(make_player_matched_odds([{}, {"price": -120}]))


def test_rushing_prop_warnings_are_game_scoped_and_retrieval_time_is_deterministic():
    summary = make_summary()
    participant = summary.loc[summary["player_id"].eq("id_one")].iloc[0]
    props = make_player_matched_odds([
        {"match_status": "unmatched"},
        {"match_status": "ambiguous", "player_name": "Other RB"},
        {"nflverse_game_id": "2026_01_OTHER", "nflverse_home_team": "BUF", "nflverse_away_team": "MIA", "event_match_status": "ambiguous"},
    ])
    odds_result = type("OddsResult", (), {"player_matched_odds": props})()
    warnings = build_selected_rb_prop_warnings(odds_result, participant)

    assert any("could not be matched to nflverse" in warning for warning in warnings)
    assert any("ambiguous nflverse matches" in warning for warning in warnings)
    assert not any("multiple schedule games" in warning for warning in warnings)
    unresolved = participant.copy()
    unresolved["player_id"] = pd.NA
    assert build_selected_rb_prop_warnings(odds_result, unresolved) == [
        "Selected participant is unresolved, so rushing-yard lines cannot be matched."
    ]
    assert build_selected_rb_prop_warnings(None, participant) == [
        "Rushing-yard odds were not requested for this report."
    ]
    assert format_odds_retrieval_time("2026-09-10T12:34:56-05:00") == "2026-09-10 17:34 UTC"
    with pytest.raises(ValueError, match="timezone-aware"):
        format_odds_retrieval_time("2026-09-10T12:34:56")


class DecisionEntryStreamlit:
    """Focused fake Streamlit surface for deliberate RB decision entry."""

    def __init__(self, state, *, record_submitted=False, outcome_text="Over", notes=""):
        self.session_state = state
        self.record_submitted = record_submitted
        self.outcome_text = outcome_text
        self.notes = notes
        self.messages = []

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
        if label == "Expected backfield participant":
            return next(value for value in options if "depth 2" in value)
        if label == "Exact displayed outcome":
            formatter = kwargs["format_func"]
            return next(value for value in options if self.outcome_text in formatter(value))
        return options[0]
    def text_area(self, label, **kwargs): return self.notes
    def dataframe(self, value, **kwargs): pass
    def stop(self): raise AssertionError("unexpected page stop")


def _decision_report(rows=None, *, low_volume=False):
    summary = make_summary()
    summary.loc[summary["player_id"].eq("id_two"), "limited_rb_sample"] = low_volume
    props = make_player_matched_odds([
        {"player_id": "id_two", "nflverse_player_name": "Alex Runner", "outcome_name": "Over", "price": -110},
        {"player_id": "id_two", "nflverse_player_name": "Alex Runner", "outcome_name": "Under", "price": -115},
    ] if rows is None else rows)
    return WeeklyRBResearchResult(
        summary=summary,
        rb_game_logs=make_rb_logs(),
        defense_game_logs=make_defense_logs(),
        player_prop_odds=type("Odds", (), {"player_matched_odds": props})(),
    )


def _decision_state(report):
    return {
        "football_rb_research_inputs": {"report_season": 2026, "report_week": 1, "as_of_date": date(2026, 9, 10), "history_season": 2025},
        "football_rb_research_result": report,
        "football_rb_research_odds_retrieved_at": datetime(2026, 9, 10, 12, tzinfo=timezone.utc),
        "football_qb_research_result": "qb-state",
    }


def test_rb_decision_options_and_snapshot_keep_backup_identity_and_source_unchanged():
    participant = make_summary().loc[lambda rows: rows["player_id"].eq("id_two")].iloc[0]
    props = make_player_matched_odds([
        {"player_id": "id_two", "nflverse_player_name": "Alex Runner"},
        {"player_id": "id_two", "nflverse_player_name": "Alex Runner", "outcome_name": "Under", "price": -115},
    ])
    before = props.copy(deep=True)
    options = build_rb_decision_outcome_options(props)
    decision = build_rb_pending_decision(
        props, participant, next(option.option_id for option in options if "Over -110" in option.label),
        datetime(2026, 9, 10, 12, tzinfo=timezone.utc),
        datetime(2026, 9, 10, 13, tzinfo=timezone.utc),
    )
    assert (decision.position, decision.market_key, decision.player_id, decision.game_id) == (
        "RB", "player_rush_yds", "id_two", "2026_01_KC_LAC",
    )
    assert (decision.player_name, decision.team, decision.opponent, decision.season, decision.week) == (
        "Alex Runner", "KC", "LAC", 2026, 1,
    )
    pd.testing.assert_frame_equal(props, before)


def test_rb_decision_snapshot_rejects_context_conflict_and_duplicate_identity_conflict():
    participant = make_summary().loc[lambda rows: rows["player_id"].eq("id_two")].iloc[0]
    props = make_player_matched_odds([{"player_id": "id_two", "nflverse_player_name": "Alex Runner"}])
    option = build_rb_decision_outcome_options(props)[0]
    changed = participant.copy()
    changed["opponent"] = "DEN"
    with pytest.raises(RBDecisionEntryValidationError, match="opponent"):
        build_rb_pending_decision(
            props,
            changed,
            option.option_id,
            datetime(2026, 9, 10, 12, tzinfo=timezone.utc),
            datetime(2026, 9, 10, 13, tzinfo=timezone.utc),
        )
    conflict = make_player_matched_odds([
        {"player_id": "id_two", "nflverse_player_name": "Alex Runner"},
        {"player_id": "id_two", "nflverse_player_name": "Different Runner"},
    ])
    with pytest.raises(RBDecisionEntryValidationError, match="conflicting"):
        build_rb_decision_outcome_options(conflict)


def test_rb_selector_rerun_never_records_opens_database_or_reloads_backend():
    report = _decision_report()
    state = _decision_state(report)
    render_football_rb_research_page(
        streamlit_module=DecisionEntryStreamlit(state),
        report_loader=lambda **_kwargs: pytest.fail("RB backend reran"),
        database_opener=lambda: pytest.fail("SQLite opened without submission"),
    )
    assert state["football_rb_research_result"] is report
    assert state["football_qb_research_result"] == "qb-state"


@pytest.mark.parametrize(
    ("outcome_text", "selection", "price", "notes"),
    [("Over", "over", -110, "  backup note  "), ("Under", "under", -115, "   ")],
)
def test_deliberate_backup_rb_submission_records_exact_snapshot(tmp_path, outcome_text, selection, price, notes):
    path = tmp_path / "decisions.sqlite3"
    report = _decision_report(low_volume=True)
    state = _decision_state(report)
    opened = []

    def opener():
        connection = open_local_decision_database(path)
        opened.append(connection)
        return connection

    render_football_rb_research_page(
        streamlit_module=DecisionEntryStreamlit(state, record_submitted=True, outcome_text=outcome_text, notes=notes),
        report_loader=lambda **_kwargs: pytest.fail("RB backend reran"), database_opener=opener,
        clock=lambda: datetime(2026, 9, 10, 13, tzinfo=timezone.utc), id_factory=lambda: f"rb-{selection}",
    )
    check = open_local_decision_database(path)
    try:
        decision = get_decision(check, f"rb-{selection}")
    finally:
        check.close()
    assert decision is not None
    assert (decision.line, decision.selected_price, decision.selection, decision.sportsbook) == (Decimal("55.5"), price, selection, "draftkings")
    assert (decision.player_id, decision.game_id, decision.team, decision.opponent) == ("id_two", "2026_01_KC_LAC", "KC", "LAC")
    assert decision.odds_retrieved_at == datetime(2026, 9, 10, 12, tzinfo=timezone.utc)
    assert decision.recorded_at == datetime(2026, 9, 10, 13, tzinfo=timezone.utc)
    assert decision.research_notes == ("backup note" if selection == "over" else None)
    for connection in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")
    assert state["football_rb_research_result"] is report
    assert state["football_qb_research_result"] == "qb-state"


def test_rb_exact_duplicate_is_non_destructive_and_context_failure_never_opens_sqlite(tmp_path):
    path = tmp_path / "decisions.sqlite3"
    ids = iter(("first", "second"))
    messages = []
    for _ in range(2):
        ui = DecisionEntryStreamlit(_decision_state(_decision_report()), record_submitted=True)
        render_football_rb_research_page(
            streamlit_module=ui, database_opener=lambda: open_local_decision_database(path),
            clock=lambda: datetime(2026, 9, 10, 13, tzinfo=timezone.utc), id_factory=lambda: next(ids),
        )
        messages.extend(ui.messages)
    check = open_local_decision_database(path)
    try:
        assert [decision.decision_id for decision in list_decisions(check)] == ["first"]
    finally:
        check.close()
    assert any(kind == "info" and "already recorded" in message for kind, message in messages)

    bad = _decision_report([{"player_id": "id_two", "nflverse_player_name": "Alex Runner", "nflverse_team": "DEN"}])
    render_football_rb_research_page(
        streamlit_module=DecisionEntryStreamlit(_decision_state(bad), record_submitted=True),
        database_opener=lambda: pytest.fail("context mismatch opened SQLite"),
        clock=lambda: datetime(2026, 9, 10, 13, tzinfo=timezone.utc),
    )
