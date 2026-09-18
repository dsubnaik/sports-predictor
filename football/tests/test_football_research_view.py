from datetime import date, datetime, timezone

import pandas as pd
import pytest

from football.pipeline import WeeklyQBResearchResult, WeeklyRBResearchResult
from football.pipeline.build_weekly_player_prop_odds import WEEKLY_PLAYER_PROP_ODDS_COLUMNS
from football.ui.football_research_view import FootballResearchResult, prepare_game_research


def _summary(position, rows):
    common = {"game_date": "2026-09-13", "game_time": "13:00"}
    if position == "QB":
        return pd.DataFrame([{**common, "game_id": game, "team": team, "home_away": side, "expected_player_id": player, "expected_player_name": name} for game, team, side, player, name, *_ in rows])
    return pd.DataFrame([{**common, "game_id": game, "team": team, "home_away": side, "player_id": player, "player_name": name, "participant_order": order, "participant_resolution_missing": unresolved, "limited_rb_sample": low} for game, team, side, player, name, order, unresolved, low in rows])


def _odds(rows=(), events=()):
    base = {column: pd.NA for column in WEEKLY_PLAYER_PROP_ODDS_COLUMNS}
    odds = pd.DataFrame([{**base, **row} for row in rows], columns=WEEKLY_PLAYER_PROP_ODDS_COLUMNS)
    event_frame = pd.DataFrame(list(events), columns=["event_id", "nflverse_game_id", "event_match_status"])
    return type("Odds", (), {"player_matched_odds": odds, "event_matches": event_frame})()


def _result(qb_rows=(), rb_rows=(), qb_odds=None, rb_odds=None):
    return FootballResearchResult(
        WeeklyQBResearchResult(_summary("QB", qb_rows), pd.DataFrame(), pd.DataFrame(), qb_odds) if qb_rows is not None else None,
        WeeklyRBResearchResult(_summary("RB", rb_rows), pd.DataFrame(), pd.DataFrame(), rb_odds) if rb_rows is not None else None,
    )


def _line(game="g1", player="qb1", market="player_pass_yds", outcome="Over", **extra):
    return {"nflverse_game_id": game, "player_id": player, "market_key": market, "match_status": "matched", "event_match_status": "matched", "bookmaker_title": "DraftKings", "point": 250.5, "outcome_name": outcome, "price": -110, "market_last_update": "2026-09-12T00:00:00Z", **extra}


def test_groups_stably_by_game_with_away_home_qb_and_all_rbs():
    result = _result(
        [("g1", "HOME", "home", "qb-home", "Home QB"), ("g1", "AWAY", "away", "qb-away", "Away QB")],
        [("g1", "AWAY", "away", "rb2", "Backup", 2, False, True), ("g1", "AWAY", "away", "rb1", "Lead", 1, False, False), ("g1", "HOME", "home", "rb3", "Home RB", 1, False, False)],
    )
    game = prepare_game_research(result, None)[0]
    assert (game.away_team.team, game.home_team.team) == ("AWAY", "HOME")
    assert [p.player_id for p in game.away_team.participants] == ["qb-away", "rb1"]
    assert game.away_team.hidden_backup_count == 1


def test_props_match_stable_game_player_and_position_not_name():
    result = _result(
        [("g1", "A", "away", "qb-a", "Same Name"), ("g1", "H", "home", "qb-h", "Same Name")],
        [("g1", "A", "away", "rb-a", "Same Name", 1, False, False)],
        _odds([_line(player="qb-a"), _line(player="rb-a")]),
        _odds([_line(player="rb-a", market="player_rush_yds")]),
    )
    away = prepare_game_research(result, datetime.now(timezone.utc))[0].away_team.participants
    assert len(away[0].props) == 1
    assert len(away[1].props) == 1
    assert away[0].props[0].outcome == "Over"


def test_unresolved_blank_id_is_retained_and_no_line_diagnostic_is_safe():
    game = prepare_game_research(_result([("g1", "A", "away", None, "Unknown QB")], []), None)[0]
    assert game.away_team.participants[0].role == "unresolved participant"
    assert any(d.category == "unresolved_player" for d in game.diagnostics)


def test_deduplicates_exact_rows_but_rejects_conflicting_identity():
    row = ("g1", "A", "away", "qb-a", "QB")
    assert len(prepare_game_research(_result([row, row], []), None)[0].away_team.participants) == 1
    with pytest.raises(ValueError, match="Conflicting participant"):
        prepare_game_research(_result([row, ("g1", "A", "away", "qb-a", "Different")], []), None)


def test_deterministic_shuffled_input_and_diagnostics_cover_odds_states():
    qbs = [("g2", "B", "home", "q2", "Q2"), ("g2", "A", "away", "q1", "Q1"), ("g1", "D", "home", "q4", "Q4"), ("g1", "C", "away", "q3", "Q3")]
    odds = _odds([_line("g2", "q1", outcome="Over")], [("bad", "g2", "ambiguous")])
    first = prepare_game_research(_result(qbs, [], odds), None)
    second = prepare_game_research(_result(list(reversed(qbs)), [], odds), None)
    assert first == second
    assert [game.game_id for game in first] == ["g1", "g2"]
    assert any(d.category == "ambiguous_event" for d in first[1].diagnostics)
    assert any(d.category == "incomplete_over_under" for d in first[1].diagnostics)


def test_unmatched_player_and_missing_odds_diagnostics_are_visible():
    odds = _odds([_line(player=None, match_status="unmatched", player_name="Unknown")])
    game = prepare_game_research(_result([("g1", "A", "away", "q", "QB")], [], odds), None)[0]
    categories = {diagnostic.category for diagnostic in game.diagnostics}
    assert {"unmatched_player", "odds_unavailable"}.issubset(categories)
    assert game.away_team.participants[0].props == ()
    assert "Unknown" in next(d.message for d in game.diagnostics if d.category == "unmatched_player")


def test_no_selected_sportsbook_event_has_a_specific_empty_state_diagnostic():
    odds = _odds()
    odds.selected_events = pd.DataFrame()
    game = prepare_game_research(_result([("g1", "A", "away", "q", "QB")], [], odds), None)[0]
    assert any(d.category == "no_sportsbook_event" for d in game.diagnostics)


def test_prop_metadata_is_retained_including_retrieval_time():
    retrieved = datetime(2026, 9, 12, tzinfo=timezone.utc)
    game = prepare_game_research(_result([("g1", "A", "away", "q", "QB")], [], _odds([_line(player="q"), _line(player="q", outcome="Under")])), retrieved)[0]
    prop = game.away_team.participants[0].props[0]
    assert (prop.sportsbook, prop.line, prop.market_updated_at, prop.retrieved_at) == ("DraftKings", 250.5, "2026-09-12T00:00:00Z", retrieved)


def test_decision_snapshots_retain_complete_exact_rows_and_do_not_cross_positions():
    qb = _line(player="qb", outcome="Over")
    rb = _line(player="rb", market="player_rush_yds", outcome="Under", price=-125)
    result = _result([("g1", "A", "away", "qb", "Same")], [("g1", "A", "away", "rb", "Same", 1, False, False)], _odds([qb]), _odds([rb]))
    people = prepare_game_research(result, None)[0].away_team.participants
    qb_view, rb_view = people
    assert len(qb_view.decision_rows[0]) == len(WEEKLY_PLAYER_PROP_ODDS_COLUMNS)
    assert dict(zip(WEEKLY_PLAYER_PROP_ODDS_COLUMNS, qb_view.decision_rows[0]))["player_id"] == "qb"
    assert dict(zip(WEEKLY_PLAYER_PROP_ODDS_COLUMNS, rb_view.decision_rows[0]))["market_key"] == "player_rush_yds"
    assert qb_view.decision_rows != rb_view.decision_rows


def test_missing_results_and_input_frames_remain_unchanged():
    qb = _summary("QB", [("g1", "A", "away", "q", "QB")])
    before = qb.copy(deep=True)
    result = FootballResearchResult(WeeklyQBResearchResult(qb, pd.DataFrame(), pd.DataFrame(), None), None)
    games = prepare_game_research(result, None)
    assert games[0].away_team.participants[0].props == ()
    pd.testing.assert_frame_equal(qb, before)
    assert prepare_game_research(FootballResearchResult(None, None), None) == ()


def test_prepares_isolated_qb_and_rb_research_in_reverse_order_without_mutation():
    qb_summary = pd.DataFrame([{"game_id": "g1", "team": "A", "opponent": "H", "home_away": "away", "game_date": "2026-09-13", "game_time": "13:00", "expected_player_id": "qb-a", "expected_player_name": "Same"}])
    rb_summary = pd.DataFrame([{"game_id": "g1", "team": "A", "opponent": "H", "home_away": "away", "game_date": "2026-09-13", "game_time": "13:00", "player_id": "rb-a", "player_name": "Same", "participant_order": 1, "participant_resolution_missing": False, "limited_rb_sample": False}])
    qb_logs = pd.DataFrame([{"player_id": "qb-a", "season": 2025, "week": week, "game_id": f"q{week}", "opponent": "X", "passing_attempts": 30, "completions": 20, "passing_yards": 200, "passing_touchdowns": 2, "interceptions": 0} for week in range(1, 11)] + [{"player_id": "qb-other", "season": 2026, "week": 9, "game_id": "bad", "opponent": "X"}])
    qb_defense = pd.DataFrame([{"defense": "H", "season": 2025, "week": 2, "game_id": "d2", "offense_team": "A", "opposing_qb_name": "Q", "passing_attempts_allowed": 30, "completions_allowed": 20, "passing_yards_allowed": 200, "passing_touchdowns_allowed": 2, "opposing_interceptions": 1}, {"defense": "X", "season": 2026, "week": 9, "game_id": "bad"}])
    rb_logs = pd.DataFrame([{"player_id": "rb-a", "season": 2025, "week": 3, "game_id": "r3", "opponent": "X", "rushing_attempts": 10, "rushing_yards": 50, "rushing_touchdowns": 1, "targets": 2, "receptions": 1, "receiving_yards": 8, "opportunities": 12, "opportunity_share": .4}, {"player_id": "rb-other", "season": 2025, "week": 9, "game_id": "bad"}])
    rb_defense = pd.DataFrame([{"defense": "H", "season": 2025, "week": 3, "game_id": "rd3", "offense_team": "A", "rb_rushing_attempts_allowed": 20, "rb_rushing_yards_allowed": 100, "rb_rushing_touchdowns_allowed": 1}, {"defense": "X", "season": 2025, "week": 9, "game_id": "bad"}])
    originals = [frame.copy(deep=True) for frame in (qb_logs, qb_defense, rb_logs, rb_defense)]
    result = FootballResearchResult(WeeklyQBResearchResult(qb_summary, qb_logs, qb_defense), WeeklyRBResearchResult(rb_summary, rb_logs, rb_defense))
    people = prepare_game_research(result, None)[0].away_team.participants
    qb, rb = people
    assert len(qb.research.player_rows) == 8
    assert qb.research.player_rows[0][0] == 10
    assert qb.research.defense_rows and rb.research.defense_rows
    assert rb.research.player_rows[0][0:2] == (2025, 3)
    for actual, original in zip((qb_logs, qb_defense, rb_logs, rb_defense), originals): pd.testing.assert_frame_equal(actual, original)


class _Table:
    def __init__(self, frame): self.frame = frame
    def to_frame(self): return self.frame.copy(deep=True)


def test_selected_snapshot_balanced_filter_preserves_all_rows_and_exact_boundaries():
    rows = [
        _line(player="q", outcome="Over", price=-130, point=250.5),
        _line(player="q", outcome="Under", price=130, point=250.5),
        _line(player="q", outcome="Over", price=-110, point=260.5),
        _line(player="q", outcome="Under", price=-131, point=260.5),
        _line(player="q", outcome="Over", price=-110, point=270.5),
    ]
    frame = pd.DataFrame([{column: row.get(column, pd.NA) for column in WEEKLY_PLAYER_PROP_ODDS_COLUMNS} for row in rows], columns=WEEKLY_PLAYER_PROP_ODDS_COLUMNS)
    snapshot = type("Snapshot", (), {"player_matched_odds": _Table(frame), "event_matches": _Table(pd.DataFrame(columns=["event_id", "commence_time", "home_team", "away_team", "nflverse_game_id", "nflverse_season", "nflverse_week", "nflverse_home_team", "nflverse_away_team", "nflverse_kickoff_time", "event_match_status", "event_match_method", "event_match_candidate_count", "event_match_note"])), "retrieved_at": "now"})()
    result = _result([("g1", "A", "away", "q", "QB")], [])
    balanced = prepare_game_research(result, game_odds={"g1": snapshot}, line_filter="balanced")[0].away_team.participants[0]
    all_lines = prepare_game_research(result, game_odds={"g1": snapshot}, line_filter="all")[0].away_team.participants[0]
    assert [(prop.line, prop.outcome, prop.price) for prop in balanced.props] == [(250.5, "Over", -130), (250.5, "Under", 130)]
    assert len(all_lines.props) == 5
    pd.testing.assert_frame_equal(frame, snapshot.player_matched_odds.to_frame())
