"""Unit tests for pure pending-decision result matching."""

from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd
import pytest

from football.decisions import StoredDecision
from football.results.decision_result_matching import (
    ConflictingResultDataError,
    DecisionResultValidationError,
    match_decision_results,
)


TIME = datetime(2026, 9, 10, tzinfo=timezone.utc)
QB_COLUMNS = ["season", "week", "game_id", "player_id", "passing_yards"]
RB_COLUMNS = ["season", "week", "game_id", "player_id", "rushing_yards"]


def decision(**changes):
    values = dict(
        decision_id="qb-decision", position="QB", player_id="qb-1", player_name="Same Name",
        team="KC", opponent="LAC", season=2026, week=1, game_id="2026_01_KC_LAC",
        market_key="player_pass_yds", sportsbook="draftkings", line=Decimal("250.5"),
        selection="over", selected_price=-110, recorded_at=TIME, odds_retrieved_at=TIME,
        research_notes="note", status="pending", actual_result=None, settled_at=None,
    )
    values.update(changes)
    return StoredDecision(**values)


def qb_games(rows=None):
    if rows is None:
        rows = [{"season": 2026, "week": 1, "game_id": "2026_01_KC_LAC", "player_id": "qb-1", "passing_yards": 275}]
    return pd.DataFrame(rows, columns=QB_COLUMNS)


def rb_games(rows=None):
    if rows is None:
        rows = [{"season": 2026, "week": 1, "game_id": "2026_01_KC_LAC", "player_id": "rb-1", "rushing_yards": 75}]
    return pd.DataFrame(rows, columns=RB_COLUMNS)


def report(decisions, qbs=None, rbs=None, completed=("2026_01_KC_LAC",)):
    return match_decision_results(decisions, qbs if qbs is not None else qb_games(), rbs if rbs is not None else rb_games(), completed)


def test_completed_qb_and_rb_decisions_use_only_their_correct_result_columns():
    qb = decision()
    rb = decision(decision_id="rb-decision", position="RB", player_id="rb-1", market_key="player_rush_yds")
    matches = report([qb, rb]).matched_results
    assert [(item.decision_id, item.actual_result) for item in matches] == [
        ("qb-decision", Decimal("275")), ("rb-decision", Decimal("75"))
    ]


def test_exact_identity_forbids_name_team_or_opponent_fallbacks():
    source = qb_games([
        {"season": 2026, "week": 2, "game_id": "2026_01_KC_LAC", "player_id": "qb-1", "passing_yards": 275},
        {"season": 2026, "week": 1, "game_id": "2026_01_KC_LAC", "player_id": "other-id", "passing_yards": 275},
    ])
    item = report([decision()], qbs=source).matches[0]
    assert item.match_status == "player_result_missing"
    assert item.actual_result is None


@pytest.mark.parametrize("change", [
    {"season": 2025}, {"week": 2}, {"game_id": "2026_01_OTHER"}, {"player_id": "other-id"},
])
def test_each_exact_identity_component_is_required(change):
    source = {"season": 2026, "week": 1, "game_id": "2026_01_KC_LAC", "player_id": "qb-1", "passing_yards": 275}
    source.update(change)
    item = report([decision()], qbs=qb_games([source]), completed=("2026_01_KC_LAC", source["game_id"])).matches[0]
    assert item.match_status == "player_result_missing"


def test_player_name_team_and_opponent_never_affect_identifier_match():
    source = qb_games()
    renamed = decision(player_name="Different", team="BUF", opponent="MIA")
    assert report([renamed], qbs=source).matches[0].actual_result == Decimal("275")


def test_pending_and_missing_player_diagnostics_never_expose_results():
    pending = report([decision()], completed=()).matches[0]
    missing = report([decision()], qbs=qb_games([])).matches[0]
    assert (pending.match_status, pending.actual_result) == ("game_pending", None)
    assert (missing.match_status, missing.actual_result) == ("player_result_missing", None)
    assert pending.diagnostic == "Game is not in completed_game_ids."
    assert missing.diagnostic == "No normalized player result exists for the completed game."


def test_pending_game_ignores_unrelated_malformed_source_rows():
    malformed = qb_games([{"season": True, "week": 1, "game_id": "2026_01_KC_LAC", "player_id": "qb-1", "passing_yards": float("nan")}])
    item = report([decision()], qbs=malformed, completed=()).matches[0]
    assert (item.match_status, item.actual_result) == ("game_pending", None)


def test_malformed_identifier_for_requested_completed_player_is_rejected():
    malformed = qb_games([{"season": True, "week": 1, "game_id": "2026_01_KC_LAC", "player_id": "qb-1", "passing_yards": 275}])
    with pytest.raises(DecisionResultValidationError, match="quarterback season"):
        report([decision()], qbs=malformed)


@pytest.mark.parametrize("result", [0, -3, Decimal("275.125")])
def test_zero_negative_and_decimal_results_are_matched_without_rounding(result):
    item = report([decision()], qbs=qb_games([{"season": 2026, "week": 1, "game_id": "2026_01_KC_LAC", "player_id": "qb-1", "passing_yards": result}])).matches[0]
    assert item.match_status == "matched"
    expected = Decimal(str(result)).normalize() if result != 0 else Decimal(0)
    assert item.actual_result == expected


@pytest.mark.parametrize("result", [float("nan"), float("inf"), float("-inf"), True])
def test_invalid_relevant_results_are_rejected(result):
    with pytest.raises(DecisionResultValidationError, match="passing_yards"):
        report([decision()], qbs=qb_games([{"season": 2026, "week": 1, "game_id": "2026_01_KC_LAC", "player_id": "qb-1", "passing_yards": result}]))


@pytest.mark.parametrize("frame_factory, missing", [(qb_games, "passing_yards"), (rb_games, "rushing_yards")])
def test_missing_required_source_columns_are_rejected(frame_factory, missing):
    frame = frame_factory().drop(columns=[missing])
    with pytest.raises(DecisionResultValidationError, match=missing):
        report([decision()], qbs=frame if missing == "passing_yards" else qb_games(), rbs=frame if missing == "rushing_yards" else rb_games())


def test_valid_empty_sources_are_safe_and_settled_decisions_are_excluded():
    settled = replace(
        decision(), decision_id="settled", status="win", actual_result=Decimal("275"), settled_at=TIME
    )
    result = report([decision(), settled], qbs=qb_games([]), rbs=rb_games([]))
    assert len(result.matches) == 1
    assert result.matches[0].match_status == "player_result_missing"


def test_duplicate_source_rows_collapse_but_conflicts_raise_deterministically():
    row = {"season": 2026, "week": 1, "game_id": "2026_01_KC_LAC", "player_id": "qb-1", "passing_yards": Decimal("275.0")}
    duplicate = qb_games([row, {**row, "passing_yards": 275}])
    assert report([decision()], qbs=duplicate).matches[0].actual_result == Decimal("275")
    conflict = qb_games([row, {**row, "passing_yards": 276}])
    with pytest.raises(ConflictingResultDataError, match="Conflicting quarterback"):
        report([decision()], qbs=conflict)


def test_conflicting_running_back_results_are_rejected():
    rb = decision(decision_id="rb", position="RB", player_id="rb-1", market_key="player_rush_yds")
    conflict = rb_games([
        {"season": 2026, "week": 1, "game_id": "2026_01_KC_LAC", "player_id": "rb-1", "rushing_yards": 75},
        {"season": 2026, "week": 1, "game_id": "2026_01_KC_LAC", "player_id": "rb-1", "rushing_yards": 76},
    ])
    with pytest.raises(ConflictingResultDataError, match="Conflicting running-back"):
        report([rb], rbs=conflict)


def test_unrelated_source_conflicts_do_not_block_requested_match():
    qbs = qb_games([
        {"season": 2026, "week": 1, "game_id": "2026_01_KC_LAC", "player_id": "qb-1", "passing_yards": 275},
        {"season": 2026, "week": 1, "game_id": "2026_01_KC_LAC", "player_id": "other", "passing_yards": 1},
        {"season": 2026, "week": 1, "game_id": "2026_01_KC_LAC", "player_id": "other", "passing_yards": 2},
    ])
    assert report([decision()], qbs=qbs).matches[0].actual_result == Decimal("275")


def test_duplicate_decisions_ordering_and_reordered_inputs_are_deterministic():
    first = decision(decision_id="z", player_id="z-player", game_id="2026_02_Z", week=2)
    second = decision(decision_id="a", player_id="a-player", game_id="2026_01_A", week=1)
    qbs = qb_games([
        {"season": 2026, "week": 2, "game_id": "2026_02_Z", "player_id": "z-player", "passing_yards": 20},
        {"season": 2026, "week": 1, "game_id": "2026_01_A", "player_id": "a-player", "passing_yards": 10},
    ])
    forward = report([first, second, second], qbs=qbs, completed=("2026_02_Z", "2026_01_A"))
    reverse = report([second, first], qbs=qbs.iloc[::-1], completed=("2026_01_A", "2026_02_Z"))
    assert forward == reverse
    assert [item.decision_id for item in forward.matches] == ["a", "z"]
    assert forward.matched_results == forward.matches


def test_conflicting_duplicate_decisions_and_unsupported_position_market_are_rejected():
    with pytest.raises(DecisionResultValidationError, match="conflicting decisions"):
        report([decision(), replace(decision(), player_name="Different")])
    with pytest.raises(DecisionResultValidationError, match="unsupported position"):
        report([decision(position="QB", market_key="player_rush_yds")])


def test_completed_game_ids_are_materialized_validated_and_not_mutated():
    completed = [" 2026_01_KC_LAC ", "2026_01_KC_LAC"]
    assert report([decision()], completed=(value for value in completed)).matches[0].match_status == "matched"
    assert completed == [" 2026_01_KC_LAC ", "2026_01_KC_LAC"]
    with pytest.raises(DecisionResultValidationError, match="completed_game_ids"):
        report([decision()], completed=[" "])


def test_inputs_are_not_mutated():
    qbs = qb_games()
    rbs = rb_games()
    before_qbs = qbs.copy(deep=True)
    before_rbs = rbs.copy(deep=True)
    original = decision()
    report([original], qbs=qbs, rbs=rbs)
    pd.testing.assert_frame_equal(qbs, before_qbs)
    pd.testing.assert_frame_equal(rbs, before_rbs)
    assert original == decision()
