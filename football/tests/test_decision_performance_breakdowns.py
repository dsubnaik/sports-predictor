"""Unit tests for pure deterministic decision-performance breakdowns."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from football.decisions import StoredDecision
from football.results.decision_performance import (
    DecisionPerformanceConflictError,
    DecisionPerformanceValidationError,
)
from football.results.decision_performance_breakdowns import (
    NET_UNITS_RECONCILIATION_RELATIVE_TOLERANCE,
    SeasonWeekPerformanceKey,
    build_decision_performance_breakdowns,
)


ODDS_TIME = datetime(2026, 9, 10, 17, tzinfo=timezone.utc)
RECORDED_TIME = ODDS_TIME + timedelta(minutes=5)
SETTLED_TIME = RECORDED_TIME + timedelta(hours=2)


def decision(
    decision_id,
    *,
    position="QB",
    season=2026,
    week=1,
    sportsbook="draftkings",
    status="pending",
    selection="over",
    line=Decimal("250.5"),
    actual_result=None,
    selected_price=-110,
    settled_at=None,
):
    return StoredDecision(
        decision_id=decision_id,
        position=position,
        player_id=f"player-{decision_id}",
        player_name="Player",
        team="KC",
        opponent="LAC",
        season=season,
        week=week,
        game_id=f"{season}_{week:02d}_KC_LAC_{decision_id}",
        market_key="player_pass_yds" if position == "QB" else "player_rush_yds",
        sportsbook=sportsbook,
        line=line,
        selection=selection,
        selected_price=selected_price,
        recorded_at=RECORDED_TIME,
        odds_retrieved_at=ODDS_TIME,
        research_notes=None,
        status=status,
        actual_result=actual_result,
        settled_at=settled_at,
    )


def settled(decision_id, *, status="win", selection="over", line=Decimal("250.5"), actual_result=None, **changes):
    if actual_result is None:
        actual_result = {
            ("over", "win"): line + Decimal(1),
            ("over", "loss"): line - Decimal(1),
            ("under", "win"): line - Decimal(1),
            ("under", "loss"): line + Decimal(1),
        }.get((selection, status), line)
    return decision(
        decision_id,
        status=status,
        selection=selection,
        line=line,
        actual_result=actual_result,
        settled_at=SETTLED_TIME,
        **changes,
    )


def _group_map(groups):
    return {group.key: group.summary for group in groups}


def _assert_exact_count_reconciliation(report, groups):
    for field in (
        "total_decisions", "pending_count", "win_count", "loss_count",
        "push_count", "settled_count", "graded_count",
    ):
        assert sum(getattr(group.summary, field) for group in groups) == getattr(report.overall, field)


def _assert_net_units_reconcile(report, groups):
    group_total = sum((group.summary.net_units for group in groups), Decimal(0))
    tolerance = NET_UNITS_RECONCILIATION_RELATIVE_TOLERANCE * max(
        Decimal(1), abs(report.overall.net_units)
    )
    assert abs(group_total - report.overall.net_units) <= tolerance


def test_empty_input_returns_empty_groups_and_existing_empty_overall_summary():
    report = build_decision_performance_breakdowns([])
    assert report.overall.total_decisions == 0
    assert report.overall.hit_rate is None and report.overall.net_units == Decimal(0)
    assert report.position_groups == report.market_key_groups == ()
    assert report.season_week_groups == report.sportsbook_groups == ()


def test_dimension_keys_and_summaries_are_separate_and_explicitly_ordered():
    values = [
        settled("rb", position="RB", season=2026, week=1, sportsbook="fanduel", selection="under", selected_price=100),
        settled("qb-old", season=2025, week=1, sportsbook="betmgm", selected_price=150),
        settled("qb-new", season=2026, week=1, sportsbook="draftkings", status="loss"),
    ]
    report = build_decision_performance_breakdowns(reversed(values))
    assert [group.key for group in report.position_groups] == ["QB", "RB"]
    assert [group.key for group in report.market_key_groups] == ["player_pass_yds", "player_rush_yds"]
    assert [group.key for group in report.season_week_groups] == [
        SeasonWeekPerformanceKey(2025, 1), SeasonWeekPerformanceKey(2026, 1)
    ]
    assert [group.key for group in report.sportsbook_groups] == ["betmgm", "draftkings", "fanduel"]
    assert _group_map(report.position_groups)["QB"].net_units == Decimal("0.5")
    assert _group_map(report.position_groups)["RB"].net_units == Decimal(1)


def test_counts_hit_rates_and_saved_price_returns_reconcile_in_every_dimension():
    values = [
        settled("qb-win", selected_price=150),
        settled("qb-loss", status="loss", selected_price=-200),
        settled("rb-push", position="RB", status="push", actual_result=Decimal("250.5"), sportsbook="fanduel"),
        decision("rb-pending", position="RB", sportsbook="fanduel"),
    ]
    report = build_decision_performance_breakdowns(values)
    assert report.overall.net_units == Decimal("0.5")
    assert _group_map(report.position_groups)["QB"].hit_rate == Decimal("0.5")
    assert _group_map(report.position_groups)["RB"].hit_rate is None
    for groups in (
        report.position_groups,
        report.market_key_groups,
        report.season_week_groups,
        report.sportsbook_groups,
    ):
        _assert_exact_count_reconciliation(report, groups)
        _assert_net_units_reconcile(report, groups)
        assert all(isinstance(group.summary.net_units, Decimal) for group in groups)


def test_duplicate_behavior_generator_input_and_order_are_deterministic():
    first = settled("a", selected_price=-150)
    second = settled("b", position="RB", selection="under", selected_price=100)
    values = [second, first, first]
    consumed = []

    def source():
        for value in values:
            consumed.append(value.decision_id)
            yield value

    generated = build_decision_performance_breakdowns(source())
    reordered = build_decision_performance_breakdowns(reversed(values))
    assert consumed == ["b", "a", "a"]
    assert generated == reordered
    assert generated.overall.total_decisions == 2


def test_conflicting_duplicate_id_raises_before_any_report_is_returned():
    original = settled("same")
    with pytest.raises(DecisionPerformanceConflictError, match="conflicting decisions share decision_id: same"):
        build_decision_performance_breakdowns([original, replace(original, selected_price=150)])


def test_invalid_decision_is_rejected_before_any_groups_are_built():
    with pytest.raises(DecisionPerformanceValidationError, match="settlement status"):
        build_decision_performance_breakdowns([
            settled("invalid", status="win", actual_result=Decimal("249"))
        ])


def test_input_decisions_remain_unchanged():
    values = [settled("qb"), decision("rb", position="RB")]
    before = tuple(values)
    build_decision_performance_breakdowns(values)
    assert tuple(values) == before
