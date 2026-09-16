"""Unit tests for pure hypothetical decision-performance summaries."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Context, Decimal, getcontext, localcontext

import pytest

from football.decisions import StoredDecision
from football.results.decision_performance import (
    DecisionPerformanceConflictError,
    DecisionPerformanceValidationError,
    summarize_decision_performance,
    validated_unique_decisions,
)


ODDS_TIME = datetime(2026, 9, 10, 17, tzinfo=timezone.utc)
RECORDED_TIME = ODDS_TIME + timedelta(minutes=5)
SETTLED_TIME = RECORDED_TIME + timedelta(hours=2)


def decision(
    decision_id="decision",
    *,
    status="pending",
    selection="over",
    line=Decimal("250.5"),
    actual_result=None,
    selected_price=-110,
    settled_at=None,
    position="QB",
):
    return StoredDecision(
        decision_id=decision_id,
        position=position,
        player_id=f"player-{decision_id}",
        player_name="Player",
        team="KC",
        opponent="LAC",
        season=2026,
        week=1,
        game_id=f"2026_01_KC_LAC_{decision_id}",
        market_key="player_pass_yds" if position == "QB" else "player_rush_yds",
        sportsbook="draftkings",
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


def settled(decision_id, *, status="win", selection="over", line=Decimal("250.5"), actual_result=None, selected_price=-110, position="QB"):
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
        selected_price=selected_price,
        settled_at=SETTLED_TIME,
        position=position,
    )


def test_empty_and_pending_only_inputs_have_no_realized_performance():
    empty = summarize_decision_performance([])
    assert empty.total_decisions == empty.settled_count == empty.graded_count == 0
    assert empty.hit_rate is None and empty.net_units == Decimal(0)

    summary = summarize_decision_performance([decision("pending")])
    assert (summary.total_decisions, summary.pending_count, summary.settled_count) == (1, 1, 0)
    assert summary.hit_rate is None and summary.net_units == Decimal(0)


@pytest.mark.parametrize(
    "price, expected",
    [(100, Decimal("1")), (150, Decimal("1.5")), (-100, Decimal("1"))],
)
def test_positive_and_even_negative_price_wins_return_expected_units(price, expected):
    summary = summarize_decision_performance([settled("win", selected_price=price)])
    assert summary.win_count == 1
    assert summary.net_units == expected


def test_negative_price_win_uses_fixed_decimal_precision_independent_of_ambient_context():
    winner = settled("negative", selected_price=-150)
    original = getcontext().prec
    try:
        getcontext().prec = 6
        first = summarize_decision_performance([winner])
        getcontext().prec = 3
        second = summarize_decision_performance([winner])
    finally:
        getcontext().prec = original
    with localcontext(Context(prec=50)):
        expected = Decimal(100) / Decimal(150)
    assert first.net_units == second.net_units
    assert first.net_units == expected


def test_losses_pushes_and_pending_have_defined_unit_and_hit_rate_treatment():
    summary = summarize_decision_performance([
        settled("win", selected_price=150),
        settled("loss", status="loss", selected_price=-200),
        settled("push", status="push", actual_result=Decimal("250.5")),
        decision("pending"),
    ])
    assert (summary.pending_count, summary.win_count, summary.loss_count, summary.push_count) == (1, 1, 1, 1)
    assert (summary.settled_count, summary.graded_count, summary.hit_rate) == (3, 2, Decimal("0.5"))
    assert summary.net_units == Decimal("0.5")


def test_pushes_do_not_change_hit_rate_and_qb_rb_use_the_same_return_rules():
    summary = summarize_decision_performance([
        settled("qb", selected_price=100),
        settled("rb", position="RB", selection="under", selected_price=-100),
        settled("push", status="push", actual_result=Decimal("250.5")),
    ])
    assert summary.hit_rate == Decimal(1)
    assert summary.net_units == Decimal(2)


def test_selected_price_is_authoritative_and_distinct_ids_count_separately():
    first = settled("first", selected_price=100)
    second = settled("second", selected_price=150)
    summary = summarize_decision_performance([first, second])
    assert summary.total_decisions == 2
    assert summary.net_units == Decimal("2.5")


def test_identical_duplicate_ids_collapse_and_conflicting_ids_raise_independent_of_order():
    first = settled("same", selected_price=100)
    assert summarize_decision_performance([first, first]).total_decisions == 1
    assert validated_unique_decisions([first, first]) == (first,)
    conflicting = replace(first, selected_price=150)
    for values in ([first, conflicting], [conflicting, first]):
        with pytest.raises(DecisionPerformanceConflictError, match="conflicting decisions share decision_id: same"):
            summarize_decision_performance(values)


def test_generator_is_consumed_once_and_input_order_does_not_change_summary():
    values = [settled("b", selected_price=-150), settled("a", status="loss"), decision("c")]
    consumed = []

    def source():
        for value in values:
            consumed.append(value.decision_id)
            yield value

    generated = summarize_decision_performance(source())
    reordered = summarize_decision_performance(reversed(values))
    assert consumed == ["b", "a", "c"]
    assert generated == reordered


@pytest.mark.parametrize("price", [0, 99, -99, True, Decimal("100")])
def test_invalid_selected_prices_are_rejected(price):
    with pytest.raises(DecisionPerformanceValidationError, match="selected_price"):
        summarize_decision_performance([settled("invalid-price", selected_price=price)])


@pytest.mark.parametrize(
    "changed, message",
    [
        ({"status": "unknown"}, "status"),
        ({"status": "pending", "actual_result": Decimal("251"), "settled_at": SETTLED_TIME}, "pending decisions"),
        ({"status": "win", "actual_result": None, "settled_at": SETTLED_TIME}, "settled decisions"),
        ({"status": "win", "actual_result": Decimal("249"), "settled_at": SETTLED_TIME}, "settlement status"),
        ({"recorded_at": ODDS_TIME - timedelta(seconds=1)}, "recorded_at must not precede"),
        ({"settled_at": RECORDED_TIME - timedelta(seconds=1)}, "settled_at must not precede"),
    ],
)
def test_inconsistent_states_and_timestamps_are_rejected(changed, message):
    value = settled("invalid")
    if changed.get("status") == "pending":
        value = replace(decision("invalid"), **changed)
    else:
        value = replace(value, **changed)
    with pytest.raises(DecisionPerformanceValidationError, match=message):
        summarize_decision_performance([value])


@pytest.mark.parametrize(
    "actual_result",
    [Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity"), True],
)
def test_nonfinite_actual_results_are_rejected(actual_result):
    with pytest.raises(DecisionPerformanceValidationError, match="actual_result"):
        summarize_decision_performance([settled("nonfinite", actual_result=actual_result)])


def test_input_decisions_remain_unchanged_and_outputs_are_decimal():
    value = settled("immutable", selected_price=-150)
    before = value
    summary = summarize_decision_performance([value])
    assert value == before
    assert isinstance(summary.hit_rate, Decimal)
    assert isinstance(summary.net_units, Decimal)
