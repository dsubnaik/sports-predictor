"""Pure hypothetical one-unit performance summaries for recorded decisions.

This module neither reads nor writes SQLite.  Its ``net_units`` value is a
hypothetical flat one-unit return based on saved odds for research decisions;
it is not evidence that a wager was placed.  Grouped reporting and display
formatting remain outside this focused calculation boundary.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from decimal import Context, Decimal, localcontext
from numbers import Integral

from football.decisions import StoredDecision


_SUPPORTED_MARKETS = {"QB": "player_pass_yds", "RB": "player_rush_yds"}
_SETTLED_STATUSES = frozenset({"win", "loss", "push"})
_VALID_STATUSES = _SETTLED_STATUSES | {"pending"}
_CALCULATION_CONTEXT = Context(prec=50)


class DecisionPerformanceError(ValueError):
    """Base exception for decision-performance input errors."""


class DecisionPerformanceValidationError(DecisionPerformanceError):
    """Raised when a supplied StoredDecision is internally inconsistent."""


class DecisionPerformanceConflictError(DecisionPerformanceError):
    """Raised when different decisions use the same decision identifier."""


@dataclass(frozen=True)
class DecisionPerformanceSummary:
    """Aggregate hypothetical one-unit results for validated decisions.

    ``hit_rate`` excludes pushes and pending decisions. ``net_units`` includes
    settled wins, losses, and pushes only, with no display rounding.
    """

    total_decisions: int
    pending_count: int
    win_count: int
    loss_count: int
    push_count: int
    settled_count: int
    graded_count: int
    hit_rate: Decimal | None
    net_units: Decimal


def summarize_decision_performance(
    decisions: Iterable[StoredDecision],
) -> DecisionPerformanceSummary:
    """Return a deterministic summary of validated recorded decisions.

    Computation uses a fixed 50-digit local Decimal context so recurring
    negative-odds returns and their accumulation do not depend on callers'
    ambient Decimal context. Individual returns are never quantized.
    """

    unique_decisions = validated_unique_decisions(decisions)

    pending_count = sum(decision.status == "pending" for decision in unique_decisions)
    win_count = sum(decision.status == "win" for decision in unique_decisions)
    loss_count = sum(decision.status == "loss" for decision in unique_decisions)
    push_count = sum(decision.status == "push" for decision in unique_decisions)
    graded_count = win_count + loss_count

    with localcontext(_CALCULATION_CONTEXT):
        net_units = sum((_unit_return(decision) for decision in unique_decisions), Decimal(0))
        hit_rate = Decimal(win_count) / Decimal(graded_count) if graded_count else None

    return DecisionPerformanceSummary(
        total_decisions=len(unique_decisions),
        pending_count=pending_count,
        win_count=win_count,
        loss_count=loss_count,
        push_count=push_count,
        settled_count=win_count + loss_count + push_count,
        graded_count=graded_count,
        hit_rate=hit_rate,
        net_units=net_units,
    )


def validated_unique_decisions(
    decisions: Iterable[StoredDecision],
) -> tuple[StoredDecision, ...]:
    """Return validated decisions once each, sorted by ``decision_id``.

    Identical repeated decision IDs collapse. Differing representations that
    share an ID raise ``DecisionPerformanceConflictError``. This public helper
    lets pure reports reuse exactly the same input boundary as the summary.
    """

    if isinstance(decisions, (str, bytes)):
        raise DecisionPerformanceValidationError(
            "decisions must be a non-string iterable of StoredDecision objects"
        )
    try:
        values = tuple(decisions)
    except TypeError as error:
        raise DecisionPerformanceValidationError(
            "decisions must be an iterable of StoredDecision objects"
        ) from error

    if not all(isinstance(decision, StoredDecision) for decision in values):
        raise DecisionPerformanceValidationError(
            "decisions must contain StoredDecision objects"
        )
    for decision in values:
        if not isinstance(decision.decision_id, str) or not decision.decision_id.strip():
            raise DecisionPerformanceValidationError("decision_id must be nonblank text")

    by_id: dict[str, StoredDecision] = {}
    for decision in sorted(values, key=lambda item: item.decision_id):
        existing = by_id.get(decision.decision_id)
        if existing is not None and existing != decision:
            raise DecisionPerformanceConflictError(
                f"conflicting decisions share decision_id: {decision.decision_id}"
            )
        by_id[decision.decision_id] = decision
    unique_decisions = tuple(by_id[decision_id] for decision_id in sorted(by_id))
    for decision in unique_decisions:
        _validate_decision(decision)
    return unique_decisions


def _validate_decision(decision: StoredDecision) -> None:
    if decision.status not in _VALID_STATUSES:
        raise DecisionPerformanceValidationError("decision status must be pending, win, loss, or push")
    if _SUPPORTED_MARKETS.get(decision.position) != decision.market_key:
        raise DecisionPerformanceValidationError(
            "decision position and market_key must be QB/player_pass_yds or RB/player_rush_yds"
        )
    if decision.selection not in {"over", "under"}:
        raise DecisionPerformanceValidationError("decision selection must be over or under")
    _validate_price(decision.selected_price)
    _validate_decimal(decision.line, "line")
    _validate_timestamp(decision.odds_retrieved_at, "odds_retrieved_at")
    _validate_timestamp(decision.recorded_at, "recorded_at")
    if decision.recorded_at < decision.odds_retrieved_at:
        raise DecisionPerformanceValidationError("recorded_at must not precede odds_retrieved_at")

    if decision.status == "pending":
        if decision.actual_result is not None or decision.settled_at is not None:
            raise DecisionPerformanceValidationError(
                "pending decisions must have null actual_result and settled_at"
            )
        return

    if decision.actual_result is None or decision.settled_at is None:
        raise DecisionPerformanceValidationError(
            "settled decisions must have actual_result and settled_at"
        )
    _validate_decimal(decision.actual_result, "actual_result")
    _validate_timestamp(decision.settled_at, "settled_at")
    if decision.settled_at < decision.recorded_at:
        raise DecisionPerformanceValidationError("settled_at must not precede recorded_at")
    if _settlement_status(decision.selection, decision.line, decision.actual_result) != decision.status:
        raise DecisionPerformanceValidationError("stored settlement status is inconsistent")


def _validate_price(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise DecisionPerformanceValidationError("selected_price must be an integer")
    if -99 <= int(value) <= 99:
        raise DecisionPerformanceValidationError("selected_price must be <= -100 or >= 100")


def _validate_decimal(value: object, field: str) -> None:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise DecisionPerformanceValidationError(f"{field} must be a finite Decimal")


def _validate_timestamp(value: object, field: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise DecisionPerformanceValidationError(f"{field} must be a timezone-aware datetime")


def _settlement_status(selection: str, line: Decimal, actual_result: Decimal) -> str:
    """Mirror the store's Decimal-only over/under settlement comparison."""

    if actual_result == line:
        return "push"
    if selection == "over":
        return "win" if actual_result > line else "loss"
    return "win" if actual_result < line else "loss"


def _unit_return(decision: StoredDecision) -> Decimal:
    if decision.status == "pending" or decision.status == "push":
        return Decimal(0)
    if decision.status == "loss":
        return Decimal(-1)
    if decision.selected_price > 0:
        return Decimal(int(decision.selected_price)) / Decimal(100)
    return Decimal(100) / Decimal(abs(int(decision.selected_price)))
