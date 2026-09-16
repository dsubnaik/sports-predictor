"""Pure deterministic performance breakdowns for recorded decisions.

The report reuses the existing hypothetical one-unit performance summary; it
does not access SQLite or claim that a recorded research decision was wagered.
Counts reconcile exactly to the overall summary. Independently accumulated
``net_units`` values can differ in a final Decimal place for recurring
negative-odds fractions, so callers may use the documented relative tolerance
when reconciling group totals to the overall total.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from decimal import Decimal

from football.decisions import StoredDecision
from football.results.decision_performance import (
    DecisionPerformanceSummary,
    summarize_decision_performance,
    validated_unique_decisions,
)


NET_UNITS_RECONCILIATION_RELATIVE_TOLERANCE = Decimal("1E-40")
"""Tight relative Decimal tolerance for independently accumulated group sums.

Compare ``abs(group_total - overall)`` to this tolerance multiplied by
``max(Decimal(1), abs(overall))``. The 50-digit summary context is retained;
no per-decision return is rounded to satisfy reconciliation.
"""


@dataclass(frozen=True)
class SeasonWeekPerformanceKey:
    """The inseparable season/week key for a weekly performance group."""

    season: int
    week: int


@dataclass(frozen=True)
class DecisionPerformanceGroup:
    """One exact grouping key paired with its existing performance summary."""

    key: str | SeasonWeekPerformanceKey
    summary: DecisionPerformanceSummary


@dataclass(frozen=True)
class DecisionPerformanceBreakdownReport:
    """Overall and explicitly ordered dimension-level performance summaries."""

    overall: DecisionPerformanceSummary
    position_groups: tuple[DecisionPerformanceGroup, ...]
    market_key_groups: tuple[DecisionPerformanceGroup, ...]
    season_week_groups: tuple[DecisionPerformanceGroup, ...]
    sportsbook_groups: tuple[DecisionPerformanceGroup, ...]


def build_decision_performance_breakdowns(
    decisions: Iterable[StoredDecision],
) -> DecisionPerformanceBreakdownReport:
    """Build pure overall and single-dimension performance breakdowns.

    The original iterable is prepared exactly once. Position, market key, and
    canonical sportsbook keys sort lexicographically; season/week pairs sort
    by ascending season then week.
    """

    unique_decisions = validated_unique_decisions(decisions)
    return DecisionPerformanceBreakdownReport(
        overall=summarize_decision_performance(unique_decisions),
        position_groups=_string_groups(unique_decisions, lambda decision: decision.position),
        market_key_groups=_string_groups(unique_decisions, lambda decision: decision.market_key),
        season_week_groups=_season_week_groups(unique_decisions),
        sportsbook_groups=_string_groups(unique_decisions, lambda decision: decision.sportsbook),
    )


def _string_groups(
    decisions: tuple[StoredDecision, ...],
    key_for: Callable[[StoredDecision], str],
) -> tuple[DecisionPerformanceGroup, ...]:
    grouped: dict[str, list[StoredDecision]] = {}
    for decision in decisions:
        grouped.setdefault(key_for(decision), []).append(decision)
    return tuple(
        DecisionPerformanceGroup(key=key, summary=summarize_decision_performance(grouped[key]))
        for key in sorted(grouped)
    )


def _season_week_groups(
    decisions: tuple[StoredDecision, ...],
) -> tuple[DecisionPerformanceGroup, ...]:
    grouped: dict[SeasonWeekPerformanceKey, list[StoredDecision]] = {}
    for decision in decisions:
        key = SeasonWeekPerformanceKey(season=decision.season, week=decision.week)
        grouped.setdefault(key, []).append(decision)
    return tuple(
        DecisionPerformanceGroup(key=key, summary=summarize_decision_performance(grouped[key]))
        for key in sorted(grouped, key=lambda item: (item.season, item.week))
    )
