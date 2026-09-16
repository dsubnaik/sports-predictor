"""Pure filtering and display preparation for read-only decision history."""

from __future__ import annotations

from datetime import datetime
from decimal import Context, Decimal, localcontext

from football.decisions import StoredDecision
from football.results.batch_settlement import BatchSettlementEntry
from football.results.completed_games import CompletedGameDiagnostic
from football.results.decision_performance import DecisionPerformanceSummary
from football.results.decision_performance_breakdowns import (
    DecisionPerformanceGroup,
    SeasonWeekPerformanceKey,
)


ALL_FILTER = "All"
STATUS_FILTERS = (ALL_FILTER, "pending", "win", "loss", "push")


def pending_decision_seasons(decisions: tuple[StoredDecision, ...]) -> tuple[int, ...]:
    """Return sorted seasons with at least one loaded pending decision."""

    return tuple(sorted({decision.season for decision in decisions if decision.status == "pending"}))


def completed_game_diagnostic_rows(
    diagnostics: tuple[CompletedGameDiagnostic, ...],
) -> list[dict[str, str]]:
    """Return mapping diagnostics without interpreting provider finality."""

    return [
        {
            "Provider Event ID": diagnostic.provider_event_id,
            "Match Status": diagnostic.match_status,
            "Diagnostic": diagnostic.diagnostic,
        }
        for diagnostic in diagnostics
    ]


def unresolved_settlement_rows(
    entries: tuple[BatchSettlementEntry, ...],
) -> list[dict[str, object]]:
    """Return unresolved decision diagnostics in existing batch-report order."""

    return [
        {
            "Decision ID": entry.decision_id,
            "Position": entry.position,
            "Market": entry.market_key,
            "Game ID": entry.game_id,
            "Player ID": entry.player_id,
            "Match Status": entry.match_status,
            "Diagnostic": entry.diagnostic or "",
        }
        for entry in entries
    ]


def settled_settlement_rows(
    entries: tuple[BatchSettlementEntry, ...],
) -> list[dict[str, object]]:
    """Return settled-or-confirmed entries in existing batch-report order."""

    return [
        {
            "Decision ID": entry.decision_id,
            "Position": entry.position,
            "Market": entry.market_key,
            "Game ID": entry.game_id,
            "Player ID": entry.player_id,
            "Actual Result": format(entry.actual_result, "f"),
            "Decision Status": entry.decision_status,
        }
        for entry in entries
    ]


def decision_history_filter_options(
    decisions: tuple[StoredDecision, ...],
) -> dict[str, tuple[object, ...]]:
    """Return deterministic filter values derived from already loaded decisions."""

    return {
        "position": (ALL_FILTER, *sorted({decision.position for decision in decisions})),
        "market_key": (ALL_FILTER, *sorted({decision.market_key for decision in decisions})),
        "season": (ALL_FILTER, *sorted({decision.season for decision in decisions})),
        "week": (ALL_FILTER, *sorted({decision.week for decision in decisions})),
        "sportsbook": (ALL_FILTER, *sorted({decision.sportsbook for decision in decisions})),
        "status": STATUS_FILTERS,
    }


def filter_decision_history(
    decisions: tuple[StoredDecision, ...],
    *,
    position: object = ALL_FILTER,
    market_key: object = ALL_FILTER,
    season: object = ALL_FILTER,
    week: object = ALL_FILTER,
    sportsbook: object = ALL_FILTER,
    status: object = ALL_FILTER,
) -> tuple[StoredDecision, ...]:
    """Filter immutable loaded decisions without changing their store order."""

    filters = {
        "position": position,
        "market_key": market_key,
        "season": season,
        "week": week,
        "sportsbook": sportsbook,
        "status": status,
    }
    return tuple(
        decision
        for decision in decisions
        if all(
            value == ALL_FILTER or getattr(decision, field) == value
            for field, value in filters.items()
        )
    )


def decision_history_rows(decisions: tuple[StoredDecision, ...]) -> list[dict[str, object]]:
    """Return readable history rows while preserving stored snapshot values."""

    return [
        {
            "Decision ID": decision.decision_id,
            "Recorded At": _timestamp(decision.recorded_at),
            "Odds Retrieved At": _timestamp(decision.odds_retrieved_at),
            "Season": decision.season,
            "Week": decision.week,
            "Game ID": decision.game_id,
            "Position": decision.position,
            "Player": decision.player_name,
            "Player ID": decision.player_id,
            "Team": decision.team,
            "Opponent": decision.opponent,
            "Market": decision.market_key,
            "Sportsbook": decision.sportsbook,
            "Line": format(decision.line, "f"),
            "Selection": decision.selection,
            "Selected Price": decision.selected_price,
            "Research Notes": decision.research_notes or "",
            "Status": decision.status,
            "Actual Result": (
                format(decision.actual_result, "f")
                if decision.actual_result is not None
                else "Unsettled"
            ),
            "Settled At": (
                _timestamp(decision.settled_at)
                if decision.settled_at is not None
                else "Unsettled"
            ),
        }
        for decision in decisions
    ]


def decision_performance_summary_rows(
    summary: DecisionPerformanceSummary,
) -> list[dict[str, object]]:
    """Return one display-only performance row for the current filters."""

    return [{
        "Scope": "Current filters",
        "Decisions": summary.total_decisions,
        "Pending": summary.pending_count,
        "Wins": summary.win_count,
        "Losses": summary.loss_count,
        "Pushes": summary.push_count,
        "Settled": summary.settled_count,
        "Graded": summary.graded_count,
        "Hit Rate": _hit_rate(summary.hit_rate),
        "Hypothetical Flat-Stake Units": _units(summary.net_units),
    }]


def decision_performance_breakdown_rows(
    groups: tuple[DecisionPerformanceGroup, ...],
) -> list[dict[str, object]]:
    """Return display-only rows while retaining the report's group order."""

    return [{
        "Group": _group_key(group.key),
        "Decisions": group.summary.total_decisions,
        "Pending": group.summary.pending_count,
        "Wins": group.summary.win_count,
        "Losses": group.summary.loss_count,
        "Pushes": group.summary.push_count,
        "Hit Rate": _hit_rate(group.summary.hit_rate),
        "Hypothetical Flat-Stake Units": _units(group.summary.net_units),
    } for group in groups]


def _timestamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _hit_rate(value: Decimal | None) -> str:
    if value is None:
        return "N/A (no wins or losses)"
    with localcontext(Context(prec=50)):
        return f"{format(value * Decimal(100), '.1f')}%"


def _units(value: Decimal) -> str:
    with localcontext(Context(prec=50)):
        return format(value, ".2f")


def _group_key(value: str | SeasonWeekPerformanceKey) -> str:
    if isinstance(value, SeasonWeekPerformanceKey):
        return f"{value.season} Week {value.week}"
    return value
