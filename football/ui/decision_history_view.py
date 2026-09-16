"""Pure filtering and display preparation for read-only decision history."""

from __future__ import annotations

from datetime import datetime

from football.decisions import StoredDecision


ALL_FILTER = "All"
STATUS_FILTERS = (ALL_FILTER, "pending", "win", "loss", "push")


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


def _timestamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")
