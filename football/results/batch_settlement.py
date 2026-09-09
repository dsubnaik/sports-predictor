"""Atomically coordinate existing decision matching and settlement components.

Matching completes before writes begin. Only matcher-confirmed results are sent
to ``settle_decision``; unresolved diagnostics are preserved unchanged. Each
batch uses one shared settlement timestamp and rolls back all of its writes if
any settlement fails. Sequential reruns leave already settled decisions alone.
External result loading remains outside this module.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd

from football.decisions import StoredDecision, list_decisions, settle_decision
from football.results.decision_result_matching import (
    DecisionResultMatch,
    DecisionResultMatchReport,
    match_decision_results,
)


@dataclass(frozen=True)
class BatchSettlementEntry:
    """One matcher result annotated with its batch-settlement outcome."""

    decision_id: str
    position: str
    market_key: str
    season: int
    week: int
    game_id: str
    player_id: str
    batch_status: str
    match_status: str
    actual_result: Decimal | None
    decision_status: str
    diagnostic: str | None


@dataclass(frozen=True)
class BatchSettlementReport:
    """Deterministically ordered entries from one batch-settlement attempt."""

    entries: tuple[BatchSettlementEntry, ...]

    @property
    def settled_entries(self) -> tuple[BatchSettlementEntry, ...]:
        return tuple(entry for entry in self.entries if entry.batch_status == "settled")

    @property
    def unresolved_entries(self) -> tuple[BatchSettlementEntry, ...]:
        return tuple(entry for entry in self.entries if entry.batch_status == "unresolved")

    @property
    def settled_count(self) -> int:
        return len(self.settled_entries)

    @property
    def unresolved_count(self) -> int:
        return len(self.unresolved_entries)


def settle_decision_batch(
    connection: sqlite3.Connection,
    quarterback_games: pd.DataFrame,
    running_back_games: pd.DataFrame,
    completed_game_ids: object,
    *,
    settled_at: object | None = None,
    clock: Callable[[], object] | None = None,
) -> BatchSettlementReport:
    """Match and atomically settle every currently pending completed decision."""

    decisions = list_decisions(connection)
    match_report = match_decision_results(
        decisions,
        quarterback_games,
        running_back_games,
        completed_game_ids,
    )
    matched = match_report.matched_results
    if not matched:
        return _build_report(match_report, {})

    timestamp = settled_at if settled_at is not None else (clock or _utc_now)()
    savepoint = _savepoint_name()
    settled: dict[str, StoredDecision] = {}
    connection.execute(f"SAVEPOINT {savepoint}")
    try:
        for match in matched:
            settled[match.decision_id] = settle_decision(
                connection,
                match.decision_id,
                match.actual_result,
                settled_at=timestamp,
            )
    except Exception:
        connection.execute(f"ROLLBACK TO {savepoint}")
        connection.execute(f"RELEASE {savepoint}")
        raise
    connection.execute(f"RELEASE {savepoint}")
    return _build_report(match_report, settled)


def _build_report(
    match_report: DecisionResultMatchReport,
    settled: dict[str, StoredDecision],
) -> BatchSettlementReport:
    entries: list[BatchSettlementEntry] = []
    for match in match_report.matches:
        decision = settled.get(match.decision_id)
        if decision is None:
            entries.append(_unresolved_entry(match))
        else:
            entries.append(
                BatchSettlementEntry(
                    decision_id=match.decision_id,
                    position=match.position,
                    market_key=match.market_key,
                    season=match.season,
                    week=match.week,
                    game_id=match.game_id,
                    player_id=match.player_id,
                    batch_status="settled",
                    match_status="matched",
                    actual_result=match.actual_result,
                    decision_status=decision.status,
                    diagnostic=None,
                )
            )
    return BatchSettlementReport(tuple(entries))


def _unresolved_entry(match: DecisionResultMatch) -> BatchSettlementEntry:
    return BatchSettlementEntry(
        decision_id=match.decision_id,
        position=match.position,
        market_key=match.market_key,
        season=match.season,
        week=match.week,
        game_id=match.game_id,
        player_id=match.player_id,
        batch_status="unresolved",
        match_status=match.match_status,
        actual_result=None,
        decision_status="pending",
        diagnostic=match.diagnostic,
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _savepoint_name() -> str:
    """Return a generated SQL-safe identifier independent of caller input."""

    return f"batch_settlement_{uuid.uuid4().hex}"
