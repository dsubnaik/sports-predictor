"""Deliberately invoked composition of completed-event mapping and settlement.

Callers own score retrieval, nflverse data loading, and the SQLite connection.
This module maps only explicitly completed provider events before delegating all
decision mutation and transaction handling to the existing batch settler.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from football.results.batch_settlement import (
    BatchSettlementReport,
    settle_decision_batch,
)
from football.results.completed_games import (
    CompletedGameReport,
    identify_completed_nflverse_games,
)


@dataclass(frozen=True)
class DecisionSettlementWorkflowReport:
    """Reports retained from one completed-event mapping and settlement attempt."""

    completed_game_report: CompletedGameReport
    batch_settlement_report: BatchSettlementReport


def run_decision_settlement_workflow(
    connection: sqlite3.Connection,
    scores_payload: Sequence[Mapping[str, Any]],
    normalized_schedule: pd.DataFrame,
    quarterback_games: pd.DataFrame,
    running_back_games: pd.DataFrame,
    *,
    settled_at: object | None = None,
    clock: Callable[[], object] | None = None,
) -> DecisionSettlementWorkflowReport:
    """Map explicitly completed events once, then invoke batch settlement once."""

    completed_game_report = identify_completed_nflverse_games(
        scores_payload,
        normalized_schedule,
    )
    batch_settlement_report = settle_decision_batch(
        connection,
        quarterback_games,
        running_back_games,
        completed_game_report.completed_game_ids,
        settled_at=settled_at,
        clock=clock,
    )
    return DecisionSettlementWorkflowReport(
        completed_game_report=completed_game_report,
        batch_settlement_report=batch_settlement_report,
    )
