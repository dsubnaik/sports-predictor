"""Deliberately run local decision settlement for one explicitly requested season.

This runner performs only preflight and composition. It opens one caller-owned
local connection for the invocation, avoids live inputs when no target-season
pending decision exists, and delegates all matching and mutation to the
existing workflow.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Literal

from football.decision_database import open_local_decision_database
from football.decisions import list_decisions
from football.results.live_settlement_inputs import (
    LiveSettlementInputs,
    load_live_settlement_inputs,
)
from football.results.settlement_workflow import (
    DecisionSettlementWorkflowReport,
    run_decision_settlement_workflow,
)


class LocalSettlementRunnerValidationError(ValueError):
    """Raised when a runner request cannot be safely started."""


@dataclass(frozen=True)
class LocalSettlementRunReport:
    """The no-work or executed outcome of one deliberate runner invocation."""

    season: int
    outcome: Literal["no_pending_decisions", "executed"]
    workflow_report: DecisionSettlementWorkflowReport | None


def run_local_decision_settlement(
    season: int,
    *,
    database_path: Path | str | None = None,
    database_opener: Callable[[Path | str | None], sqlite3.Connection] | None = None,
    inputs_loader: Callable[..., LiveSettlementInputs] | None = None,
    workflow_runner: Callable[..., DecisionSettlementWorkflowReport] | None = None,
    days_from: int = 3,
    settled_at: object | None = None,
    clock: Callable[[], object] | None = None,
) -> LocalSettlementRunReport:
    """Run settlement only when the requested season has pending decisions.

    Validation happens before SQLite opens. The connection is closed on every
    post-open path; live inputs and the workflow run at most once each.
    """

    requested_season = _season(season)
    requested_days_from = _days_from(days_from)
    opener = database_opener or open_local_decision_database
    connection: sqlite3.Connection | None = None
    try:
        connection = opener(database_path)
        decisions = list_decisions(connection)
        if not any(
            decision.status == "pending" and decision.season == requested_season
            for decision in decisions
        ):
            return LocalSettlementRunReport(
                season=requested_season,
                outcome="no_pending_decisions",
                workflow_report=None,
            )

        inputs = (inputs_loader or load_live_settlement_inputs)(
            requested_season,
            days_from=requested_days_from,
        )
        workflow_report = (workflow_runner or run_decision_settlement_workflow)(
            connection,
            inputs.scores_payload,
            inputs.normalized_schedule,
            inputs.quarterback_games,
            inputs.running_back_games,
            settled_at=settled_at,
            clock=clock,
        )
        return LocalSettlementRunReport(
            season=requested_season,
            outcome="executed",
            workflow_report=workflow_report,
        )
    finally:
        if connection is not None:
            connection.close()


def _season(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise LocalSettlementRunnerValidationError("season must be a positive integer")
    return int(value)


def _days_from(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value not in {1, 2, 3}:
        raise LocalSettlementRunnerValidationError("days_from must be an integer from 1 through 3")
    return int(value)
