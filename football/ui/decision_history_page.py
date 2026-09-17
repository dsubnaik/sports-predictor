"""Read-only Streamlit controller for locally recorded football decisions."""

from __future__ import annotations

import sqlite3
from typing import Any, Callable
from datetime import datetime, timezone

import streamlit as st

from football.decision_database import (
    DecisionDatabaseValidationError,
    open_local_decision_database,
)
from football.decisions import (
    DecisionStoreError,
    SettlementConflictError,
    StoredDecision,
    get_decision,
    list_decisions,
    settle_decision,
)
from football.odds.scores import OddsScoresError
from football.results.decision_performance import DecisionPerformanceError
from football.results.decision_performance_breakdowns import (
    DecisionPerformanceBreakdownReport,
    build_decision_performance_breakdowns,
)
from football.results.completed_games import CompletedGameMappingError
from football.results.decision_result_matching import DecisionResultMatchError
from football.results.live_settlement_inputs import LiveSettlementInputsValidationError
from football.results.local_settlement_runner import (
    LocalSettlementRunReport,
    LocalSettlementRunnerValidationError,
    run_local_decision_settlement,
)
from football.ui.decision_history_view import (
    completed_game_diagnostic_rows,
    decision_performance_breakdown_rows,
    decision_performance_summary_rows,
    decision_history_filter_options,
    decision_history_rows,
    filter_decision_history,
    ManualSettlementInputError,
    manual_pending_decision_options,
    parse_manual_actual_result,
    pending_decision_seasons,
    settled_settlement_rows,
    unresolved_settlement_rows,
)


_HISTORY_STATE_KEY = "football_decision_history_decisions"
_SETTLEMENT_REPORT_STATE_KEY = "football_decision_history_settlement_report"
_HISTORY_STALE_STATE_KEY = "football_decision_history_may_be_stale"


def render_decision_history_page(
    *,
    streamlit_module: Any | None = None,
    database_opener: Callable[[], sqlite3.Connection] = open_local_decision_database,
    settlement_runner: Callable[[int], LocalSettlementRunReport] = run_local_decision_settlement,
    manual_settlement_clock: Callable[[], object] | None = None,
) -> None:
    """Render a deliberately loaded, read-only history of stored decisions."""

    ui = streamlit_module or st
    ui.title("Decision History")
    ui.info("Load recorded football decisions to review immutable odds snapshots and statuses.")

    if ui.button("Load Decision History", type="primary"):
        _load_history(ui, database_opener)

    decisions = ui.session_state.get(_HISTORY_STATE_KEY)
    if decisions is None:
        ui.caption("Load Decision History to view locally recorded decisions.")
        return
    if not decisions:
        ui.info("No decisions recorded yet.")
        return
    if not isinstance(decisions, tuple) or not all(
        isinstance(decision, StoredDecision) for decision in decisions
    ):
        raise TypeError("loaded decision history must be a tuple of StoredDecision values")

    _render_settlement_section(ui, decisions, database_opener, settlement_runner)
    decisions = ui.session_state[_HISTORY_STATE_KEY]
    _render_manual_historical_settlement(
        ui,
        decisions,
        database_opener,
        manual_settlement_clock or (lambda: datetime.now(timezone.utc)),
    )
    if ui.session_state.get(_HISTORY_STALE_STATE_KEY):
        ui.warning(
            "Displayed History and performance may be stale until Load Decision History succeeds."
        )

    # A successful deliberate settlement rereads local history once. Use that
    # refreshed immutable tuple for the current table and filtered performance.
    decisions = ui.session_state[_HISTORY_STATE_KEY]

    options = decision_history_filter_options(decisions)
    position, market_key, season, week, sportsbook, status = _render_filters(ui, options)
    filtered = filter_decision_history(
        decisions,
        position=position,
        market_key=market_key,
        season=season,
        week=week,
        sportsbook=sportsbook,
        status=status,
    )
    if not filtered:
        ui.info("No decisions match these filters.")
        return
    try:
        performance = build_decision_performance_breakdowns(filtered)
    except DecisionPerformanceError:
        ui.error("Could not calculate performance for the loaded decision history.")
        return
    _render_performance(ui, performance)
    ui.dataframe(decision_history_rows(filtered), use_container_width=True, hide_index=True)


def _load_history(
    ui: Any,
    database_opener: Callable[[], sqlite3.Connection],
    *,
    warn_if_stale: bool = False,
) -> bool:
    connection: sqlite3.Connection | None = None
    try:
        connection = database_opener()
        decisions = tuple(list_decisions(connection))
    except (OSError, sqlite3.Error, DecisionStoreError):
        ui.error("Could not load the local decision history.")
        if warn_if_stale:
            ui.session_state[_HISTORY_STALE_STATE_KEY] = True
        return False
    else:
        ui.session_state[_HISTORY_STATE_KEY] = decisions
        ui.session_state[_HISTORY_STALE_STATE_KEY] = False
        return True
    finally:
        if connection is not None:
            connection.close()


def _render_settlement_section(
    ui: Any,
    decisions: tuple[StoredDecision, ...],
    database_opener: Callable[[], sqlite3.Connection],
    settlement_runner: Callable[[int], LocalSettlementRunReport],
) -> None:
    """Render only an explicitly acknowledged, user-submitted settlement action."""

    ui.subheader("Settle Pending Decisions")
    seasons = pending_decision_seasons(decisions)
    if not seasons:
        ui.info("No loaded pending decisions are available to settle.")
        _render_prior_settlement_report(ui)
        return

    ui.warning(
        "This deliberate action may load live nflverse data and make one Odds API "
        "scores request, which may consume provider credits."
    )
    with ui.form("football_history_settlement_form"):
        selected_season = ui.selectbox(
            "Season to settle",
            seasons,
            key="football_history_settlement_season",
        )
        acknowledged = ui.checkbox(
            "I understand this may make a live scores request and consume Odds API credits.",
            key="football_history_settlement_acknowledged",
        )
        submitted = ui.form_submit_button("Settle Pending Decisions", type="primary")

    if submitted:
        if not acknowledged:
            ui.error("Acknowledge the live-data and Odds API credit notice before settling.")
        else:
            _run_settlement(ui, int(selected_season), database_opener, settlement_runner)
    _render_prior_settlement_report(ui)


def _run_settlement(
    ui: Any,
    season: int,
    database_opener: Callable[[], sqlite3.Connection],
    settlement_runner: Callable[[int], LocalSettlementRunReport],
) -> None:
    try:
        report = settlement_runner(season)
    except (
        LocalSettlementRunnerValidationError,
        LiveSettlementInputsValidationError,
        OddsScoresError,
        CompletedGameMappingError,
        DecisionResultMatchError,
        DecisionStoreError,
        DecisionDatabaseValidationError,
        OSError,
        sqlite3.Error,
    ):
        ui.error("Could not settle pending decisions. History was left unchanged.")
        return

    if report.outcome == "no_pending_decisions":
        ui.info(
            f"No pending decisions remain for {report.season}; no live inputs or Odds API scores request was made."
        )
        return

    workflow_report = report.workflow_report
    if workflow_report is None:
        raise RuntimeError("executed settlement runner report must include a workflow report")
    ui.session_state[_SETTLEMENT_REPORT_STATE_KEY] = report
    batch = workflow_report.batch_settlement_report
    ui.success(
        f"Settlement run completed: {batch.settled_count} settled or confirmed; "
        f"{batch.unresolved_count} unresolved."
    )
    _load_history(ui, database_opener, warn_if_stale=True)


def _render_prior_settlement_report(ui: Any) -> None:
    report = ui.session_state.get(_SETTLEMENT_REPORT_STATE_KEY)
    if not isinstance(report, LocalSettlementRunReport) or report.outcome != "executed":
        return
    workflow_report = report.workflow_report
    if workflow_report is None:
        return

    ui.caption("Most recent deliberate settlement run (not a new settlement).")
    mapping_rows = completed_game_diagnostic_rows(
        workflow_report.completed_game_report.diagnostics
    )
    if mapping_rows:
        ui.caption("Completed-event mapping diagnostics")
        ui.dataframe(mapping_rows, use_container_width=True, hide_index=True)
    settled_rows = settled_settlement_rows(
        workflow_report.batch_settlement_report.settled_entries
    )
    if settled_rows:
        ui.caption("Settled or confirmed decisions")
        ui.dataframe(settled_rows, use_container_width=True, hide_index=True)
    unresolved_rows = unresolved_settlement_rows(
        workflow_report.batch_settlement_report.unresolved_entries
    )
    if unresolved_rows:
        ui.caption("Unresolved decision diagnostics")
        ui.info(
            "Unresolved decisions remain pending. A missing player result is never assumed to be zero."
        )
        ui.dataframe(unresolved_rows, use_container_width=True, hide_index=True)


def _render_manual_historical_settlement(
    ui: Any,
    decisions: tuple[StoredDecision, ...],
    database_opener: Callable[[], sqlite3.Connection],
    clock: Callable[[], object],
) -> None:
    """Render a deliberate no-network recovery path for older pending decisions."""

    ui.subheader("Manual Historical Settlement")
    options = manual_pending_decision_options(decisions)
    if not options:
        ui.info("No loaded pending decisions are available for manual historical settlement.")
        return

    option_by_id = {option.decision_id: option for option in options}
    ui.caption(
        "Recovery tool for officially verified results not handled through the recent live-results window. "
        "This action does not use the Odds API and consumes no API credits."
    )
    with ui.form("football_history_manual_settlement_form"):
        decision_id = ui.selectbox(
            "Pending decision to settle manually",
            list(option_by_id),
            format_func=lambda value: option_by_id[value].label,
            key="football_history_manual_settlement_decision_id",
        )
        actual_result_text = ui.text_input(
            "Official actual result",
            key="football_history_manual_settlement_actual_result",
        )
        acknowledged = ui.checkbox(
            "I verified this official player result. This manual action is intended for decisions "
            "that could not be settled through the recent live-results window.",
            key="football_history_manual_settlement_acknowledged",
        )
        submitted = ui.form_submit_button("Settle Manually Verified Result", type="primary")

    if not submitted:
        return
    if not acknowledged:
        ui.error("Acknowledge that you verified the official player result before settling.")
        return
    if decision_id not in option_by_id:
        ui.error("The selected pending decision is no longer available for manual settlement.")
        return
    try:
        actual_result = parse_manual_actual_result(actual_result_text)
    except ManualSettlementInputError as error:
        ui.error(f"Could not settle manually: {error}")
        return

    _run_manual_historical_settlement(
        ui,
        decision_id,
        actual_result,
        clock(),
        database_opener,
    )


def _run_manual_historical_settlement(
    ui: Any,
    decision_id: str,
    actual_result: object,
    settled_at: object,
    database_opener: Callable[[], sqlite3.Connection],
) -> None:
    """Reread one ID and delegate mutation, idempotency, and conflicts to the store."""

    connection: sqlite3.Connection | None = None
    try:
        connection = database_opener()
        current = get_decision(connection, decision_id)
        if current is None:
            ui.error("The selected decision no longer exists in the local decision database.")
            return
        stored = settle_decision(
            connection,
            decision_id,
            actual_result,
            settled_at=settled_at,
        )
    except SettlementConflictError:
        ui.error("Manual settlement conflicts with the stored result; nothing was overwritten.")
        return
    except DecisionStoreError:
        ui.error("Could not settle manually because the stored decision or timestamp was invalid.")
        return
    except (DecisionDatabaseValidationError, OSError, sqlite3.Error):
        ui.error("Could not access the local decision database for manual settlement.")
        return
    finally:
        if connection is not None:
            connection.close()

    ui.success(
        f"Manual settlement {stored.decision_id}: {stored.player_name} official result "
        f"{format(stored.actual_result, 'f')} — {stored.status}. No Odds API request was made."
    )
    _load_history(ui, database_opener, warn_if_stale=True)


def _render_filters(ui: Any, options: dict[str, tuple[object, ...]]) -> tuple[object, ...]:
    columns = ui.columns(6)
    with columns[0]:
        position = ui.selectbox("Position", options["position"], key="football_history_position")
    with columns[1]:
        market_key = ui.selectbox("Market", options["market_key"], key="football_history_market")
    with columns[2]:
        season = ui.selectbox("Season", options["season"], key="football_history_season")
    with columns[3]:
        week = ui.selectbox("Week", options["week"], key="football_history_week")
    with columns[4]:
        sportsbook = ui.selectbox("Sportsbook", options["sportsbook"], key="football_history_sportsbook")
    with columns[5]:
        status = ui.selectbox("Result", options["status"], key="football_history_status")
    return position, market_key, season, week, sportsbook, status


def _render_performance(ui: Any, report: DecisionPerformanceBreakdownReport) -> None:
    """Render display-only performance for exactly the current filtered tuple."""

    ui.subheader("Performance — Current filters")
    ui.info(
        "Hypothetical flat-stake units apply to recorded research decisions, not verified placed wagers."
    )
    ui.caption("Pending decisions have no realized unit return. Hit rate excludes pending decisions and pushes.")
    ui.dataframe(
        decision_performance_summary_rows(report.overall),
        use_container_width=True,
        hide_index=True,
    )
    for label, groups in (
        ("By Position", report.position_groups),
        ("By Market", report.market_key_groups),
        ("By Season / Week", report.season_week_groups),
        ("By Sportsbook", report.sportsbook_groups),
    ):
        if groups:
            ui.caption(label)
            ui.dataframe(
                decision_performance_breakdown_rows(groups),
                use_container_width=True,
                hide_index=True,
            )
