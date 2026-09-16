"""Read-only Streamlit controller for locally recorded football decisions."""

from __future__ import annotations

import sqlite3
from typing import Any, Callable

import streamlit as st

from football.decision_database import open_local_decision_database
from football.decisions import DecisionStoreError, StoredDecision, list_decisions
from football.ui.decision_history_view import (
    decision_history_filter_options,
    decision_history_rows,
    filter_decision_history,
)


_HISTORY_STATE_KEY = "football_decision_history_decisions"


def render_decision_history_page(
    *,
    streamlit_module: Any | None = None,
    database_opener: Callable[[], sqlite3.Connection] = open_local_decision_database,
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
    ui.dataframe(decision_history_rows(filtered), use_container_width=True, hide_index=True)


def _load_history(
    ui: Any,
    database_opener: Callable[[], sqlite3.Connection],
) -> None:
    connection: sqlite3.Connection | None = None
    try:
        connection = database_opener()
        decisions = tuple(list_decisions(connection))
    except (OSError, sqlite3.Error, DecisionStoreError):
        ui.error("Could not load the local decision history.")
    else:
        ui.session_state[_HISTORY_STATE_KEY] = decisions
    finally:
        if connection is not None:
            connection.close()


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
