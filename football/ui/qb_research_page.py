"""Streamlit page for weekly football quarterback research."""

from __future__ import annotations

from datetime import date, datetime, timezone
import sqlite3
from typing import Any, Callable
from urllib.error import URLError

import streamlit as st

try:
    from requests.exceptions import RequestException
except ImportError:  # pragma: no cover - requests is an app dependency.
    RequestException = OSError

from football.pipeline import WeeklyQBResearchResult, build_weekly_qb_research
from football.decision_database import open_local_decision_database
from football.decisions import (
    DecisionStoreError,
    DecisionValidationError,
    DuplicateDecisionError,
    record_decision,
)
from football.ui.qb_research_view import (
    DEFENSIVE_MATCHUP_RANK_HELP,
    build_matchup_options,
    build_qb_decision_outcome_options,
    build_qb_pending_decision,
    build_matchup_warnings,
    build_selected_qb_prop_warnings,
    default_history_season,
    display_value,
    filter_defense_game_log,
    filter_qb_game_log,
    filter_selected_qb_passing_props,
    find_matchup,
    format_odds_retrieval_time,
    history_label,
    prepare_defense_log_display,
    prepare_passing_prop_display,
    prepare_qb_log_display,
    prepare_summary_display,
    QBDecisionEntryValidationError,
)


def load_qb_research(
    report_season: int,
    report_week: int,
    as_of_date: date,
    history_season: int,
) -> WeeklyQBResearchResult:
    """Build a fresh QB report with current passing-yard odds."""

    return build_weekly_qb_research(
        report_season=report_season,
        report_week=report_week,
        as_of_date=as_of_date,
        history_season=history_season,
        include_player_props=True,
    )


def render_football_qb_research_page(
    *,
    streamlit_module: Any | None = None,
    report_loader: Callable[..., WeeklyQBResearchResult] = load_qb_research,
    database_opener: Callable[[], sqlite3.Connection] = open_local_decision_database,
    clock: Callable[[], object] | None = None,
    id_factory: Callable[[], object] | None = None,
) -> None:
    """Render QB research and deliberately record one displayed odds snapshot."""

    ui = streamlit_module or st
    selected_clock = clock or (lambda: datetime.now(timezone.utc))
    ui.title("Football QB Research")
    ui.info("Choose your settings and generate a report.")

    today = date.today()
    with ui.form("football_report_form"):
        season_col, week_col, date_col, history_col, button_col = ui.columns(
            [1, 1, 1.3, 1, 1.2]
        )
        with season_col:
            report_season = ui.number_input(
                "Report season",
                min_value=1999,
                max_value=today.year + 1,
                value=today.year,
                step=1,
            )
        with week_col:
            report_week = ui.number_input(
                "Report week",
                min_value=1,
                max_value=22,
                value=1,
                step=1,
            )
        automatic_history_season = default_history_season(
            int(report_season),
            int(report_week),
        )
        with date_col:
            as_of_date = ui.date_input("As-of date", value=today)
        with history_col:
            history_season = ui.number_input(
                "History season",
                min_value=min(1999, automatic_history_season),
                max_value=int(report_season),
                value=automatic_history_season,
                step=1,
                help=(
                    "Defaults to previous season for Week 1 and current season "
                    "for later weeks."
                ),
            )
        with button_col:
            ui.write("")
            generate_report = ui.form_submit_button(
                "Generate Football Report",
                type="primary",
            )

    if generate_report:
        report_inputs = {
            "report_season": int(report_season),
            "report_week": int(report_week),
            "as_of_date": as_of_date,
            "history_season": int(history_season),
        }
        try:
            with ui.spinner("Loading research and sportsbook data..."):
                result = report_loader(**report_inputs)
                retrieval_time = datetime.now(timezone.utc)
                ui.session_state["football_qb_research_inputs"] = report_inputs
                ui.session_state["football_qb_research_result"] = result
                ui.session_state["football_qb_research_odds_retrieved_at"] = retrieval_time
        except ValueError as error:
            ui.error(f"Could not build the report from the available data: {error}")
            ui.caption("Research data: nflverse via nflreadpy. Sportsbook data: The Odds API.")
            ui.stop()
        except (ImportError, ModuleNotFoundError) as error:
            ui.error(f"Could not load the nflverse dependency: {error}")
            ui.caption("Research data: nflverse via nflreadpy. Sportsbook data: The Odds API.")
            ui.stop()
        except (OSError, TimeoutError, URLError, RequestException) as error:
            ui.error(
                "Could not download report or sportsbook data. Check the network and try again. "
                f"{error}"
            )
            ui.caption("Research data: nflverse via nflreadpy. Sportsbook data: The Odds API.")
            ui.stop()

    report_inputs = ui.session_state.get("football_qb_research_inputs")
    result = ui.session_state.get("football_qb_research_result")

    if report_inputs is None or result is None:
        ui.caption("No football data will load until you generate the report.")
        ui.caption("Research data: nflverse via nflreadpy. Sportsbook data: The Odds API.")
        ui.stop()

    ui.caption(
        f"Report season {report_inputs['report_season']}, "
        f"Week {report_inputs['report_week']}. "
        f"Historical season used: {report_inputs['history_season']}."
    )

    summary_display = prepare_summary_display(result.summary)
    ui.subheader("Ranked Matchup Summary")
    ui.caption(DEFENSIVE_MATCHUP_RANK_HELP)
    if summary_display.empty:
        ui.info("No scheduled team matchups were found for these report inputs.")
        ui.caption("Research data: nflverse via nflreadpy. Sportsbook data: The Odds API.")
        ui.stop()

    ui.dataframe(summary_display, use_container_width=True, hide_index=True)

    options = build_matchup_options(result.summary)
    option_by_label = {option.label: option.option_id for option in options}
    selected_label = ui.selectbox("Matchup", list(option_by_label))
    selected_matchup = find_matchup(result.summary, option_by_label[selected_label])

    if selected_matchup is None:
        ui.error("The selected matchup could not be found in the report result.")
        ui.caption("Research data: nflverse via nflreadpy. Sportsbook data: The Odds API.")
        ui.stop()

    ui.subheader("Selected Matchup")
    left, right = ui.columns(2)
    with left:
        ui.markdown(
            f"**{display_value(selected_matchup.get('team'))} vs "
            f"{display_value(selected_matchup.get('opponent'))}**  \n"
            f"Expected QB: "
            f"**{display_value(selected_matchup.get('expected_player_name'), 'Unresolved')}**"
        )
        ui.caption(history_label(selected_matchup))
    with right:
        ui.markdown(
            f"Expected-QB source: "
            f"**{display_value(selected_matchup.get('selection_source'))}**  \n"
            f"Depth-chart date: "
            f"**{display_value(selected_matchup.get('depth_chart_date'))}**"
        )
        notes = selected_matchup.get("selection_notes")
        notes_text = display_value(notes, fallback="")
        if notes_text:
            ui.caption(notes_text)

    for warning in build_matchup_warnings(selected_matchup):
        ui.warning(warning)

    ui.subheader("Live Passing-Yard Lines")
    player_prop_odds = result.player_prop_odds
    selected_props = None
    if player_prop_odds is not None:
        selected_props = filter_selected_qb_passing_props(
            player_prop_odds.player_matched_odds,
            selected_matchup,
        )
    for warning in build_selected_qb_prop_warnings(
        player_prop_odds,
        selected_matchup,
        selected_props,
    ):
        ui.warning(warning)
    if selected_props is None or selected_props.empty:
        ui.info("No matched passing-yard sportsbook line is available for this matchup.")
    else:
        ui.dataframe(
            prepare_passing_prop_display(selected_props),
            use_container_width=True,
            hide_index=True,
        )
    retrieval_time = ui.session_state.get("football_qb_research_odds_retrieved_at")
    if player_prop_odds is not None and retrieval_time is not None:
        ui.caption(
            f"Odds retrieved: {format_odds_retrieval_time(retrieval_time)}. "
            "Lines are the values returned when this report was generated; "
            "sportsbook market-update times are shown in the table."
        )

    qb_log = filter_qb_game_log(result.qb_game_logs, selected_matchup)
    defense_log = filter_defense_game_log(result.defense_game_logs, selected_matchup)

    _render_qb_decision_entry(
        ui,
        selected_props,
        selected_matchup,
        retrieval_time,
        database_opener=database_opener,
        clock=selected_clock,
        id_factory=id_factory,
    )

    ui.subheader("Selected QB Historical Game Log")
    if qb_log.empty:
        ui.info("No historical game log is available for the selected expected QB.")
    else:
        ui.dataframe(
            prepare_qb_log_display(qb_log),
            use_container_width=True,
            hide_index=True,
        )

    ui.subheader("Opponent Defense vs QB Game Log")
    if defense_log.empty:
        ui.info("No defense-versus-QB game log is available for the selected opponent.")
    else:
        ui.dataframe(
            prepare_defense_log_display(defense_log),
            use_container_width=True,
            hide_index=True,
        )

    ui.caption("Research data: nflverse via nflreadpy. Sportsbook data: The Odds API.")


def _render_qb_decision_entry(
    ui: Any,
    selected_props: Any,
    selected_matchup: Any,
    retrieval_time: object | None,
    *,
    database_opener: Callable[[], sqlite3.Connection],
    clock: Callable[[], object],
    id_factory: Callable[[], object] | None,
) -> None:
    """Render the separate deliberate action for one currently displayed outcome."""

    if selected_props is None or selected_props.empty:
        return
    try:
        options = build_qb_decision_outcome_options(selected_props)
    except QBDecisionEntryValidationError as error:
        ui.error(f"Passing-yard decision entry is unavailable: {error}")
        return
    if not options:
        return

    option_by_id = {option.option_id: option for option in options}
    ui.subheader("Record QB Passing-Yards Decision")
    with ui.form("football_qb_decision_entry_form"):
        selected_option_id = ui.selectbox(
            "Exact displayed outcome",
            list(option_by_id),
            format_func=lambda value: option_by_id[value].label,
            key="football_qb_decision_selected_outcome",
        )
        research_notes = ui.text_area(
            "Research notes (optional)",
            key="football_qb_decision_research_notes",
        )
        submitted = ui.form_submit_button("Record Decision", type="primary")

    if not submitted:
        return

    try:
        decision = build_qb_pending_decision(
            selected_props,
            selected_matchup,
            selected_option_id,
            retrieval_time,
            clock(),
            research_notes,
        )
    except QBDecisionEntryValidationError as error:
        ui.error(f"Could not record the displayed outcome: {error}")
        return

    connection: sqlite3.Connection | None = None
    try:
        connection = database_opener()
        if id_factory is None:
            stored = record_decision(connection, decision)
        else:
            stored = record_decision(connection, decision, id_factory=id_factory)
    except DuplicateDecisionError:
        ui.info("This exact sportsbook snapshot was already recorded.")
    except DecisionValidationError as error:
        ui.error(f"Could not record the displayed outcome: {error}")
    except DecisionStoreError:
        ui.error("Could not record the displayed outcome because the decision store rejected it.")
    except (OSError, sqlite3.Error):
        ui.error("Could not open or write the local decision database.")
    else:
        ui.success(
            f"Recorded {stored.decision_id}: {stored.sportsbook} {stored.line} "
            f"{stored.selection} {stored.selected_price}."
        )
    finally:
        if connection is not None:
            connection.close()
