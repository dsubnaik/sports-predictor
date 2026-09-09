"""Streamlit page for weekly football running-back research."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Callable
from urllib.error import URLError

import streamlit as st
import pandas as pd

try:
    from requests.exceptions import RequestException
except ImportError:  # pragma: no cover - requests is an app dependency.
    RequestException = OSError

from football.pipeline import WeeklyRBResearchResult, build_weekly_rb_research
from football.ui.rb_research_view import (
    DEFENSIVE_MATCHUP_RANK_HELP,
    build_participant_options,
    build_selected_rb_prop_warnings,
    build_warning_counts,
    default_history_season,
    display_value,
    filter_defense_game_log,
    filter_rb_game_log,
    filter_selected_rb_rushing_props,
    find_participant,
    format_odds_retrieval_time,
    prepare_defense_log_display,
    prepare_rushing_prop_display,
    prepare_rb_log_display,
    prepare_summary_display,
)


def load_rb_research(
    report_season: int,
    report_week: int,
    as_of_date: date,
    history_season: int,
    include_player_props: bool = True,
) -> WeeklyRBResearchResult:
    """Build a fresh RB report with current rushing-yard odds."""

    return build_weekly_rb_research(
        report_season=report_season,
        report_week=report_week,
        as_of_date=as_of_date,
        history_season=history_season,
        include_player_props=include_player_props,
    )


def render_football_rb_research_page(
    *,
    streamlit_module: Any | None = None,
    report_loader: Callable[..., WeeklyRBResearchResult] = load_rb_research,
) -> None:
    """Render the RB research page; optional dependencies support isolated UI tests."""

    ui = streamlit_module or st
    ui.title("Football RB Research")
    ui.info("Generate research context for expected backfield participants, not predictions.")

    today = date.today()
    with ui.form("football_rb_report_form"):
        season_col, week_col, date_col, history_col, button_col = ui.columns([1, 1, 1.3, 1, 1.3])
        with season_col:
            report_season = ui.number_input("Report season", min_value=1999, max_value=today.year + 1, value=today.year, step=1)
        with week_col:
            report_week = ui.number_input("Report week", min_value=1, max_value=22, value=1, step=1)
        automatic_history = default_history_season(int(report_season), int(report_week))
        with date_col:
            as_of_date = ui.date_input("As-of date", value=today)
        with history_col:
            history_season = ui.number_input(
                "History season",
                min_value=min(1999, automatic_history),
                max_value=int(report_season),
                value=automatic_history,
                step=1,
                help="Defaults to the previous season for Week 1 and the current season for later weeks.",
            )
        with button_col:
            ui.write("")
            generate_report = ui.form_submit_button("Generate Football RB Report", type="primary")

    if generate_report:
        inputs = {
            "report_season": int(report_season),
            "report_week": int(report_week),
            "as_of_date": as_of_date,
            "history_season": int(history_season),
        }
        try:
            with ui.spinner("Loading research and sportsbook data..."):
                result = report_loader(**inputs, include_player_props=True)
            retrieval_time = datetime.now(timezone.utc)
            ui.session_state["football_rb_research_inputs"] = inputs
            ui.session_state["football_rb_research_result"] = result
            ui.session_state["football_rb_research_odds_retrieved_at"] = retrieval_time
            ui.session_state.pop("football_rb_research_selected_participant", None)
        except ValueError as error:
            ui.error(f"Could not build the report from the available data: {error}")
            ui.caption("Research data: nflverse via nflreadpy. Sportsbook data: The Odds API.")
            ui.stop()
        except (ImportError, ModuleNotFoundError) as error:
            ui.error(f"Could not load the nflverse dependency: {error}")
            ui.caption("Research data: nflverse via nflreadpy. Sportsbook data: The Odds API.")
            ui.stop()
        except (OSError, TimeoutError, URLError, RequestException) as error:
            ui.error(f"Could not download report or sportsbook data. Check the network and try again. {error}")
            ui.caption("Research data: nflverse via nflreadpy. Sportsbook data: The Odds API.")
            ui.stop()

    inputs = ui.session_state.get("football_rb_research_inputs")
    result = ui.session_state.get("football_rb_research_result")
    if inputs is None or result is None:
        ui.caption("No football data will load until you generate the report.")
        ui.caption("Research data: nflverse via nflreadpy. Sportsbook data: The Odds API.")
        ui.stop()

    ui.caption(f"Report season {inputs['report_season']}, Week {inputs['report_week']}. Historical season used: {inputs['history_season']}.")
    _render_warning_summary(ui, result.summary)
    ui.subheader("Ranked Backfield Matchup Summary")
    ui.caption(DEFENSIVE_MATCHUP_RANK_HELP)
    if result.summary.empty:
        ui.info("No scheduled team matchups were found for these report inputs.")
        ui.caption("Research data: nflverse via nflreadpy. Sportsbook data: The Odds API.")
        ui.stop()
    ui.dataframe(prepare_summary_display(result.summary), use_container_width=True, hide_index=True)

    options = build_participant_options(result.summary)
    if not options:
        ui.info("No resolved expected RB participants are available to select. Unresolved rows remain in the summary above.")
        ui.caption("Research data: nflverse via nflreadpy. Sportsbook data: The Odds API.")
        ui.stop()
    option_by_label = {option.label: option.option_id for option in options}
    selected_label = ui.selectbox("Expected backfield participant", list(option_by_label), key="football_rb_research_selected_participant")
    participant = find_participant(result.summary, option_by_label[selected_label])
    if participant is None:
        ui.error("The selected participant could not be found in the report result.")
        ui.stop()

    _render_selected_context(ui, participant)

    ui.subheader("Live Rushing-Yard Lines")
    player_prop_odds = result.player_prop_odds
    selected_props = None
    if player_prop_odds is not None:
        selected_props = filter_selected_rb_rushing_props(
            player_prop_odds.player_matched_odds,
            participant,
        )
    for warning in build_selected_rb_prop_warnings(
        player_prop_odds,
        participant,
        selected_props,
    ):
        ui.warning(warning)
    if selected_props is None or selected_props.empty:
        ui.info("No matched rushing-yard sportsbook line is available for this participant.")
    else:
        ui.dataframe(
            prepare_rushing_prop_display(selected_props),
            use_container_width=True,
            hide_index=True,
        )
    retrieval_time = ui.session_state.get("football_rb_research_odds_retrieved_at")
    if player_prop_odds is not None and retrieval_time is not None:
        ui.caption(
            f"Odds retrieved: {format_odds_retrieval_time(retrieval_time)}. "
            "Lines are the values returned when this report was generated; "
            "sportsbook market-update times are shown in the table."
        )

    rb_log = filter_rb_game_log(result.rb_game_logs, participant)
    defense_log = filter_defense_game_log(result.defense_game_logs, participant)
    ui.subheader("Selected RB Historical Game Log")
    if rb_log.empty:
        ui.info("No historical game log is available for the selected expected participant.")
    else:
        ui.dataframe(prepare_rb_log_display(rb_log), use_container_width=True, hide_index=True)
    ui.subheader("Opponent Defense vs RB Game Log")
    if defense_log.empty:
        ui.info("No defense-versus-RB game log is available for the selected opponent.")
    else:
        ui.dataframe(prepare_defense_log_display(defense_log), use_container_width=True, hide_index=True)
    ui.caption("Research data: nflverse via nflreadpy. Sportsbook data: The Odds API.")


def _render_warning_summary(ui: Any, summary: Any) -> None:
    counts = build_warning_counts(summary)
    warning_labels = ["Missing RB history", "Missing defense history", "Unresolved participants", "Shared backfields", "Limited RB samples", "Limited defense samples"]
    messages = [f"{label}: {counts[label]}" for label in warning_labels if counts[label]]
    if messages:
        ui.warning("Report warnings — " + "; ".join(messages))
    mismatches = counts["Historical-team mismatches"]
    if mismatches:
        ui.info(f"Historical-team mismatches: {mismatches}. latest_team reflects historical-season games; team reflects the selected report-week depth chart.")


def _render_selected_context(ui: Any, participant: Any) -> None:
    ui.subheader("Selected Expected Backfield Participant")
    left, right = ui.columns(2)
    with left:
        ui.markdown(
            f"**{display_value(participant.get('player_name'), 'Unnamed RB')}**  \n"
            f"{display_value(participant.get('team'))} vs {display_value(participant.get('opponent'))}  \n"
            f"Depth order: **{display_value(participant.get('participant_order'))}**; "
            f"depth position: **{display_value(participant.get('depth_position'))}**"
        )
    with right:
        ui.markdown(
            f"Source: **{display_value(participant.get('selection_source'))}**  \n"
            f"Historical season: **{display_value(participant.get('historical_season'))}**  \n"
            f"Defensive matchup rank: **{display_value(participant.get('matchup_rank'), 'Unranked')}**"
        )
    ui.caption(
        f"RB season/last 3 rushing yards: {display_value(participant.get('rb_season_rushing_yards_avg'))} / {display_value(participant.get('rb_last3_rushing_yards_avg'))}. "
        f"Opponent season/last 3 RB rushing yards allowed: {display_value(participant.get('defense_season_rb_rushing_yards_avg_allowed'))} / {display_value(participant.get('defense_last3_rb_rushing_yards_avg_allowed'))}."
    )
    flags = {
        "rb_history_missing": "missing RB history",
        "defense_history_missing": "missing defense history",
        "participant_team_mismatch": "historical-team mismatch",
        "multiple_expected_rbs": "shared backfield",
        "limited_rb_sample": "limited RB sample",
        "limited_defense_sample": "limited defense sample",
    }
    active_flags = [
        label for column, label in flags.items() if _enabled(participant.get(column, False))
    ]
    if active_flags:
        ui.caption("Participant warnings: " + "; ".join(active_flags) + ".")


def _enabled(value: object) -> bool:
    return False if pd.isna(value) else bool(value)
