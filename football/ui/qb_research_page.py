"""Streamlit page for weekly football quarterback research."""

from __future__ import annotations

from datetime import date
from urllib.error import URLError

import streamlit as st

try:
    from requests.exceptions import RequestException
except ImportError:  # pragma: no cover - requests is an app dependency.
    RequestException = OSError

from football.pipeline import WeeklyQBResearchResult, build_weekly_qb_research
from football.ui.qb_research_view import (
    DEFENSIVE_MATCHUP_RANK_HELP,
    build_matchup_options,
    build_matchup_warnings,
    default_history_season,
    display_value,
    filter_defense_game_log,
    filter_qb_game_log,
    find_matchup,
    history_label,
    prepare_defense_log_display,
    prepare_qb_log_display,
    prepare_summary_display,
)


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_qb_research(
    report_season: int,
    report_week: int,
    as_of_date: date,
    history_season: int,
) -> WeeklyQBResearchResult:
    """Build the weekly report once per input set to avoid repeated downloads."""

    return build_weekly_qb_research(
        report_season=report_season,
        report_week=report_week,
        as_of_date=as_of_date,
        history_season=history_season,
    )


def render_football_qb_research_page() -> None:
    """Render the weekly football quarterback research page."""

    st.title("Football QB Research")
    st.info("Choose your settings and generate a report.")

    today = date.today()
    with st.form("football_report_form"):
        season_col, week_col, date_col, history_col, button_col = st.columns(
            [1, 1, 1.3, 1, 1.2]
        )
        with season_col:
            report_season = st.number_input(
                "Report season",
                min_value=1999,
                max_value=today.year + 1,
                value=today.year,
                step=1,
            )
        with week_col:
            report_week = st.number_input(
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
            as_of_date = st.date_input("As-of date", value=today)
        with history_col:
            history_season = st.number_input(
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
            st.write("")
            generate_report = st.form_submit_button(
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
            with st.spinner("Loading nflverse data and building QB research tables..."):
                result = load_qb_research(
                    report_inputs["report_season"],
                    report_inputs["report_week"],
                    report_inputs["as_of_date"],
                    report_inputs["history_season"],
                )
                st.session_state["football_qb_research_inputs"] = report_inputs
                st.session_state["football_qb_research_result"] = result
        except ValueError as error:
            st.error(f"Could not build the report from the available data: {error}")
            st.caption("Data source: nflverse via nflreadpy.")
            st.stop()
        except (ImportError, ModuleNotFoundError) as error:
            st.error(f"Could not load the nflverse dependency: {error}")
            st.caption("Data source: nflverse via nflreadpy.")
            st.stop()
        except (OSError, TimeoutError, URLError, RequestException) as error:
            st.error(
                "Could not download nflverse data. Check the network and try again. "
                f"{error}"
            )
            st.caption("Data source: nflverse via nflreadpy.")
            st.stop()

    report_inputs = st.session_state.get("football_qb_research_inputs")
    result = st.session_state.get("football_qb_research_result")

    if report_inputs is None or result is None:
        st.caption("No football data will load until you generate the report.")
        st.caption("Data source: nflverse via nflreadpy.")
        st.stop()

    st.caption(
        f"Report season {report_inputs['report_season']}, "
        f"Week {report_inputs['report_week']}. "
        f"Historical season used: {report_inputs['history_season']}."
    )

    summary_display = prepare_summary_display(result.summary)
    st.subheader("Ranked Matchup Summary")
    st.caption(DEFENSIVE_MATCHUP_RANK_HELP)
    if summary_display.empty:
        st.info("No scheduled team matchups were found for these report inputs.")
        st.caption("Data source: nflverse via nflreadpy.")
        st.stop()

    st.dataframe(summary_display, use_container_width=True, hide_index=True)

    options = build_matchup_options(result.summary)
    option_by_label = {option.label: option.option_id for option in options}
    selected_label = st.selectbox("Matchup", list(option_by_label))
    selected_matchup = find_matchup(result.summary, option_by_label[selected_label])

    if selected_matchup is None:
        st.error("The selected matchup could not be found in the report result.")
        st.caption("Data source: nflverse via nflreadpy.")
        st.stop()

    st.subheader("Selected Matchup")
    left, right = st.columns(2)
    with left:
        st.markdown(
            f"**{display_value(selected_matchup.get('team'))} vs "
            f"{display_value(selected_matchup.get('opponent'))}**  \n"
            f"Expected QB: "
            f"**{display_value(selected_matchup.get('expected_player_name'), 'Unresolved')}**"
        )
        st.caption(history_label(selected_matchup))
    with right:
        st.markdown(
            f"Expected-QB source: "
            f"**{display_value(selected_matchup.get('selection_source'))}**  \n"
            f"Depth-chart date: "
            f"**{display_value(selected_matchup.get('depth_chart_date'))}**"
        )
        notes = selected_matchup.get("selection_notes")
        notes_text = display_value(notes, fallback="")
        if notes_text:
            st.caption(notes_text)

    for warning in build_matchup_warnings(selected_matchup):
        st.warning(warning)

    qb_log = filter_qb_game_log(result.qb_game_logs, selected_matchup)
    defense_log = filter_defense_game_log(result.defense_game_logs, selected_matchup)

    st.subheader("Selected QB Historical Game Log")
    if qb_log.empty:
        st.info("No historical game log is available for the selected expected QB.")
    else:
        st.dataframe(
            prepare_qb_log_display(qb_log),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Opponent Defense vs QB Game Log")
    if defense_log.empty:
        st.info("No defense-versus-QB game log is available for the selected opponent.")
    else:
        st.dataframe(
            prepare_defense_log_display(defense_log),
            use_container_width=True,
            hide_index=True,
        )

    st.caption("Data source: nflverse via nflreadpy.")
