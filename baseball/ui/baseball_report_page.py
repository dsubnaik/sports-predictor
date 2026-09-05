"""Streamlit page for MLB pitcher strikeout projections."""

from __future__ import annotations

from typing import Any

import streamlit as st

from baseball.model.projection import (
    PitcherProjectionUnavailable,
    build_pitcher_projection,
)
from baseball.odds.fetch_lines import fetch_strikeout_lines, parse_lines
from baseball.ui.report_view import build_projection_rows, filter_pitcher_lines


@st.cache_data(ttl=3600)
def load_lines() -> list[dict[str, Any]]:
    """Fetch and parse sportsbook lines, cached to protect API credits."""

    return parse_lines(fetch_strikeout_lines())


@st.cache_data(ttl=86400)
def get_pitcher_projection(pitcher_name: str) -> float | None:
    """Return the latest saved-model strikeout projection for one pitcher."""

    try:
        return build_pitcher_projection(pitcher_name)
    except PitcherProjectionUnavailable:
        return None


def render_baseball_page() -> None:
    """Render the baseball report page."""

    st.title("Gain an Edge")
    st.subheader("MLB Pitcher Strikeout Projections")
    st.info("Choose your settings and generate a report.")

    with st.form("baseball_report_form"):
        search_col, button_col = st.columns([3, 1])
        with search_col:
            search = st.text_input("Search Pitcher...", key="baseball_search_input")
        with button_col:
            st.write("")
            generate_baseball_report = st.form_submit_button(
                "Generate Baseball Report",
                type="primary",
            )

    if generate_baseball_report:
        with st.spinner("Loading sportsbook lines and saved-model projections..."):
            lines = filter_pitcher_lines(load_lines(), search)
            projections = {
                line["pitcher"]: get_pitcher_projection(line["pitcher"])
                for line in lines
            }
            st.session_state["baseball_report_rows"] = build_projection_rows(
                lines,
                projections,
            )
            st.session_state["baseball_report_search"] = search

    rows = st.session_state.get("baseball_report_rows")

    if rows is None:
        st.caption("No baseball data will load until you generate the report.")
        st.stop()

    if not rows:
        st.info("No pitcher lines matched your report settings.")
        st.stop()

    generated_search = st.session_state.get("baseball_report_search")
    search_suffix = f" for search: {generated_search}" if generated_search else ""
    st.caption(f"Showing generated baseball report{search_suffix}.")

    for line in rows:
        with st.container(border=True):
            predicted_ks = line["predicted_ks"]
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"### {line['pitcher']}")
                st.caption(line["game"])

            with col2:
                st.metric(label="Line", value=line["line"])
                if predicted_ks is not None:
                    st.metric(label="Projection", value=f"{predicted_ks:.1f} K")
                else:
                    st.metric(label="Projection", value="N/A")

            if predicted_ks is not None:
                diff = predicted_ks - line["line"]
                if diff > 0:
                    st.success(f"↑ LEAN OVER — {diff:.1f} above the line")
                elif diff < 0:
                    st.error(f"↓ LEAN UNDER — {abs(diff):.1f} below the line")
                else:
                    st.warning("⚖ TOO CLOSE TO CALL")
            else:
                st.warning("No 2025 data available for this pitcher")
