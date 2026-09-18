"""Streamlit entrypoint for Sports Predictor."""

from __future__ import annotations

import streamlit as st

from baseball.ui.baseball_report_page import render_baseball_page
from football.ui.decision_history_page import render_decision_history_page
from football.ui.football_research_page import render_football_research_page
from football.ui.qb_research_page import render_football_qb_research_page
from football.ui.rb_research_page import render_football_rb_research_page


st.set_page_config(page_title="Sports Predictor", layout="wide")

pages = {
    "Football": [
        st.Page(render_football_research_page, title="Football Research", default=True),
        st.Page(render_football_qb_research_page, title="QB Research"),
        st.Page(render_football_rb_research_page, title="RB Research"),
        st.Page(render_decision_history_page, title="Decision History"),
    ],
    "Baseball": [st.Page(render_baseball_page, title="Baseball Research")],
}

st.navigation(pages, position="top").run()
