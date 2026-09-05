"""Streamlit entrypoint for Sports Predictor."""

from __future__ import annotations

import streamlit as st

from baseball.ui.baseball_report_page import render_baseball_page
from football.ui.qb_research_page import render_football_qb_research_page


st.set_page_config(page_title="Sports Predictor", layout="wide")

pages = [
    st.Page(render_baseball_page, title="Baseball"),
    st.Page(render_football_qb_research_page, title="Football QB Research"),
]

st.navigation(pages, position="top").run()
