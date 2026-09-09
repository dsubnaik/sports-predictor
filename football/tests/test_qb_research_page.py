"""Small non-interactive tests for the QB page's Generate boundary."""

from datetime import date

import pandas as pd

import football.ui.qb_research_page as qb_page
from football.pipeline import WeeklyQBResearchResult


def test_generate_loader_requests_passing_props_on_every_deliberate_call(monkeypatch):
    calls = []
    result = WeeklyQBResearchResult(
        summary=pd.DataFrame(),
        qb_game_logs=pd.DataFrame(),
        defense_game_logs=pd.DataFrame(),
    )

    def fake_builder(**kwargs):
        calls.append(kwargs)
        return result

    monkeypatch.setattr(qb_page, "build_weekly_qb_research", fake_builder)

    assert qb_page.load_qb_research(2026, 1, date(2026, 9, 10), 2025) is result
    assert qb_page.load_qb_research(2026, 1, date(2026, 9, 10), 2025) is result
    assert len(calls) == 2
    assert all(call["include_player_props"] is True for call in calls)
