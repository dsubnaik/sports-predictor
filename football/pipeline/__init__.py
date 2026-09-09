"""Reusable football research pipelines."""

from football.pipeline.build_weekly_qb_research import (
    WeeklyQBResearchResult,
    build_weekly_qb_research,
)
from football.pipeline.build_weekly_rb_research import (
    WeeklyRBResearchResult,
    build_weekly_rb_research,
)

__all__ = [
    "WeeklyQBResearchResult",
    "build_weekly_qb_research",
    "WeeklyRBResearchResult",
    "build_weekly_rb_research",
]
