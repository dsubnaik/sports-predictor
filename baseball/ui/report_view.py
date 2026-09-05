"""Pure display helpers for the baseball Streamlit report."""

from __future__ import annotations

from typing import Any


def filter_pitcher_lines(
    lines: list[dict[str, Any]],
    search: str,
) -> list[dict[str, Any]]:
    """Return sportsbook lines matching a pitcher search without mutating input."""

    search_text = search.strip().lower()
    if not search_text:
        return [line.copy() for line in lines]

    return [
        line.copy()
        for line in lines
        if search_text in str(line.get("pitcher", "")).lower()
    ]


def build_projection_rows(
    lines: list[dict[str, Any]],
    projections: dict[str, float | None],
) -> list[dict[str, Any]]:
    """Attach saved-model projections to parsed sportsbook line dictionaries."""

    rows: list[dict[str, Any]] = []
    for line in lines:
        row = line.copy()
        row["predicted_ks"] = projections.get(str(line.get("pitcher", "")))
        rows.append(row)
    return rows
