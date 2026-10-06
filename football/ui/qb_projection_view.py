"""Pure snapshot loading, matching, and presentation for the QB research UI."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import pandas as pd

from football.odds.player_matching import match_player_prop_odds
from football.odds.player_props import PLAYER_PROP_ODDS_COLUMNS
from football.pipeline import (
    QBPassingYardsWeeklyProjectionSnapshot,
    load_qb_passing_yards_weekly_projection_snapshot,
)


@dataclass(frozen=True)
class QBProjectionSnapshotLoadResult:
    status: str
    snapshot: QBPassingYardsWeeklyProjectionSnapshot | None


@dataclass(frozen=True)
class QBProjectionLineComparison:
    sportsbook: str | None
    line: float
    projection_minus_line: float
    direction_text: str


@dataclass(frozen=True)
class QBProjectionPresentation:
    status: str
    message: str
    match_method: str | None
    predicted_passing_yards: float | None
    displayed_projection: str | None
    season: int | None
    week: int | None
    as_of_display: str | None
    line_comparisons: tuple[QBProjectionLineComparison, ...]


def load_qb_projection_snapshot_for_ui(
    path: str | Path,
    *,
    expected_artifact_sha256: str | None = None,
) -> QBProjectionSnapshotLoadResult:
    """Load only the public JSON snapshot API and classify safe UI outcomes."""
    try:
        snapshot = load_qb_passing_yards_weekly_projection_snapshot(
            path, expected_artifact_sha256=expected_artifact_sha256,
        )
    except FileNotFoundError:
        return QBProjectionSnapshotLoadResult("snapshot_unavailable", None)
    except (OSError, ValueError):
        return QBProjectionSnapshotLoadResult("snapshot_invalid", None)
    return QBProjectionSnapshotLoadResult("snapshot_available", snapshot)


def build_qb_projection_presentation(
    loaded: QBProjectionSnapshotLoadResult,
    selected_matchup: pd.Series,
    selected_props: pd.DataFrame | None,
) -> QBProjectionPresentation:
    """Match one immutable snapshot projection to one already selected matchup."""
    if not isinstance(loaded, QBProjectionSnapshotLoadResult):
        raise TypeError("loaded must be a QBProjectionSnapshotLoadResult")
    if not isinstance(selected_matchup, pd.Series):
        raise TypeError("selected_matchup must be a pandas Series")
    if loaded.status == "snapshot_unavailable":
        return _unavailable("snapshot_unavailable", "Saved model projection snapshot is unavailable.")
    if loaded.status != "snapshot_available" or loaded.snapshot is None:
        return _unavailable("snapshot_invalid", "Saved model projection snapshot is invalid or incompatible.")

    context = _matchup_context(selected_matchup)
    snapshot = loaded.snapshot
    if (snapshot.metadata.season, snapshot.metadata.week) != (context["season"], context["week"]):
        return _unavailable("snapshot_stale", "Saved model projection snapshot is for a different season or week.")
    candidates = tuple(
        value for value in snapshot.projections
        if value.game_id == context["game_id"]
        and value.team == context["team"]
        and value.opponent == context["opponent"]
    )
    if not candidates:
        return _unavailable("projection_unavailable", "No model projection is available for this game and QB.")

    expected_id = context["player_id"]
    if expected_id is not None:
        matched = tuple(value for value in candidates if value.player_id == expected_id)
        method = "player_id"
    else:
        matched, method, name_ambiguous = _name_fallback(candidates, context)
        if name_ambiguous:
            return _unavailable("projection_ambiguous", "More than one model projection matches this game and QB.")
    if len(matched) > 1:
        return _unavailable("projection_ambiguous", "More than one model projection matches this game and QB.")
    if not matched:
        return _unavailable("projection_unavailable", "No model projection is available for this game and QB.")

    projection = matched[0]
    comparisons = _line_comparisons(selected_props, projection.predicted_passing_yards)
    return QBProjectionPresentation(
        status="matched",
        message=(
            "Unscored model estimate; not a betting recommendation."
            if comparisons else
            "Unscored model estimate. No matching passing-yards line is available."
        ),
        match_method=method,
        predicted_passing_yards=projection.predicted_passing_yards,
        displayed_projection=f"{projection.predicted_passing_yards:.1f}",
        season=snapshot.metadata.season,
        week=snapshot.metadata.week,
        as_of_display=_format_utc(snapshot.metadata.as_of_utc),
        line_comparisons=comparisons,
    )


def _unavailable(status: str, message: str) -> QBProjectionPresentation:
    return QBProjectionPresentation(status, message, None, None, None, None, None, None, ())


def _matchup_context(matchup: pd.Series) -> dict[str, object]:
    season = _positive_int(matchup.get("season"), "season")
    week = _positive_int(matchup.get("report_week"), "report_week")
    return {
        "season": season,
        "week": week,
        "game_id": _text(matchup.get("game_id"), "game_id"),
        "team": _text(matchup.get("team"), "team"),
        "opponent": _text(matchup.get("opponent"), "opponent"),
        "player_id": _optional_text(matchup.get("expected_player_id")),
        "player_name": _optional_text(matchup.get("expected_player_name")),
    }


def _name_fallback(candidates: tuple[Any, ...], context: dict[str, object]) -> tuple[tuple[Any, ...], str | None, bool]:
    name = context["player_name"]
    if name is None:
        return (), None, False
    odds = pd.DataFrame([{
        "event_id": "snapshot-fallback", "commence_time": "2000-01-01T00:00:00Z",
        "home_team": str(context["team"]), "away_team": str(context["opponent"]),
        "bookmaker_key": "snapshot", "bookmaker_title": "Snapshot",
        "bookmaker_last_update": pd.NA, "market_key": "player_pass_yds",
        "market_last_update": pd.NA, "player_name": name, "outcome_name": "Over",
        "price": 1, "point": 1.0,
    }], columns=PLAYER_PROP_ODDS_COLUMNS)
    players = pd.DataFrame([
        {"player_id": value.player_id, "player_name": value.player_name, "team": value.team, "position": "QB"}
        for value in candidates if value.player_name is not None
    ], columns=["player_id", "player_name", "team", "position"])
    if players.empty:
        return (), None, False
    matched = match_player_prop_odds(odds, players).iloc[0]
    if matched["match_status"] == "ambiguous":
        return (), None, True
    if matched["match_status"] != "matched":
        return (), None, False
    player_id = str(matched["player_id"])
    return tuple(value for value in candidates if value.player_id == player_id), str(matched["match_method"]), False


def _line_comparisons(selected_props: pd.DataFrame | None, projection: float) -> tuple[QBProjectionLineComparison, ...]:
    if selected_props is None or not isinstance(selected_props, pd.DataFrame) or selected_props.empty:
        return ()
    required = {"point", "bookmaker_key"}
    if not required.issubset(selected_props.columns):
        return ()
    lines: dict[tuple[str, float], QBProjectionLineComparison] = {}
    for _, row in selected_props.loc[:, ["bookmaker_key", "point"]].iterrows():
        try:
            point = float(row["point"])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(point):
            continue
        sportsbook = _optional_text(row["bookmaker_key"])
        key = (sportsbook or "", point)
        difference = projection - point
        if difference > 0:
            text = f"Model is {difference:.1f} yards above the line."
        elif difference < 0:
            text = f"Model is {abs(difference):.1f} yards below the line."
        else:
            text = "Model equals the line."
        lines[key] = QBProjectionLineComparison(sportsbook, point, difference, text)
    return tuple(lines[key] for key in sorted(lines))


def _format_utc(value: object) -> str:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("snapshot as-of timestamp must be timezone-aware")
    return timestamp.tz_convert("UTC").strftime("%Y-%m-%d %H:%M UTC")


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"selected matchup {label} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"selected matchup {label} must be a positive integer") from error
    if result < 1 or result != value:
        raise ValueError(f"selected matchup {label} must be a positive integer")
    return result


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"selected matchup {label} must be nonblank text")
    return value.strip()


def _optional_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return _text(value, "value")
