"""Pure display preparation for the weekly QB research Streamlit page."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


SUMMARY_DISPLAY_COLUMNS = {
    "matchup_rank": "Defensive Matchup Rank",
    "expected_player_name": "Expected QB",
    "team": "Team",
    "opponent": "Opponent",
    "qb_season_avg": "QB Season Avg",
    "qb_last3_avg": "QB Last 3 Avg",
    "qb_season_attempts_avg": "QB Season Att Avg",
    "defense_season_avg_allowed": "Defense Season Avg Allowed",
    "defense_last3_avg_allowed": "Defense Last 3 Avg Allowed",
    "qb_season_games": "QB Sample",
    "defense_season_games": "Defense Sample",
    "missing_qb_history": "Missing QB History",
    "missing_defense_history": "Missing Defense History",
}

DEFENSIVE_MATCHUP_RANK_HELP = (
    "Rank 1 means the opponent allowed the most passing yards per game in the "
    "selected historical season. This is a matchup ranking, not a quarterback "
    "talent ranking or model prediction."
)

SUMMARY_ROUND_COLUMNS = [
    "qb_season_avg",
    "qb_last3_avg",
    "qb_season_attempts_avg",
    "defense_season_avg_allowed",
    "defense_last3_avg_allowed",
]

QB_LOG_DISPLAY_COLUMNS = {
    "week": "Week",
    "opponent": "Opponent",
    "passing_attempts": "Attempts",
    "completions": "Completions",
    "passing_yards": "Passing Yards",
    "passing_touchdowns": "Passing TD",
    "interceptions": "Interceptions",
    "low_attempt_primary_qb": "Low-Attempt Primary QB",
    "similar_attempt_split": "Similar Attempt Split",
    "exact_attempt_tie": "Exact Attempt Tie",
}

DEFENSE_LOG_DISPLAY_COLUMNS = {
    "week": "Week",
    "opposing_qb_name": "Opposing QB",
    "offense_team": "Offense Team",
    "passing_attempts_allowed": "Attempts Allowed",
    "completions_allowed": "Completions Allowed",
    "passing_yards_allowed": "Passing Yards Allowed",
    "passing_touchdowns_allowed": "Passing TD Allowed",
    "opposing_interceptions": "Interceptions",
    "low_attempt_primary_qb": "Low-Attempt Primary QB",
    "similar_attempt_split": "Similar Attempt Split",
    "exact_attempt_tie": "Exact Attempt Tie",
}

DETAIL_SORT_COLUMNS = ["season", "week", "game_id"]


@dataclass(frozen=True)
class MatchupOption:
    """Stable selectbox option for one scheduled team-game."""

    option_id: str
    label: str


def default_history_season(report_season: int, report_week: int) -> int:
    """Return the pipeline's automatic history-season default."""

    return report_season - 1 if report_week == 1 else report_season


def prepare_summary_display(summary: pd.DataFrame) -> pd.DataFrame:
    """Return user-facing summary columns without mutating backend rows."""

    display = summary.copy(deep=True)
    for column in SUMMARY_ROUND_COLUMNS:
        if column in display.columns:
            display[column] = pd.to_numeric(display[column], errors="coerce").round(1)

    available_columns = [
        column for column in SUMMARY_DISPLAY_COLUMNS if column in display.columns
    ]
    return display.loc[:, available_columns].rename(columns=SUMMARY_DISPLAY_COLUMNS)


def build_matchup_options(summary: pd.DataFrame) -> list[MatchupOption]:
    """Build stable matchup options from team-game keys."""

    if summary.empty:
        return []

    options: list[MatchupOption] = []
    rows = summary.copy(deep=True).reset_index(drop=True)
    for _, row in rows.iterrows():
        option_id = matchup_option_id(row)
        rank = _display_value(row.get("matchup_rank"), fallback="Unranked")
        team = _display_value(row.get("team"))
        opponent = _display_value(row.get("opponent"))
        qb_name = _display_value(row.get("expected_player_name"), fallback="Unresolved QB")
        game_id = _display_value(row.get("game_id"))
        label = f"{rank}: {team} vs {opponent} - {qb_name} ({game_id})"
        options.append(MatchupOption(option_id=option_id, label=label))

    return options


def matchup_option_id(row: pd.Series) -> str:
    """Return a stable team-game identifier for one matchup row."""

    season = _key_value(row.get("season"))
    week = _key_value(row.get("report_week"))
    game_id = _key_value(row.get("game_id"))
    team = _key_value(row.get("team"))
    return f"{season}|{week}|{game_id}|{team}"


def find_matchup(summary: pd.DataFrame, option_id: str) -> pd.Series | None:
    """Find one matchup by the stable option ID."""

    for _, row in summary.copy(deep=True).iterrows():
        if matchup_option_id(row) == option_id:
            return row.copy(deep=True)
    return None


def filter_qb_game_log(qb_game_logs: pd.DataFrame, matchup: pd.Series) -> pd.DataFrame:
    """Return historical games for the selected expected QB."""

    if qb_game_logs.empty:
        return qb_game_logs.copy(deep=True)

    player_id = matchup.get("expected_player_id")
    if pd.isna(player_id) or str(player_id).strip() == "":
        return qb_game_logs.iloc[0:0].copy(deep=True)

    rows = qb_game_logs.loc[qb_game_logs["player_id"] == player_id].copy(deep=True)
    return _sort_details(rows)


def filter_defense_game_log(
    defense_game_logs: pd.DataFrame,
    matchup: pd.Series,
) -> pd.DataFrame:
    """Return historical games for the selected opponent defense."""

    if defense_game_logs.empty:
        return defense_game_logs.copy(deep=True)

    defense = matchup.get("opponent")
    if pd.isna(defense) or str(defense).strip() == "":
        return defense_game_logs.iloc[0:0].copy(deep=True)

    rows = defense_game_logs.loc[defense_game_logs["defense"] == defense].copy(deep=True)
    return _sort_details(rows)


def prepare_qb_log_display(qb_game_log: pd.DataFrame) -> pd.DataFrame:
    """Return selected QB log columns with user-facing names."""

    return _prepare_log_display(qb_game_log, QB_LOG_DISPLAY_COLUMNS)


def prepare_defense_log_display(defense_game_log: pd.DataFrame) -> pd.DataFrame:
    """Return selected defense log columns with user-facing names."""

    return _prepare_log_display(defense_game_log, DEFENSE_LOG_DISPLAY_COLUMNS)


def build_matchup_warnings(matchup: pd.Series) -> list[str]:
    """Return data-quality and missing-history warnings for one matchup."""

    warnings: list[str] = []
    if _is_truthy(matchup.get("starter_uncertain")):
        warnings.append("Expected starter is uncertain.")
    if _is_truthy(matchup.get("missing_qb_history")):
        warnings.append("Selected expected QB has no usable history in this report.")
    if _is_truthy(matchup.get("missing_defense_history")):
        warnings.append("Opponent defense has no usable QB matchup history in this report.")
    if _numeric_value(matchup.get("qb_flagged_games")) > 0:
        warnings.append("QB history includes games with data-quality flags.")
    if _numeric_value(matchup.get("defense_flagged_games")) > 0:
        warnings.append("Defense history includes games with data-quality flags.")
    return warnings


def history_label(matchup: pd.Series) -> str:
    """Return explicit historical-season context for the selected matchup."""

    qb_season = _display_value(matchup.get("qb_history_season"), fallback="N/A")
    defense_season = _display_value(matchup.get("defense_history_season"), fallback="N/A")
    qb_cutoff = _display_value(matchup.get("qb_history_cutoff_week"), fallback="N/A")
    defense_cutoff = _display_value(
        matchup.get("defense_history_cutoff_week"),
        fallback="N/A",
    )
    return (
        f"QB history season: {qb_season} before Week {qb_cutoff}. "
        f"Defense history season: {defense_season} before Week {defense_cutoff}."
    )


def display_value(value: Any, fallback: str = "N/A") -> str:
    """Return a user-facing scalar value with a fallback for blanks."""

    return _display_value(value, fallback=fallback)


def _prepare_log_display(
    log: pd.DataFrame,
    display_columns: dict[str, str],
) -> pd.DataFrame:
    rows = log.copy(deep=True)
    available_columns = [column for column in display_columns if column in rows.columns]
    return rows.loc[:, available_columns].rename(columns=display_columns)


def _sort_details(rows: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [column for column in DETAIL_SORT_COLUMNS if column in rows.columns]
    if not sort_columns:
        return rows.reset_index(drop=True)
    return rows.sort_values(by=sort_columns, kind="mergesort").reset_index(drop=True)


def _display_value(value: Any, fallback: str = "") -> str:
    if pd.isna(value):
        return fallback
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    return text if text else fallback


def _key_value(value: Any) -> str:
    return _display_value(value, fallback="")


def _is_truthy(value: Any) -> bool:
    if pd.isna(value):
        return False
    return bool(value)


def _numeric_value(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
