"""Pure display preparation for the weekly RB research Streamlit page."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


SUMMARY_DISPLAY_COLUMNS = {
    "matchup_rank": "Defensive Matchup Rank",
    "player_name": "Expected Backfield Participant",
    "team": "Team",
    "opponent": "Opponent",
    "participant_order": "Depth Order",
    "selection_source": "Participant Source",
    "selection_notes": "Resolution Note",
    "rb_season_rushing_yards_avg": "RB Season Rush Yds Avg",
    "rb_last3_rushing_yards_avg": "RB Last 3 Rush Yds Avg",
    "rb_season_rushing_attempts_avg": "RB Season Att Avg",
    "rb_last3_rushing_attempts_avg": "RB Last 3 Att Avg",
    "rb_season_opportunities_avg": "RB Season Opportunities Avg",
    "rb_last3_opportunities_avg": "RB Last 3 Opportunities Avg",
    "rb_season_yards_per_carry": "RB Season Yards/Carry",
    "defense_season_rb_rushing_yards_avg_allowed": "Defense Season RB Rush Yds Allowed",
    "defense_last3_rb_rushing_yards_avg_allowed": "Defense Last 3 RB Rush Yds Allowed",
    "defense_season_rb_yards_per_carry_allowed": "Defense Season RB Yards/Carry Allowed",
    "rb_season_games": "RB Sample",
    "defense_season_games": "Defense Sample",
}

DEFENSIVE_MATCHUP_RANK_HELP = (
    "Rank 1 means the opponent allowed the most RB rushing yards per game in the "
    "selected historical period. This is a defensive matchup ranking, not an RB "
    "talent ranking, projection, or composite score."
)

SUMMARY_ROUND_COLUMNS = [
    "rb_season_rushing_yards_avg",
    "rb_last3_rushing_yards_avg",
    "rb_season_rushing_attempts_avg",
    "rb_last3_rushing_attempts_avg",
    "rb_season_opportunities_avg",
    "rb_last3_opportunities_avg",
    "rb_season_yards_per_carry",
    "defense_season_rb_rushing_yards_avg_allowed",
    "defense_last3_rb_rushing_yards_avg_allowed",
    "defense_season_rb_yards_per_carry_allowed",
]

RB_LOG_DISPLAY_COLUMNS = {
    "season": "Season",
    "week": "Week",
    "team": "Historical Team",
    "opponent": "Opponent",
    "rushing_attempts": "Rushing Attempts",
    "rushing_yards": "Rushing Yards",
    "rushing_touchdowns": "Rushing TD",
    "receptions": "Receptions",
    "targets": "Targets",
    "receiving_yards": "Receiving Yards",
    "receiving_touchdowns": "Receiving TD",
    "opportunities": "Opportunities",
    "carry_share": "Carry Share",
    "target_share": "Target Share",
    "opportunity_share": "Opportunity Share",
    "low_volume_rb": "Low Volume",
    "shared_backfield": "Shared Backfield",
}

DEFENSE_LOG_DISPLAY_COLUMNS = {
    "season": "Season",
    "week": "Week",
    "offense_team": "Offense Faced",
    "rb_rushing_attempts_allowed": "RB Rush Attempts Allowed",
    "rb_rushing_yards_allowed": "RB Rush Yards Allowed",
    "rb_rushing_touchdowns_allowed": "RB Rush TD Allowed",
    "rb_receptions_allowed": "RB Receptions Allowed",
    "rb_targets_allowed": "RB Targets Allowed",
    "rb_receiving_yards_allowed": "RB Receiving Yards Allowed",
    "rb_receiving_touchdowns_allowed": "RB Receiving TD Allowed",
    "rb_opportunities_allowed": "RB Opportunities Allowed",
    "rb_players_used": "RB Players Used",
}

DETAIL_SORT_COLUMNS = ["season", "week", "game_id"]


@dataclass(frozen=True)
class ParticipantOption:
    """Stable selectbox option for one resolved scheduled RB participant."""

    option_id: str
    label: str


def default_history_season(report_season: int, report_week: int) -> int:
    """Return the weekly RB pipeline's automatic history-season default."""

    return report_season - 1 if report_week == 1 else report_season


def prepare_summary_display(summary: pd.DataFrame) -> pd.DataFrame:
    """Return concise user-facing summary columns without changing backend rows."""

    display = summary.copy(deep=True)
    for column in SUMMARY_ROUND_COLUMNS:
        if column in display.columns:
            display[column] = pd.to_numeric(display[column], errors="coerce").round(1)

    available = [column for column in SUMMARY_DISPLAY_COLUMNS if column in display]
    return display.loc[:, available].rename(columns=SUMMARY_DISPLAY_COLUMNS)


def build_participant_options(summary: pd.DataFrame) -> list[ParticipantOption]:
    """Build resolved-RB selector options keyed by stable player identity."""

    if summary.empty:
        return []

    rows = summary.copy(deep=True)
    if "participant_resolution_missing" in rows:
        resolved = ~rows["participant_resolution_missing"].fillna(False).astype(bool)
    else:
        resolved = pd.Series(True, index=rows.index)
    valid_id = rows["player_id"].notna() & rows["player_id"].astype("string").str.strip().ne("")
    rows = rows.loc[resolved & valid_id].copy()
    rows["_rank_missing"] = rows["matchup_rank"].isna()
    rows = rows.sort_values(
        by=["_rank_missing", "matchup_rank", "game_id", "team", "participant_order", "player_name", "player_id"],
        kind="mergesort",
        na_position="last",
    )

    options: list[ParticipantOption] = []
    for _, row in rows.iterrows():
        rank = display_value(row.get("matchup_rank"), "Unranked")
        name = display_value(row.get("player_name"), "Unnamed RB")
        team = display_value(row.get("team"))
        opponent = display_value(row.get("opponent"))
        order = display_value(row.get("participant_order"), "N/A")
        options.append(
            ParticipantOption(
                option_id=participant_option_id(row),
                label=f"{rank}: {name} — {team} vs {opponent} (depth {order})",
            )
        )
    return options


def participant_option_id(row: pd.Series) -> str:
    """Return a stable scheduled-participant identifier."""

    fields = ["report_season", "report_week", "game_id", "team", "participant_order", "player_id"]
    return "|".join(display_value(row.get(field), "") for field in fields)


def find_participant(summary: pd.DataFrame, option_id: str) -> pd.Series | None:
    """Return a selected resolved participant by its stable selector ID."""

    for _, row in summary.copy(deep=True).iterrows():
        if participant_option_id(row) == option_id:
            return row.copy(deep=True)
    return None


def filter_rb_game_log(rb_game_logs: pd.DataFrame, participant: pd.Series) -> pd.DataFrame:
    """Return all chronological historical RB games for the selected player ID."""

    if rb_game_logs.empty:
        return rb_game_logs.copy(deep=True)
    player_id = participant.get("player_id")
    if _blank(player_id):
        return rb_game_logs.iloc[0:0].copy(deep=True)
    return _sort_details(rb_game_logs.loc[rb_game_logs["player_id"] == player_id].copy(deep=True))


def filter_defense_game_log(defense_game_logs: pd.DataFrame, participant: pd.Series) -> pd.DataFrame:
    """Return all chronological defense-vs-RB games for the scheduled opponent."""

    if defense_game_logs.empty:
        return defense_game_logs.copy(deep=True)
    defense = participant.get("opponent")
    if _blank(defense):
        return defense_game_logs.iloc[0:0].copy(deep=True)
    return _sort_details(defense_game_logs.loc[defense_game_logs["defense"] == defense].copy(deep=True))


def prepare_rb_log_display(rb_game_log: pd.DataFrame) -> pd.DataFrame:
    """Return RB log columns with user-facing labels and rounded shares."""

    return _prepare_log_display(rb_game_log, RB_LOG_DISPLAY_COLUMNS, share_columns={"carry_share", "target_share", "opportunity_share"})


def prepare_defense_log_display(defense_game_log: pd.DataFrame) -> pd.DataFrame:
    """Return defense game-log columns with user-facing labels."""

    return _prepare_log_display(defense_game_log, DEFENSE_LOG_DISPLAY_COLUMNS)


def build_warning_counts(summary: pd.DataFrame) -> dict[str, int]:
    """Return concise report-level data-quality warning counts."""

    flags = {
        "rb_history_missing": "Missing RB history",
        "defense_history_missing": "Missing defense history",
        "participant_resolution_missing": "Unresolved participants",
        "participant_team_mismatch": "Historical-team mismatches",
        "multiple_expected_rbs": "Shared backfields",
        "limited_rb_sample": "Limited RB samples",
        "limited_defense_sample": "Limited defense samples",
    }
    return {
        label: int(summary.get(column, pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
        for column, label in flags.items()
    }


def display_value(value: Any, fallback: str = "N/A") -> str:
    """Return a user-facing scalar, preserving legitimate numeric zero values."""

    if _blank(value):
        return fallback
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _prepare_log_display(
    log: pd.DataFrame,
    display_columns: dict[str, str],
    share_columns: set[str] | None = None,
) -> pd.DataFrame:
    rows = log.copy(deep=True)
    for column in share_columns or set():
        if column in rows:
            rows[column] = pd.to_numeric(rows[column], errors="coerce").round(3)
    available = [column for column in display_columns if column in rows]
    return rows.loc[:, available].rename(columns=display_columns)


def _sort_details(rows: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in DETAIL_SORT_COLUMNS if column in rows]
    if not columns:
        return rows.reset_index(drop=True)
    return rows.sort_values(by=columns, kind="mergesort").reset_index(drop=True)


def _blank(value: Any) -> bool:
    return pd.isna(value) or (isinstance(value, str) and not value.strip())
