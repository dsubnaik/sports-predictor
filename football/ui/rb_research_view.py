"""Pure display preparation for the weekly RB research Streamlit page."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import pandas as pd

from football.decisions import PendingDecision
from football.pipeline.build_weekly_player_prop_odds import (
    WEEKLY_PLAYER_PROP_ODDS_COLUMNS,
    WeeklyPlayerPropOddsResult,
)


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

RUSHING_PROP_DISPLAY_COLUMNS = [
    "Sportsbook",
    "Rushing Yards Line",
    "Over Price",
    "Under Price",
    "Market Updated",
]


@dataclass(frozen=True)
class ParticipantOption:
    """Stable selectbox option for one resolved scheduled RB participant."""

    option_id: str
    label: str


@dataclass(frozen=True)
class RBDecisionOutcomeOption:
    """One stable, matched rushing-yard outcome eligible for recording."""

    option_id: str
    label: str


class RBDecisionEntryValidationError(ValueError):
    """Raised when displayed RB odds cannot safely form a decision snapshot."""


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


def filter_selected_rb_rushing_props(
    player_matched_odds: pd.DataFrame,
    participant: pd.Series,
) -> pd.DataFrame:
    """Return matched rushing quotes for one selected RB participant-game."""

    _require_player_prop_schema(player_matched_odds)
    empty = player_matched_odds.iloc[0:0].copy(deep=True).reset_index(drop=True)
    game_id = participant.get("game_id")
    player_id = participant.get("player_id")
    if _blank(game_id) or _blank(player_id):
        return empty
    rows = player_matched_odds.loc[
        (player_matched_odds["market_key"] == "player_rush_yds")
        & (player_matched_odds["event_match_status"] == "matched")
        & (player_matched_odds["match_status"] == "matched")
        & (player_matched_odds["nflverse_game_id"] == game_id)
        & (player_matched_odds["player_id"] == player_id)
    ].copy(deep=True)
    return rows.reset_index(drop=True)


def build_rb_decision_outcome_options(
    props: pd.DataFrame,
) -> list[RBDecisionOutcomeOption]:
    """Return deterministic exact matched-outcome choices without mutating ``props``."""

    _require_player_prop_schema(props)
    choices: dict[str, tuple[tuple[str, ...], str]] = {}
    for _, row in _eligible_rb_decision_rows(props).iterrows():
        option_id = _rb_decision_option_id(row)
        snapshot = _rb_snapshot_values(row)
        label = _rb_decision_option_label(row)
        existing = choices.get(option_id)
        if existing is not None and existing[0] != snapshot:
            raise RBDecisionEntryValidationError(
                "conflicting matched odds rows share one decision selection identity"
            )
        if existing is None or label < existing[1]:
            choices[option_id] = (snapshot, label)
    return [
        RBDecisionOutcomeOption(option_id=option_id, label=values[1])
        for option_id, values in sorted(choices.items(), key=lambda item: (item[1][1], item[0]))
    ]


def build_rb_pending_decision(
    props: pd.DataFrame,
    participant: pd.Series,
    option_id: object,
    odds_retrieved_at: object,
    recorded_at: object,
    research_notes: object | None = None,
) -> PendingDecision:
    """Create one RB pending-decision input from a currently displayed outcome.

    The UI validates exact participant and game identity here; the decision
    store remains authoritative for line, price, notes, and timestamp rules.
    """

    _require_player_prop_schema(props)
    if not isinstance(option_id, str) or not option_id:
        raise RBDecisionEntryValidationError("a displayed rushing-yard outcome must be selected")
    matches = [
        row.copy(deep=True)
        for _, row in _eligible_rb_decision_rows(props).iterrows()
        if _rb_decision_option_id(row) == option_id
    ]
    if not matches:
        raise RBDecisionEntryValidationError(
            "the selected rushing-yard outcome is no longer available for this participant"
        )
    if len({_rb_snapshot_values(row) for row in matches}) != 1:
        raise RBDecisionEntryValidationError(
            "conflicting matched odds rows share one decision selection identity"
        )
    row = min(matches, key=_rb_row_sort_key)
    context = _rb_decision_context(participant, row)
    return PendingDecision(
        position="RB",
        player_id=context["player_id"],
        player_name=_required_context_text(row.get("nflverse_player_name"), "player_name"),
        team=context["team"],
        opponent=context["opponent"],
        season=row.get("nflverse_season"),
        week=row.get("nflverse_week"),
        game_id=context["game_id"],
        market_key="player_rush_yds",
        sportsbook=_required_context_text(row.get("bookmaker_key"), "sportsbook"),
        line=row.get("point"),
        selection=_required_context_text(row.get("outcome_name"), "selection").lower(),
        selected_price=row.get("price"),
        odds_retrieved_at=odds_retrieved_at,
        recorded_at=recorded_at,
        research_notes=research_notes,
    )


def _eligible_rb_decision_rows(props: pd.DataFrame) -> pd.DataFrame:
    return props.loc[
        (props["market_key"] == "player_rush_yds")
        & (props["event_match_status"] == "matched")
        & (props["match_status"] == "matched")
    ].copy(deep=True)


def _rb_decision_option_id(row: pd.Series) -> str:
    return json.dumps(
        [_identity_value(row.get(column)) for column in (
            "bookmaker_key", "nflverse_game_id", "player_id", "market_key",
            "point", "outcome_name", "price",
        )],
        ensure_ascii=True,
        separators=(",", ":"),
    )


def _rb_snapshot_values(row: pd.Series) -> tuple[str, ...]:
    return tuple(_identity_value(row.get(column)) for column in (
        "bookmaker_key", "nflverse_game_id", "player_id", "market_key", "point",
        "outcome_name", "price", "nflverse_season", "nflverse_week",
        "nflverse_player_name", "nflverse_team",
    ))


def _rb_decision_option_label(row: pd.Series) -> str:
    title = display_value(row.get("bookmaker_title"), _identity_value(row.get("bookmaker_key")))
    return (
        f"{title} ({_identity_value(row.get('bookmaker_key'))}) | "
        f"Rushing yards {_identity_value(row.get('point'))} | "
        f"{_identity_value(row.get('outcome_name'))} {_identity_value(row.get('price'))}"
    )


def _rb_row_sort_key(row: pd.Series) -> tuple[str, ...]:
    return tuple(_identity_value(row.get(column)) for column in WEEKLY_PLAYER_PROP_ODDS_COLUMNS)


def _rb_decision_context(participant: pd.Series, row: pd.Series) -> dict[str, str]:
    if not isinstance(participant, pd.Series):
        raise RBDecisionEntryValidationError("selected RB participant is invalid")
    game_id = _required_context_text(participant.get("game_id"), "game_id")
    player_id = _required_context_text(participant.get("player_id"), "player_id")
    team = _required_context_text(participant.get("team"), "team")
    opponent = _required_context_text(participant.get("opponent"), "opponent")
    if _required_context_text(row.get("nflverse_game_id"), "game_id") != game_id:
        raise RBDecisionEntryValidationError("selected odds no longer match the selected RB game")
    if _required_context_text(row.get("player_id"), "player_id") != player_id:
        raise RBDecisionEntryValidationError("selected odds no longer match the selected RB participant")
    if _required_context_text(row.get("nflverse_team"), "team") != team:
        raise RBDecisionEntryValidationError("selected odds team does not match the selected RB context")
    home_team = _required_context_text(row.get("nflverse_home_team"), "home team")
    away_team = _required_context_text(row.get("nflverse_away_team"), "away team")
    if {home_team, away_team} != {team, opponent}:
        raise RBDecisionEntryValidationError("selected odds opponent does not match the selected RB context")
    if _identity_value(row.get("nflverse_season")) != _identity_value(participant.get("report_season")):
        raise RBDecisionEntryValidationError("selected odds season does not match the selected RB context")
    if _identity_value(row.get("nflverse_week")) != _identity_value(participant.get("report_week")):
        raise RBDecisionEntryValidationError("selected odds week does not match the selected RB context")
    return {"game_id": game_id, "player_id": player_id, "team": team, "opponent": opponent}


def _required_context_text(value: object, field: str) -> str:
    if _blank(value) or not isinstance(value, str):
        raise RBDecisionEntryValidationError(f"selected odds {field} must be nonblank")
    return value.strip()


def _identity_value(value: object) -> str:
    return "<missing>" if pd.isna(value) else str(value).strip()


def prepare_rushing_prop_display(props: pd.DataFrame) -> pd.DataFrame:
    """Pivot side-level rushing quotes into deterministic sportsbook rows."""

    _require_player_prop_schema(props)
    if props.empty:
        return pd.DataFrame(columns=RUSHING_PROP_DISPLAY_COLUMNS)

    display_rows: list[dict[str, object]] = []
    grouping_columns = ["event_id", "bookmaker_key", "player_id", "market_key", "point"]
    for key, group in props.groupby(grouping_columns, sort=True, dropna=False):
        _event_id, bookmaker_key, _player_id, market_key, point = key
        if market_key != "player_rush_yds":
            raise ValueError("Rushing-prop display requires player_rush_yds rows")
        updates = group["market_last_update"].dropna().drop_duplicates()
        if len(updates) > 1:
            raise ValueError("Conflicting market update timestamps for one rushing line")
        prices: dict[str, object] = {}
        for side, side_rows in group.groupby("outcome_name", sort=True, dropna=False):
            if side not in {"Over", "Under"}:
                raise ValueError("Rushing-prop display requires Over or Under outcomes")
            if len(side_rows) != 1:
                raise ValueError("Duplicate sportsbook outcome for one rushing line")
            prices[side] = side_rows.iloc[0]["price"]

        title_values = group["bookmaker_title"].dropna().astype("string").str.strip()
        title_values = title_values[title_values != ""].drop_duplicates()
        if len(title_values) > 1:
            raise ValueError("Conflicting bookmaker titles for one rushing line")
        display_rows.append(
            {
                "Sportsbook": title_values.iloc[0] if len(title_values) else bookmaker_key,
                "Rushing Yards Line": point,
                "Over Price": prices.get("Over", pd.NA),
                "Under Price": prices.get("Under", pd.NA),
                "Market Updated": updates.iloc[0] if len(updates) else pd.NA,
            }
        )

    return pd.DataFrame(display_rows, columns=RUSHING_PROP_DISPLAY_COLUMNS).sort_values(
        ["Sportsbook", "Rushing Yards Line"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)


def build_selected_rb_prop_warnings(
    player_prop_odds: WeeklyPlayerPropOddsResult | None,
    participant: pd.Series,
    selected_props: pd.DataFrame | None = None,
) -> list[str]:
    """Return concise rushing-odds diagnostics for the selected participant."""

    if player_prop_odds is None:
        return ["Rushing-yard odds were not requested for this report."]
    game_id = participant.get("game_id")
    player_id = participant.get("player_id")
    if _blank(player_id):
        return ["Selected participant is unresolved, so rushing-yard lines cannot be matched."]
    if _blank(game_id):
        return ["Selected participant has no game identifier for rushing-yard line matching."]

    odds = player_prop_odds.player_matched_odds
    _require_player_prop_schema(odds)
    warnings: list[str] = []
    event_rows = _selected_event_diagnostic_rows(odds, participant)
    if (event_rows["event_match_status"] == "unmatched").any():
        warnings.append("A sportsbook event for this matchup could not be matched to the schedule.")
    if (event_rows["event_match_status"] == "ambiguous").any():
        warnings.append("A sportsbook event for this matchup matched multiple schedule games.")

    player_rows = odds.loc[
        (odds["market_key"] == "player_rush_yds")
        & (odds["event_match_status"] == "matched")
        & (odds["nflverse_game_id"] == game_id)
    ]
    if (player_rows["match_status"] == "unmatched").any():
        warnings.append("One or more sportsbook players in this game could not be matched to nflverse.")
    if (player_rows["match_status"] == "ambiguous").any():
        warnings.append("One or more sportsbook players in this game have ambiguous nflverse matches.")

    matched_props = selected_props
    if matched_props is None:
        matched_props = filter_selected_rb_rushing_props(odds, participant)
    if matched_props.empty:
        warnings.append("No matched rushing-yard line is available for the selected participant.")
    elif _has_incomplete_pair(matched_props):
        warnings.append("One or more sportsbooks have an incomplete Over/Under rushing-yard pair.")
    return warnings


def format_odds_retrieval_time(value: object) -> str:
    """Format a successful UTC retrieval timestamp deterministically."""

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("odds retrieval time must be timezone-aware")
    return timestamp.tz_convert("UTC").strftime("%Y-%m-%d %H:%M UTC")


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


def _require_player_prop_schema(props: pd.DataFrame) -> None:
    if not isinstance(props, pd.DataFrame):
        raise TypeError("player_matched_odds must be a pandas DataFrame")
    if props.columns.duplicated().any():
        raise ValueError("player_matched_odds must not contain duplicate columns")
    if props.columns.tolist() != WEEKLY_PLAYER_PROP_ODDS_COLUMNS:
        raise ValueError(
            "player_matched_odds columns must exactly match "
            "WEEKLY_PLAYER_PROP_ODDS_COLUMNS in order"
        )


def _selected_event_diagnostic_rows(
    odds: pd.DataFrame,
    participant: pd.Series,
) -> pd.DataFrame:
    teams = {participant.get("team"), participant.get("opponent")}
    if any(_blank(team) for team in teams):
        return odds.iloc[0:0].copy(deep=True)
    return odds.loc[
        odds["nflverse_home_team"].isin(teams)
        & odds["nflverse_away_team"].isin(teams)
    ].copy(deep=True)


def _has_incomplete_pair(props: pd.DataFrame) -> bool:
    grouping_columns = ["event_id", "bookmaker_key", "player_id", "market_key", "point"]
    return any(
        set(group["outcome_name"]) != {"Over", "Under"}
        for _, group in props.groupby(grouping_columns, sort=False, dropna=False)
    )
