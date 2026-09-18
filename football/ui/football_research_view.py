"""Pure game-first preparation for the combined football research shell."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from football.pipeline import WeeklyQBResearchResult, WeeklyRBResearchResult
from football.pipeline.build_weekly_player_prop_odds import WEEKLY_PLAYER_PROP_ODDS_COLUMNS
from football.ui.qb_research_view import filter_qb_game_log, filter_defense_game_log as filter_qb_defense_game_log, prepare_qb_log_display, prepare_defense_log_display as prepare_qb_defense_log_display
from football.ui.rb_research_view import filter_rb_game_log, filter_defense_game_log as filter_rb_defense_game_log, prepare_rb_log_display, prepare_defense_log_display as prepare_rb_defense_log_display


@dataclass(frozen=True)
class FootballResearchResult:
    """The two existing research results, deliberately kept unmerged."""

    qb_result: WeeklyQBResearchResult | None
    rb_result: WeeklyRBResearchResult | None


@dataclass(frozen=True)
class PropView:
    sportsbook: str
    line: object
    outcome: str
    price: object
    market_updated_at: object
    retrieved_at: object

@dataclass(frozen=True)
class ResearchView:
    player_columns: tuple[str, ...]
    player_rows: tuple[tuple[object, ...], ...]
    defense_columns: tuple[str, ...]
    defense_rows: tuple[tuple[object, ...], ...]


@dataclass(frozen=True)
class ParticipantView:
    position: str
    player_id: str | None
    player_name: str
    role: str
    unresolved: bool
    low_volume: bool
    props: tuple[PropView, ...]
    visible: bool = True
    visibility_reason: str = ""
    participant_order: float = float("inf")
    research: ResearchView | None = None
    decision_rows: tuple[tuple[object, ...], ...] = ()
    has_filtered_out_props: bool = False


@dataclass(frozen=True)
class TeamView:
    team: str
    home_away: str
    participants: tuple[ParticipantView, ...]
    hidden_backup_count: int = 0


@dataclass(frozen=True)
class DiagnosticView:
    category: str
    message: str


@dataclass(frozen=True)
class GameView:
    game_id: str
    kickoff: object
    away_team: TeamView | None
    home_team: TeamView | None
    diagnostics: tuple[DiagnosticView, ...]


def prepare_game_research(
    result: FootballResearchResult,
    odds_retrieved_at: object | None = None,
    game_odds: Mapping[str, Any] | None = None,
    line_filter: str = "all",
) -> tuple[GameView, ...]:
    """Create an immutable game model without changing pipeline-owned tables.

    Stable identifiers are required for prop association.  Blank player IDs stay
    visible but cannot receive a line, which prevents accidental name matching.
    """

    sources = _participant_sources(result)
    schedule = _schedule_rows(sources)
    props, decisions, diagnostics, all_props = _prop_context(result, odds_retrieved_at, game_odds, line_filter)
    participants = _participants(sources, props, decisions, result, all_props)
    games: list[GameView] = []
    for game_id, game_schedule in schedule.groupby("game_id", sort=False, dropna=False):
        game_key = _text(game_id) or "<missing game id>"
        game_people = [item for item in participants if item[0] == game_key]
        team_views: dict[str, TeamView] = {}
        for _, row in game_schedule.iterrows():
            team = _text(row["team"]) or "Unresolved team"
            home_away = _text(row["home_away"]) or "unresolved"
            values = [person for _, person_team, person in game_people if person_team == team]
            visible, hidden = _visible_team_participants(values)
            team_views[home_away] = TeamView(team, home_away, tuple(visible), hidden)
        game_diagnostics = list(diagnostics.get(game_key, ()))
        for position, source in (("QB", result.qb_result), ("RB", result.rb_result)):
            if game_odds is None and source is not None and source.player_prop_odds is None:
                game_diagnostics.append(
                    DiagnosticView(
                        "odds_unavailable",
                        f"{position} odds were not requested or are unavailable for this report.",
                    )
                )
            elif source is not None and getattr(source.player_prop_odds, "selected_events", pd.DataFrame()).empty:
                game_diagnostics.append(
                    DiagnosticView(
                        "no_sportsbook_event",
                        f"No matched sportsbook event is available for {position} props.",
                    )
                )
        for _, team, person in game_people:
            if person.unresolved:
                game_diagnostics.append(DiagnosticView("unresolved_player", f"{team}: unresolved {person.position} participant."))
        kickoff = game_schedule.iloc[0].get("kickoff")
        games.append(GameView(game_key, kickoff, team_views.get("away"), team_views.get("home"), tuple(_unique_diagnostics(game_diagnostics))))
    return tuple(sorted(games, key=lambda item: (_kickoff_key(item.kickoff), item.game_id)))


def _participant_sources(result: FootballResearchResult) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if result.qb_result is not None:
        qb = result.qb_result.summary.copy(deep=True)
        if not qb.empty:
            frames.append(pd.DataFrame({
                "game_id": qb.get("game_id"), "team": qb.get("team"), "opponent": qb.get("opponent"), "home_away": qb.get("home_away"),
                "game_date": qb.get("game_date"), "game_time": qb.get("game_time"), "position": "QB",
                "player_id": qb.get("expected_player_id"), "player_name": qb.get("expected_player_name"),
                "participant_order": 0, "unresolved": qb.get("expected_player_id").isna() | qb.get("expected_player_id").astype("string").str.strip().eq(""),
                "low_volume": False,
            }))
    if result.rb_result is not None:
        rb = result.rb_result.summary.copy(deep=True)
        if not rb.empty:
            frames.append(pd.DataFrame({
                "game_id": rb.get("game_id"), "team": rb.get("team"), "opponent": rb.get("opponent"), "home_away": rb.get("home_away"),
                "game_date": rb.get("game_date"), "game_time": rb.get("game_time"), "position": "RB",
                "player_id": rb.get("player_id"), "player_name": rb.get("player_name"), "participant_order": rb.get("participant_order"),
                "unresolved": rb.get("participant_resolution_missing", rb.get("resolution_missing", False)).fillna(False).astype(bool),
                "low_volume": rb.get("limited_rb_sample", False).fillna(False).astype(bool),
                "recent_share": rb.get("rb_last3_opportunity_share_avg", pd.NA),
                "season_share": rb.get("rb_season_opportunity_share_avg", pd.NA),
            }))
    if not frames:
        return pd.DataFrame(columns=["game_id", "team", "home_away", "game_date", "game_time", "position", "player_id", "player_name", "participant_order", "unresolved", "low_volume", "recent_share", "season_share"])
    data = pd.concat(frames, ignore_index=True)
    data["game_id"] = data["game_id"].map(_text)
    data["team"] = data["team"].map(_text)
    _reject_conflicting_participants(data)
    return data.drop_duplicates().copy(deep=True)


def _schedule_rows(sources: pd.DataFrame) -> pd.DataFrame:
    if sources.empty:
        return pd.DataFrame(columns=["game_id", "team", "home_away", "kickoff"])
    rows = sources.loc[:, ["game_id", "team", "home_away", "game_date", "game_time"]].drop_duplicates().copy()
    _reject_conflicting_schedule(rows)
    rows["kickoff"] = rows["game_date"].astype("string").fillna("") + " " + rows["game_time"].astype("string").fillna("")
    return rows.sort_values(["game_id", "home_away", "team"], kind="mergesort").reset_index(drop=True)


def _participants(sources: pd.DataFrame, props: dict[tuple[str, str, str], tuple[PropView, ...]], decisions: dict[tuple[str, str, str], tuple[tuple[object, ...], ...]], result: FootballResearchResult, all_props: dict[tuple[str, str, str], tuple[PropView, ...]]) -> list[tuple[str, str, ParticipantView]]:
    values: list[tuple[str, str, ParticipantView]] = []
    for _, row in sources.iterrows():
        game_id, team, position = _text(row["game_id"]) or "<missing game id>", _text(row["team"]) or "Unresolved team", row["position"]
        player_id = _text(row["player_id"])
        unresolved = bool(row["unresolved"]) or player_id is None
        order = _number(row["participant_order"])
        role = "unresolved participant" if unresolved else ("expected QB" if position == "QB" else "expected RB")
        research = None if unresolved else _research(row, position, result)
        prop_key = (game_id, player_id or "", position)
        values.append((game_id, team, ParticipantView(position, player_id, _text(row["player_name"]) or "Unresolved player", role, unresolved, bool(row["low_volume"]), props.get(prop_key, ()), True, _rb_reason(row, position), order, research, decisions.get(prop_key, ()), bool(all_props.get(prop_key)) and not bool(props.get(prop_key)))))
    return sorted(values, key=lambda item: (item[0], item[1], 0 if item[2].position == "QB" else 1, 0 if item[2].role == "leading expected RB" else 1, item[2].player_id or ""))


def _visible_team_participants(values: list[ParticipantView]) -> tuple[list[ParticipantView], int]:
    rbs = [value for value in values if value.position == "RB" and not value.unresolved]
    primary = min(rbs, key=lambda value: (value.participant_order, value.player_id or "")) if rbs else None
    shown, hidden = [], 0
    for value in values:
        visible = value.position != "RB" or value.unresolved or value is primary or bool(value.props) or value.visibility_reason in {"meaningful_recent_share", "meaningful_season_share_fallback"}
        if value is primary:
            value = ParticipantView(value.position, value.player_id, value.player_name, "leading expected RB", value.unresolved, value.low_volume, value.props, value.visible, value.visibility_reason, value.participant_order, value.research, value.decision_rows, value.has_filtered_out_props)
        elif value.position == "RB" and not value.unresolved:
            value = ParticipantView(value.position, value.player_id, value.player_name, "additional/back-up RB", value.unresolved, value.low_volume, value.props, value.visible, value.visibility_reason, value.participant_order, value.research, value.decision_rows, value.has_filtered_out_props)
        if visible: shown.append(value)
        else: hidden += 1
    return shown, hidden


def _rb_reason(row: pd.Series, position: str) -> str:
    if position != "RB": return ""
    recent = _share(row.get("recent_share"))
    if recent is not None: return "meaningful_recent_share" if recent >= .30 else ""
    season = _share(row.get("season_share"))
    return "meaningful_season_share_fallback" if season is not None and season >= .30 else ""


def _share(value: object) -> float | None:
    try:
        number = float(value)
        return number if pd.notna(number) and number >= 0 and number <= 1 else None
    except (TypeError, ValueError): return None


def _research(row: pd.Series, position: str, result: FootballResearchResult) -> ResearchView:
    context = pd.Series({"player_id": row.get("player_id"), "expected_player_id": row.get("player_id"), "opponent": row.get("opponent"), "game_id": row.get("game_id"), "team": row.get("team")})
    if position == "QB" and result.qb_result is not None:
        player = filter_qb_game_log(result.qb_result.qb_game_logs.copy(deep=True), context)
        defense = filter_qb_defense_game_log(result.qb_result.defense_game_logs.copy(deep=True), context)
        return _research_rows(player, defense, prepare_qb_log_display, prepare_qb_defense_log_display)
    if position == "RB" and result.rb_result is not None:
        player = filter_rb_game_log(result.rb_result.rb_game_logs.copy(deep=True), context)
        defense = filter_rb_defense_game_log(result.rb_result.defense_game_logs.copy(deep=True), context)
        return _research_rows(player, defense, prepare_rb_log_display, prepare_rb_defense_log_display)
    return ResearchView((), (), (), ())


def _research_rows(player: pd.DataFrame, defense: pd.DataFrame, player_formatter: Any, defense_formatter: Any) -> ResearchView:
    def ordered(data: pd.DataFrame) -> pd.DataFrame:
        columns = [column for column in ["season", "week", "game_id"] if column in data.columns]
        return data.sort_values(columns, ascending=False, kind="mergesort").head(8).copy(deep=True) if columns else data.head(8).copy(deep=True)
    player_display, defense_display = player_formatter(ordered(player)), defense_formatter(ordered(defense))
    return ResearchView(tuple(player_display.columns), tuple(map(tuple, player_display.itertuples(index=False, name=None))), tuple(defense_display.columns), tuple(map(tuple, defense_display.itertuples(index=False, name=None))))


def _prop_context(result: FootballResearchResult, retrieved_at: object | None, game_odds: Mapping[str, Any] | None, line_filter: str) -> tuple[dict[tuple[str, str, str], tuple[PropView, ...]], dict[tuple[str, str, str], tuple[tuple[object, ...], ...]], dict[str, list[DiagnosticView]], dict[tuple[str, str, str], tuple[PropView, ...]]]:
    props: dict[tuple[str, str, str], list[PropView]] = {}
    diagnostics: dict[str, list[DiagnosticView]] = {}
    decision_rows: dict[tuple[str, str, str], list[tuple[object, ...]]] = {}
    sources: list[tuple[str, Any, object | None]] = []
    if game_odds is None:
        sources = [(position, None if source is None else source.player_prop_odds, retrieved_at) for position, source in (("QB", result.qb_result), ("RB", result.rb_result))]
    else:
        sources = [("selected", snapshot, getattr(snapshot, "retrieved_at", retrieved_at)) for snapshot in game_odds.values()]
    for position, odds, source_retrieved_at in sources:
        if odds is None:
            continue
        odds_rows = odds.player_matched_odds.to_frame() if hasattr(odds.player_matched_odds, "to_frame") else odds.player_matched_odds.copy(deep=True)
        event_rows = odds.event_matches.to_frame() if hasattr(odds.event_matches, "to_frame") else odds.event_matches.copy(deep=True)
        for frame, status, label in ((event_rows, "event_match_status", "event"), (odds_rows, "match_status", "player")):
            for _, row in frame.copy(deep=True).iterrows():
                value = _text(row.get(status))
                if value in {"unmatched", "ambiguous"}:
                    game = _text(row.get("nflverse_game_id")) or "<unassigned>"
                    if label == "player":
                        details = f"{_text(row.get('player_name')) or 'Unknown player'} ({_text(row.get('market_key')) or 'unknown market'}) at {_text(row.get('bookmaker_title')) or _text(row.get('bookmaker_key')) or 'unknown sportsbook'}"
                        message = f"{value.title()} sportsbook player: {details}."
                    else:
                        message = f"{value.title()} sportsbook event: {_text(row.get('event_id')) or 'unknown event'}."
                    diagnostics.setdefault(game, []).append(DiagnosticView(f"{value}_{label}", message))
        for _, row in odds_rows.loc[(odds_rows["match_status"] == "matched") & (odds_rows["event_match_status"] == "matched")].iterrows():
            row_position = _text(row.get("expected_position")) or {"player_pass_yds": "QB", "player_rush_yds": "RB"}.get(row.get("market_key"), position)
            market = "player_pass_yds" if row_position == "QB" else "player_rush_yds"
            if row.get("market_key") != market:
                continue
            game, player = _text(row.get("nflverse_game_id")), _text(row.get("player_id"))
            if game and player:
                key = (game, player, row_position or position)
                props.setdefault(key, []).append(PropView(_text(row.get("bookmaker_title")) or _text(row.get("bookmaker_key")) or "Unknown sportsbook", row.get("point"), _text(row.get("outcome_name")) or "", row.get("price"), row.get("market_last_update"), source_retrieved_at))
                decision_rows.setdefault(key, []).append(tuple(row.get(column) for column in WEEKLY_PLAYER_PROP_ODDS_COLUMNS))
    frozen = {key: tuple(sorted(value, key=lambda prop: (prop.sportsbook, str(prop.line), prop.outcome))) for key, value in props.items()}
    all_props = frozen
    if line_filter == "balanced":
        allowed = _balanced_prop_keys(frozen)
        frozen = {key: tuple(prop for prop in value if (key, prop.sportsbook, prop.line) in allowed) for key, value in frozen.items()}
        frozen = {key: value for key, value in frozen.items() if value}
        decision_rows = {key: rows for key, rows in decision_rows.items() if key in frozen}
    for key, value in frozen.items():
        outcomes = {(prop.sportsbook, str(prop.line), prop.outcome.lower()) for prop in value}
        pairs = {(book, line) for book, line, _ in outcomes}
        for book, line in pairs:
            if not {"over", "under"}.issubset({outcome for current_book, current_line, outcome in outcomes if (current_book, current_line) == (book, line)}):
                diagnostics.setdefault(key[0], []).append(DiagnosticView("incomplete_over_under", f"{book} {line}: incomplete Over/Under pair for {key[2]} {key[1]}."))
    return frozen, {key: tuple(values) for key, values in decision_rows.items()}, diagnostics, all_props


def _balanced_prop_keys(props: Mapping[tuple[str, str, str], tuple[PropView, ...]]) -> set[tuple[tuple[str, str, str], str, object]]:
    allowed: set[tuple[tuple[str, str, str], str, object]] = set()
    for player_key, values in props.items():
        groups: dict[tuple[str, object], dict[str, object]] = {}
        for value in values:
            groups.setdefault((value.sportsbook, value.line), {})[value.outcome] = value.price
        for (sportsbook, line), prices in groups.items():
            if {"Over", "Under"}.issubset(prices) and all(_balanced_price(prices[side]) for side in ("Over", "Under")):
                allowed.add((player_key, sportsbook, line))
    return allowed


def _balanced_price(value: object) -> bool:
    try:
        number = float(value)
        return pd.notna(number) and -130 <= number <= 130
    except (TypeError, ValueError):
        return False


def _reject_conflicting_schedule(rows: pd.DataFrame) -> None:
    for key, group in rows.groupby(["game_id", "team"], dropna=False):
        if len(group.drop_duplicates()) > 1:
            raise ValueError(f"Conflicting schedule rows for game/team identity: {key}")


def _reject_conflicting_participants(rows: pd.DataFrame) -> None:
    identity = rows.assign(_identity=rows.apply(lambda row: (row["game_id"], row["team"], row["position"], _text(row["player_id"]) or f"unresolved:{row['participant_order']}"), axis=1))
    for key, group in identity.groupby("_identity", dropna=False):
        if len(group.drop(columns="_identity").drop_duplicates()) > 1:
            raise ValueError(f"Conflicting participant rows for stable identity: {key}")


def _text(value: object) -> str | None:
    return None if value is None or pd.isna(value) or not str(value).strip() else str(value).strip()


def _number(value: object) -> float:
    try: return float(value)
    except (TypeError, ValueError): return float("inf")


def _kickoff_key(value: object) -> tuple[int, str]:
    text = _text(value)
    return (0, text) if text else (1, "")


def _unique_diagnostics(values: list[DiagnosticView]) -> list[DiagnosticView]:
    return list(dict.fromkeys(values))
