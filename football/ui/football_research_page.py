"""Combined game-centered Football Research Streamlit shell."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Callable
import re
from zoneinfo import ZoneInfo

import streamlit as st
import pandas as pd

from football.ui.qb_research_page import load_qb_research
from football.ui.rb_research_page import load_rb_research
from football.ui.qb_research_view import default_history_season
from football.ui.football_research_view import FootballResearchResult, prepare_game_research
from football.pipeline import build_selected_game_player_prop_odds
from football.odds import DEFAULT_FOOTBALL_PROP_BOOKMAKERS
from football.data.build_schedule_dataset import OUTPUT_COLUMNS as SCHEDULE_COLUMNS
from football.pipeline.build_weekly_player_prop_odds import WEEKLY_PLAYER_PROP_ODDS_COLUMNS
from football.ui.qb_research_view import build_qb_decision_outcome_options, build_qb_pending_decision
from football.ui.rb_research_view import build_rb_decision_outcome_options, build_rb_pending_decision
from football.decision_database import open_local_decision_database
from football.decisions import record_decision, DuplicateDecisionError, DecisionStoreError, DecisionValidationError


def default_report_inputs(current_date: date) -> dict[str, object]:
    """Return deterministic regular-season defaults; opener is first September Thursday."""
    season = current_date.year if current_date.month >= 2 else current_date.year - 1
    first_september_thursday = date(season, 9, 1)
    first_september_thursday = first_september_thursday.replace(day=1 + ((3 - first_september_thursday.weekday()) % 7))
    # NFL Week 1 is the Thursday after Labor Day, i.e. the second September Thursday.
    opener = first_september_thursday.replace(day=first_september_thursday.day + 7)
    if current_date < opener:
        week = 1
    elif current_date > date(season, 1, 10) and current_date.month == 1:
        week = 18
    else:
        week = min(18, ((current_date - opener).days // 7) + 1)
    return {"report_season": season, "report_week": week, "as_of_date": current_date, "history_season": default_history_season(season, week)}


def render_football_research_page(*, streamlit_module: Any | None = None, qb_report_loader: Callable[..., Any] = load_qb_research, rb_report_loader: Callable[..., Any] = load_rb_research, selected_game_odds_loader: Callable[..., Any] = build_selected_game_player_prop_odds, clock: Callable[[], object] | None = None, database_opener: Callable = open_local_decision_database, decision_recorder: Callable = record_decision) -> None:
    """Render only the deliberately submitted combined report; no selector reloads."""
    ui, now = streamlit_module or st, clock or (lambda: datetime.now(timezone.utc))
    ui.title("Football Research")
    ui.caption("Research context only — not a prediction or recommendation.")
    today = now().date() if isinstance(now(), datetime) else date.today()
    defaults = ui.session_state.setdefault("football_research_default_inputs", default_report_inputs(today))
    has_report = "football_research_result" in ui.session_state
    with ui.expander("Report settings", expanded=not has_report):
        with ui.form("football_research_form"):
            columns = ui.columns([1, 1, 1.3, 1, 1.3])
            with columns[0]: season = ui.number_input("Report season", min_value=1999, max_value=today.year + 1, value=defaults["report_season"], step=1)
            with columns[1]: week = ui.number_input("Report week", min_value=1, max_value=22, value=defaults["report_week"], step=1)
            with columns[2]: as_of = ui.date_input("As-of date", value=defaults["as_of_date"])
            with columns[3]: history = ui.number_input("History season", min_value=1999, max_value=int(season), value=defaults["history_season"], step=1)
            label = "Refresh Football Research" if has_report else "Generate Football Research"
            with columns[4]: ui.write(""); submitted = ui.form_submit_button(label, type="primary")
    if submitted:
        inputs = {"report_season": int(season), "report_week": int(week), "as_of_date": as_of, "history_season": int(history)}
        try:
            with ui.spinner("Loading QB and RB research..."):
                combined = FootballResearchResult(qb_report_loader(**inputs, include_player_props=False), rb_report_loader(**inputs, include_player_props=False))
            previous = ui.session_state.get("football_research_game_odds", {})
            previous_contexts = ui.session_state.get("football_research_game_odds_contexts", {})
            ui.session_state["football_research_inputs"] = inputs
            ui.session_state["football_research_result"] = combined
            retained, contexts = _compatible_game_odds(previous, previous_contexts, combined, inputs)
            ui.session_state["football_research_game_odds"] = retained
            ui.session_state["football_research_game_odds_contexts"] = contexts
        except Exception as error:  # Existing loaders preserve their specific public errors.
            ui.error(f"Could not refresh football research: {error}")
    inputs = ui.session_state.get("football_research_inputs")
    combined = ui.session_state.get("football_research_result")
    if inputs is None or combined is None:
        ui.info("No live data loads until Generate Football Report is clicked.")
        ui.caption("Research data: nflverse via nflreadpy. Sportsbook data: The Odds API.")
        return
    snapshots = ui.session_state.setdefault("football_research_game_odds", {})
    line_mode_key = "football_research_line_filter"
    line_mode = ui.session_state.get(line_mode_key, "Balanced lines")
    games = prepare_game_research(combined, None, snapshots, "balanced" if line_mode == "Balanced lines" else "all")
    ui.caption(f"{inputs['report_season']} • Week {inputs['report_week']} • History {inputs['history_season']}")
    if not games:
        ui.info("No scheduled games were found for these report inputs.")
    game_by_id = {game.game_id: game for game in games}
    selected_key = "football_research_selected_game_id"
    selected_id = ui.session_state.get(selected_key)
    if selected_id not in game_by_id:
        selected_id = games[0].game_id if games else None
        ui.session_state[selected_key] = selected_id
    if games:
        labels = {game.game_id: _game_label(game) for game in games}
        if hasattr(ui, "pills"):
            selected_id = ui.pills("Selected game", list(game_by_id), format_func=lambda value: labels[value], selection_mode="single", default=selected_id, key=selected_key)
        else:
            selected_id = ui.selectbox("Selected game", list(game_by_id), format_func=lambda value: labels[value], key=selected_key)
    for game in games:
        if game.game_id != selected_id:
            continue
        away_name = game.away_team.team if game.away_team else "Unresolved away"
        home_name = game.home_team.team if game.home_team else "Unresolved home"
        with ui.expander(f"{away_name} at {home_name}"):
            kickoff = format_kickoff_central(game.kickoff)
            ui.markdown(f"### {away_name} at {home_name}" + (f" — {kickoff}" if kickoff else ""))
            ui.caption(f"Game ID: {game.game_id}")
            snapshot = snapshots.get(game.game_id)
            _render_game_odds_controls(ui, game, snapshot, combined, inputs, selected_game_odds_loader, snapshots)
            snapshots = ui.session_state.get("football_research_game_odds", {})
            snapshot = snapshots.get(game.game_id)
            if snapshot is not None:
                line_mode = ui.selectbox("Displayed lines", ["Balanced lines", "All available lines"], key=line_mode_key)
                games = prepare_game_research(combined, None, snapshots, "balanced" if line_mode == "Balanced lines" else "all")
                game = next(value for value in games if value.game_id == game.game_id)
            for team in (game.away_team, game.home_team):
                if team is None: continue
                with ui.container(border=True):
                    ui.markdown(f"**{team.home_away.title()}: {team.team}**")
                    if not team.participants: ui.info("No expected participants were supplied for this scheduled team.")
                    for person in team.participants:
                        ui.markdown(f"**{person.player_name}** — {person.role} ({person.position})")
                        if person.low_volume: ui.caption("Historical context: low-volume sample.")
                        if person.visibility_reason == "meaningful_recent_share": ui.caption("Shared backfield: meaningful recent role.")
                        if person.visibility_reason == "meaningful_season_share_fallback": ui.caption("Shared backfield: meaningful season role.")
                        if person.unresolved: ui.info("Participant is unresolved; no sportsbook line can be associated.")
                        elif not person.props:
                            ui.caption("No balanced lines from the selected sportsbooks. View all available lines." if person.has_filtered_out_props else "No matched current sportsbook line.")
                        else: ui.dataframe(_line_display(person.props), use_container_width=True, hide_index=True)
                        if person.decision_rows:
                            _render_decision(ui, person, game, inputs, ui.session_state.get("football_research_odds_retrieved_at"), database_opener, decision_recorder, now)
                        if person.research is not None:
                            with ui.expander("View matchup research"):
                                _render_research(ui, person)
                    if team.hidden_backup_count: ui.caption(f"{team.hidden_backup_count} low-volume backup{'s' if team.hidden_backup_count != 1 else ''} hidden.")
            if game.diagnostics:
                with ui.expander("Data diagnostics"):
                    for diagnostic in game.diagnostics: ui.caption(diagnostic.message)
    ui.caption("Research data: nflverse via nflreadpy. Sportsbook data: The Odds API.")


def _render_game_odds_controls(ui: Any, game: Any, snapshot: Any, combined: FootballResearchResult, inputs: dict, loader: Callable[..., Any], snapshots: dict) -> None:
    """Render the only two page actions permitted to invoke the paid loader."""
    if snapshot is None:
        ui.caption("Sportsbook lines are not loaded. Loading makes one paid selected-game odds request.")
        action_label = "Load lines for this game"
    else:
        ui.caption(f"Current sportsbook lines retrieved: {snapshot.retrieved_at}")
        ui.caption("Sportsbooks: DraftKings, FanDuel, BetRivers.")
        quota = snapshot.quota
        values = []
        if quota.requests_last is not None: values.append(f"last request: {quota.requests_last} credits")
        if quota.requests_remaining is not None: values.append(f"remaining: {quota.requests_remaining}")
        if quota.requests_used is not None: values.append(f"total used: {quota.requests_used}")
        if values: ui.caption("Quota — " + "; ".join(values))
        action_label = "Refresh lines for this game"
    clicked = ui.button(action_label, key=f"football_research_game_odds|{game.game_id}") if hasattr(ui, "button") else False
    if not clicked:
        return
    try:
        schedule, qbs, rbs = _selected_game_inputs(combined, game.game_id, inputs)
        result = loader(game.game_id, schedule, qbs, rbs, inputs["report_season"], inputs["report_week"])
        updated = dict(snapshots)
        updated[game.game_id] = result
        ui.session_state["football_research_game_odds"] = updated
        ui.session_state.setdefault("football_research_game_odds_contexts", {})[game.game_id] = _game_context(combined, inputs).get(game.game_id)
    except Exception as error:
        if snapshot is None:
            ui.error(f"Could not load lines for this game: {_safe_error(error)}")
        else:
            ui.error(f"Could not refresh lines for this game; previously retrieved lines remain displayed: {_safe_error(error)}")


def _selected_game_inputs(combined: FootballResearchResult, game_id: str, inputs: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = []
    for source, position in ((combined.qb_result, "QB"), (combined.rb_result, "RB")):
        if source is None or source.summary.empty: continue
        summary = source.summary.loc[source.summary["game_id"].eq(game_id)].copy(deep=True)
        if summary.empty: continue
        frame = pd.DataFrame({"season": inputs["report_season"], "week": inputs["report_week"], "game_id": summary["game_id"], "game_date": summary["game_date"], "game_time": summary["game_time"], "team": summary["team"], "opponent": summary["opponent"], "home_away": summary["home_away"], "home_score": pd.NA, "away_score": pd.NA})
        frames.append(frame)
    schedule = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["game_id", "team"]).copy()
    # Pipeline summaries may omit report fields; the selected loader receives
    # canonical report inputs separately, so fill them at the caller boundary.
    qbs = combined.qb_result.summary.loc[combined.qb_result.summary["game_id"].eq(game_id)].copy(deep=True) if combined.qb_result is not None and "game_id" in combined.qb_result.summary else pd.DataFrame(columns=["game_id", "team", "expected_player_id", "expected_player_name"])
    rbs = combined.rb_result.summary.loc[combined.rb_result.summary["game_id"].eq(game_id)].copy(deep=True) if combined.rb_result is not None and "game_id" in combined.rb_result.summary else pd.DataFrame(columns=["game_id", "team", "player_id", "player_name"])
    return schedule.loc[:, SCHEDULE_COLUMNS], qbs, rbs


def _game_context(result: FootballResearchResult, inputs: dict) -> dict[str, tuple]:
    games = prepare_game_research(result, None, {})
    participants: dict[str, list[tuple[str, str]]] = {}
    if result.qb_result is not None:
        for _, row in result.qb_result.summary.copy(deep=True).iterrows():
            if pd.notna(row.get("expected_player_id")):
                participants.setdefault(str(row.get("game_id")), []).append((str(row.get("expected_player_id")).strip(), "QB"))
    if result.rb_result is not None:
        for _, row in result.rb_result.summary.copy(deep=True).iterrows():
            if pd.notna(row.get("player_id")):
                participants.setdefault(str(row.get("game_id")), []).append((str(row.get("player_id")).strip(), "RB"))
    return {game.game_id: (inputs["report_season"], inputs["report_week"], game.away_team.team if game.away_team else None, game.home_team.team if game.home_team else None, tuple(sorted(set(participants.get(game.game_id, [])))), tuple(DEFAULT_FOOTBALL_PROP_BOOKMAKERS)) for game in games}


def _compatible_game_odds(previous: dict, previous_contexts: dict, result: FootballResearchResult, inputs: dict) -> tuple[dict, dict]:
    contexts = _game_context(result, inputs)
    retained = {game_id: snapshot for game_id, snapshot in previous.items() if game_id in contexts and previous_contexts.get(game_id) == contexts[game_id] and getattr(snapshot, "markets", ()) == ("player_pass_yds", "player_rush_yds") and getattr(snapshot, "bookmakers", ()) == tuple(DEFAULT_FOOTBALL_PROP_BOOKMAKERS)}
    return retained, {game_id: contexts[game_id] for game_id in retained}


def _safe_error(error: Exception) -> str:
    message = str(error)
    message = re.sub(r"https?://\S+", "[provider URL redacted]", message)
    return re.sub(r"(?i)(api[_-]?key=)[^\s&]+", r"\1[redacted]", message)


def _game_label(game: Any) -> str:
    away = game.away_team.team if game.away_team else "Unresolved away"
    home = game.home_team.team if game.home_team else "Unresolved home"
    kickoff = format_kickoff_central(game.kickoff)
    return f"{away} at {home}" + (f" — {kickoff}" if kickoff else "")


def format_kickoff_central(value: object) -> str | None:
    """Return a DST-aware twelve-hour Central kickoff label, or ``None``.

    Normalized schedule date/time values are New York wall-clock values. Aware
    provider timestamps are converted directly to America/Chicago.
    """
    if value is None or value is pd.NA:
        return None
    try:
        kickoff = pd.Timestamp(value)
        if pd.isna(kickoff):
            return None
        if kickoff.tzinfo is None:
            kickoff = kickoff.tz_localize(ZoneInfo("America/New_York"), ambiguous="raise", nonexistent="raise")
        central = kickoff.tz_convert(ZoneInfo("America/Chicago"))
    except (TypeError, ValueError, OverflowError):
        return None
    hour = central.hour % 12 or 12
    return f"{central.strftime('%a')} {hour}:{central.strftime('%M %p')} CT"


def _line_display(props: tuple[Any, ...]) -> pd.DataFrame:
    """Pivot matched side-level quotes without collapsing alternate lines."""
    rows = []
    for sportsbook, line in sorted({(prop.sportsbook, prop.line) for prop in props}, key=lambda value: (value[0], value[1])):
        group = [prop for prop in props if (prop.sportsbook, prop.line) == (sportsbook, line)]
        prices = {prop.outcome: prop.price for prop in group}
        rows.append({"Sportsbook": sportsbook, "Line": line, "Over": prices.get("Over", pd.NA), "Under": prices.get("Under", pd.NA), "Updated": group[0].market_updated_at})
    return pd.DataFrame(rows)


def _render_research(ui: Any, person: Any) -> None:
    research = person.research
    ui.markdown(f"**{person.player_name} — Recent games**")
    if research.player_rows:
        ui.dataframe(pd.DataFrame(research.player_rows, columns=research.player_columns), use_container_width=True, hide_index=True)
    else: ui.caption("No prior player games are available.")
    ui.markdown(f"**Scheduled opponent — Results allowed to {person.position}s**")
    if research.defense_rows:
        ui.dataframe(pd.DataFrame(research.defense_rows, columns=research.defense_columns), use_container_width=True, hide_index=True)
    else: ui.caption("No opponent defensive games are available before the cutoff.")


def _render_decision(ui: Any, person: Any, game: Any, inputs: dict, retrieved_at: object, database_opener: Callable, decision_recorder: Callable, clock: Callable) -> None:
    props = pd.DataFrame(person.decision_rows, columns=WEEKLY_PLAYER_PROP_ODDS_COLUMNS)
    context = pd.Series({"expected_player_id": person.player_id, "player_id": person.player_id, "team": game.away_team.team if game.away_team and any(p is person for p in game.away_team.participants) else game.home_team.team, "opponent": game.home_team.team if game.away_team and any(p is person for p in game.away_team.participants) else game.away_team.team, "game_id": game.game_id, "season": inputs["report_season"], "report_season": inputs["report_season"], "report_week": inputs["report_week"]})
    builder = build_qb_decision_outcome_options if person.position == "QB" else build_rb_decision_outcome_options
    pending = build_qb_pending_decision if person.position == "QB" else build_rb_pending_decision
    options = builder(props)
    if not options: return
    key = f"football_research_decision|{game.game_id}|{person.player_id}|{person.position}"
    with ui.expander("Record a decision"):
        with ui.form(key + "|form"):
            option_id = ui.selectbox("Exact sportsbook outcome", [option.option_id for option in options], format_func=lambda value: next(option.label for option in options if option.option_id == value), key=key + "|outcome")
            submitted = ui.form_submit_button("Record decision")
        if submitted:
            try:
                decision = pending(props, context, option_id, retrieved_at, clock())
                connection = database_opener()
                try: decision_recorder(connection, decision)
                finally: connection.close()
            except DuplicateDecisionError: ui.info("This exact sportsbook snapshot was already recorded.")
            except (DecisionValidationError, DecisionStoreError, OSError) as error: ui.error(f"Could not record decision: {error}")
            else: ui.success("Decision recorded.")
