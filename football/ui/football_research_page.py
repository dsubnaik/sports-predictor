"""Combined game-centered Football Research Streamlit shell."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Callable

import streamlit as st
import pandas as pd

from football.ui.qb_research_page import load_qb_research
from football.ui.rb_research_page import load_rb_research
from football.ui.qb_research_view import default_history_season
from football.ui.football_research_view import FootballResearchResult, prepare_game_research
from football.pipeline.build_weekly_player_prop_odds import WEEKLY_PLAYER_PROP_ODDS_COLUMNS
from football.ui.qb_research_view import build_qb_decision_outcome_options, build_qb_pending_decision
from football.ui.rb_research_view import build_rb_decision_outcome_options, build_rb_pending_decision
from football.decision_database import open_local_decision_database
from football.decisions import record_decision, DuplicateDecisionError, DecisionStoreError, DecisionValidationError


def render_football_research_page(*, streamlit_module: Any | None = None, qb_report_loader: Callable[..., Any] = load_qb_research, rb_report_loader: Callable[..., Any] = load_rb_research, clock: Callable[[], object] | None = None, database_opener: Callable = open_local_decision_database, decision_recorder: Callable = record_decision) -> None:
    """Render only the deliberately submitted combined report; no selector reloads."""
    ui, now = streamlit_module or st, clock or (lambda: datetime.now(timezone.utc))
    ui.title("Football Research")
    ui.info("Game-centered research context, not a prediction or recommendation.")
    today = date.today()
    with ui.form("football_research_form"):
        columns = ui.columns([1, 1, 1.3, 1, 1.3])
        with columns[0]: season = ui.number_input("Report season", min_value=1999, max_value=today.year + 1, value=today.year, step=1)
        with columns[1]: week = ui.number_input("Report week", min_value=1, max_value=22, value=1, step=1)
        with columns[2]: as_of = ui.date_input("As-of date", value=today)
        with columns[3]: history = ui.number_input("History season", min_value=1999, max_value=int(season), value=default_history_season(int(season), int(week)), step=1)
        with columns[4]: ui.write(""); submitted = ui.form_submit_button("Generate Football Report", type="primary")
    if submitted:
        inputs = {"report_season": int(season), "report_week": int(week), "as_of_date": as_of, "history_season": int(history)}
        try:
            with ui.spinner("Loading QB and RB research with current sportsbook data..."):
                combined = FootballResearchResult(qb_report_loader(**inputs), rb_report_loader(**inputs))
            ui.session_state["football_research_inputs"] = inputs
            ui.session_state["football_research_result"] = combined
            ui.session_state["football_research_odds_retrieved_at"] = now()
        except Exception as error:  # Existing loaders preserve their specific public errors.
            ui.error(f"Could not refresh football research: {error}")
    inputs = ui.session_state.get("football_research_inputs")
    combined = ui.session_state.get("football_research_result")
    if inputs is None or combined is None:
        ui.info("No live data loads until Generate Football Report is clicked.")
        ui.caption("Research data: nflverse via nflreadpy. Sportsbook data: The Odds API.")
        return
    games = prepare_game_research(combined, ui.session_state.get("football_research_odds_retrieved_at"))
    ui.caption(f"Report season {inputs['report_season']}, Week {inputs['report_week']}; historical season {inputs['history_season']}.")
    if not games:
        ui.info("No scheduled games were found for these report inputs.")
    for game in games:
        away_name = game.away_team.team if game.away_team else "Unresolved away"
        home_name = game.home_team.team if game.home_team else "Unresolved home"
        with ui.expander(f"{away_name} at {home_name}"):
            ui.markdown(f"### {away_name} at {home_name}")
            ui.caption(f"Game ID: {game.game_id}")
            ui.caption(f"Current sportsbook lines retrieved: {ui.session_state.get('football_research_odds_retrieved_at')}")
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
                        elif not person.props: ui.caption("No matched current sportsbook line.")
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
