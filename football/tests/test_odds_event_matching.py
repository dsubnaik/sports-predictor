"""Contract tests for deterministic Odds API event-to-schedule matching."""

import pandas as pd
import pytest

from football.data.build_schedule_dataset import OUTPUT_COLUMNS as SCHEDULE_COLUMNS
from football.odds.event_matching import (
    EVENT_MATCH_OUTPUT_COLUMNS,
    ODDS_API_TEAM_TO_NFLVERSE,
    match_odds_events_to_schedule,
)
from football.odds.player_props import PLAYER_PROP_ODDS_COLUMNS


def odds(rows):
    base = {"event_id": "event-1", "commence_time": "2026-09-10T00:20:00Z", "home_team": "Seattle Seahawks", "away_team": "New England Patriots", "bookmaker_key": "book", "bookmaker_title": "Book", "bookmaker_last_update": pd.NA, "market_key": "player_pass_yds", "market_last_update": "2026-09-09T00:00:00Z", "player_name": "Player", "outcome_name": "Over", "price": -110, "point": 200.5}
    return pd.DataFrame([{**base, **row} for row in rows], columns=PLAYER_PROP_ODDS_COLUMNS)


def schedule(rows=None):
    rows = rows or [
        (2026, 1, "game-1", "2026-09-09", "20:20", "SEA", "NE", "home"),
        (2026, 1, "game-1", "2026-09-09", "20:20", "NE", "SEA", "away"),
    ]
    return pd.DataFrame(rows, columns=SCHEDULE_COLUMNS)


def test_all_current_odds_api_teams_use_expected_nflverse_abbreviations():
    assert ODDS_API_TEAM_TO_NFLVERSE == {
        "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB", "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG", "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS"}


def test_matches_new_england_at_seattle_and_repeats_metadata_for_all_odds_rows():
    source = odds([{"bookmaker_key": book, "outcome_name": side, "market_key": market} for book in ["a", "b"] for side in ["Over", "Under"] for market in ["player_pass_yds", "player_rush_yds"]])
    result = match_odds_events_to_schedule(source, schedule())
    assert result.columns.tolist() == EVENT_MATCH_OUTPUT_COLUMNS
    assert result["nflverse_game_id"].tolist() == ["game-1"] * len(result)
    assert result["event_match_status"].tolist() == ["matched"] * len(result)
    assert result["nflverse_kickoff_time"].iloc[0] == pd.Timestamp("2026-09-10T00:20:00Z")
    assert result["nflverse_home_team"].iloc[0] == "SEA"
    assert result["nflverse_away_team"].iloc[0] == "NE"


def test_los_angeles_rams_match_nflverse_la_schedule_abbreviation():
    source = odds([{"home_team": "Los Angeles Rams", "away_team": "Seattle Seahawks"}])
    game = schedule([
        (2026, 1, "rams-game", "2026-09-09", "20:20", "LA", "SEA", "home"),
        (2026, 1, "rams-game", "2026-09-09", "20:20", "SEA", "LA", "away"),
    ])
    result = match_odds_events_to_schedule(source, game)
    assert result.loc[0, ["nflverse_game_id", "nflverse_home_team"]].tolist() == ["rams-game", "LA"]


@pytest.mark.parametrize(("date", "time", "commence"), [("2026-09-09", "20:20", "2026-09-10T00:20:00Z"), ("2026-12-06", "13:00", "2026-12-06T18:00:00Z")])
def test_eastern_daylight_and_standard_kickoffs_convert_to_utc(date, time, commence):
    game = schedule([(2026, 1, "game-1", date, time, "SEA", "NE", "home"), (2026, 1, "game-1", date, time, "NE", "SEA", "away")])
    assert match_odds_events_to_schedule(odds([{"commence_time": commence}]), game).loc[0, "event_match_status"] == "matched"


def test_unmatched_unknown_reversed_and_wrong_kickoff_are_explicit():
    unknown = match_odds_events_to_schedule(odds([{"home_team": "Unknown Team"}]), schedule())
    reversed_ = match_odds_events_to_schedule(odds([{"home_team": "New England Patriots", "away_team": "Seattle Seahawks"}]), schedule())
    wrong_time = match_odds_events_to_schedule(odds([{"commence_time": "2026-09-10T00:21:00Z"}]), schedule())
    assert unknown.loc[0, "event_match_note"] == "Unknown Odds API team name"
    assert reversed_.loc[0, "event_match_status"] == "unmatched"
    assert wrong_time.loc[0, "event_match_status"] == "unmatched"


def test_ambiguity_and_exact_duplicate_schedule_rows_are_handled_deterministically():
    duplicate = pd.concat([schedule(), schedule()], ignore_index=True)
    assert match_odds_events_to_schedule(odds([{}]), duplicate).loc[0, "event_match_status"] == "matched"
    ambiguous = pd.concat([schedule(), schedule([(2026, 2, "game-2", "2026-09-09", "20:20", "SEA", "NE", "home"), (2026, 2, "game-2", "2026-09-09", "20:20", "NE", "SEA", "away")])], ignore_index=True)
    result = match_odds_events_to_schedule(odds([{}]), ambiguous)
    assert result.loc[0, ["event_match_status", "event_match_candidate_count"]].tolist() == ["ambiguous", 2]
    assert pd.isna(result.loc[0, "nflverse_game_id"])


@pytest.mark.parametrize("bad_rows", [
    [(2026, 1, "game-1", "2026-09-09", "20:20", "SEA", "NE", "home")],
    [(2026, 1, "game-1", "2026-09-09", "20:20", "SEA", "NE", "home"), (2026, 1, "game-1", "2026-09-09", "20:20", "NE", "BUF", "away")],
    [(True, 1, "game-1", "2026-09-09", "20:20", "SEA", "NE", "home"), (True, 1, "game-1", "2026-09-09", "20:20", "NE", "SEA", "away")],
    [(2026, 1, "", "2026-09-09", "20:20", "SEA", "NE", "home"), (2026, 1, "", "2026-09-09", "20:20", "NE", "SEA", "away")],
    [(2026, 1, "game-1", "bad", "20:20", "SEA", "NE", "home"), (2026, 1, "game-1", "bad", "20:20", "NE", "SEA", "away")],
    [(2026, 1, "game-1", "2026-09-09", "bad", "SEA", "NE", "home"), (2026, 1, "game-1", "2026-09-09", "bad", "NE", "SEA", "away")],
])
def test_malformed_schedule_games_raise(bad_rows):
    with pytest.raises(ValueError): match_odds_events_to_schedule(odds([{}]), schedule(bad_rows))


def test_odds_validation_empty_schedule_and_immutability():
    with pytest.raises(ValueError, match="timezone-aware"):
        match_odds_events_to_schedule(odds([{"commence_time": "2026-09-09T20:20:00"}]), schedule())
    with pytest.raises(ValueError, match="Conflicting odds"):
        match_odds_events_to_schedule(
            odds([{}, {"home_team": "New England Patriots", "away_team": "Seattle Seahawks"}]),
            schedule(),
        )
    empty = match_odds_events_to_schedule(pd.DataFrame(columns=PLAYER_PROP_ODDS_COLUMNS), schedule())
    assert empty.columns.tolist() == EVENT_MATCH_OUTPUT_COLUMNS
    unmatched = match_odds_events_to_schedule(odds([{}]), pd.DataFrame(columns=SCHEDULE_COLUMNS))
    assert unmatched.loc[0, "event_match_status"] == "unmatched"
    source, games = odds([{}]), schedule()
    source_before, games_before = source.copy(deep=True), games.copy(deep=True)
    match_odds_events_to_schedule(source, games)
    pd.testing.assert_frame_equal(source, source_before)
    pd.testing.assert_frame_equal(games, games_before)
