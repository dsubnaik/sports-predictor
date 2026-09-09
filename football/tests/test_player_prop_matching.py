"""Contract tests for deterministic nflverse player-prop matching."""

import pandas as pd
import pytest

from football.odds.player_matching import (
    PLAYER_PROP_MATCH_COLUMNS,
    PLAYER_PROP_MATCH_OUTPUT_COLUMNS,
    match_player_prop_odds,
)
from football.odds.player_props import PLAYER_PROP_ODDS_COLUMNS


def odds_rows(rows):
    defaults = {
        "event_id": "event", "commence_time": "2026-09-10T00:00:00Z",
        "home_team": "SEA", "away_team": "NE", "bookmaker_key": "book",
        "bookmaker_title": "Book", "bookmaker_last_update": pd.NA,
        "market_last_update": "2026-09-09T00:00:00Z", "outcome_name": "Over",
        "price": -110, "point": 100.5,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows], columns=PLAYER_PROP_ODDS_COLUMNS)


def players(rows):
    return pd.DataFrame(rows, columns=["player_id", "player_name", "team", "position"])


def test_matches_qb_and_rb_and_preserves_odds_columns_first():
    odds = odds_rows([
        {"market_key": "player_pass_yds", "player_name": "Drake Maye"},
        {"market_key": "player_rush_yds", "player_name": "Rhamondre Stevenson"},
    ])
    result = match_player_prop_odds(odds, players([
        ("qb-1", "Drake Maye", "NE", "QB"),
        ("rb-1", "Rhamondre Stevenson", "NE", "RB"),
    ]))
    assert result.columns.tolist() == PLAYER_PROP_MATCH_OUTPUT_COLUMNS
    assert result["player_id"].tolist() == ["qb-1", "rb-1"]
    assert result["expected_position"].tolist() == ["QB", "RB"]
    assert result["match_status"].tolist() == ["matched", "matched"]
    assert result["match_method"].tolist() == ["canonical_name_and_position"] * 2
    assert result[PLAYER_PROP_MATCH_COLUMNS].columns.tolist() == PLAYER_PROP_MATCH_COLUMNS


def test_same_player_bookmakers_and_outcomes_resolve_consistently():
    odds = odds_rows([
        {"market_key": "player_pass_yds", "player_name": "Drake Maye", "outcome_name": side, "bookmaker_key": book}
        for book in ["a", "b"] for side in ["Over", "Under"]
    ])
    result = match_player_prop_odds(odds, players([("qb-1", "Drake Maye", "NE", "QB")]))
    assert result["player_id"].tolist() == ["qb-1"] * 4
    assert result["match_candidate_count"].tolist() == [1] * 4


@pytest.mark.parametrize(("sportsbook", "reference"), [
    ("D\u2019Andre Swift", "D'Andre Swift"),
    ("  D. J.  Moore ", "DJ Moore"), ("D’Andre Swift", "D'Andre Swift"),
    ("Amon Ra St Brown", "Amon-Ra St. Brown"), ("John Doe Jr.", "John Doe, Jr"),
    ("John Doe IV", "John Doe I.V."),
])
def test_harmless_name_formatting_matches_without_changing_display_names(sportsbook, reference):
    result = match_player_prop_odds(
        odds_rows([{"market_key": "player_rush_yds", "player_name": sportsbook}]),
        players([("rb-1", reference, "CHI", "RB")]),
    )
    assert result.loc[0, "player_id"] == "rb-1"
    assert result.loc[0, "player_name"] == sportsbook
    assert result.loc[0, "nflverse_player_name"] == reference


def test_position_filtering_unmatched_and_ambiguous_results_are_explicit():
    odds = odds_rows([
        {"market_key": "player_pass_yds", "player_name": "Same Name"},
        {"market_key": "player_rush_yds", "player_name": "Missing Name"},
        {"market_key": "player_rush_yds", "player_name": "Twin Runner"},
    ])
    result = match_player_prop_odds(odds, players([
        ("rb-same", "Same Name", "NE", "RB"),
        ("rb-1", "Twin Runner", "NE", "RB"),
        ("rb-2", "Twin Runner", "SEA", "RB"),
    ]))
    assert result["match_status"].tolist() == ["unmatched", "unmatched", "ambiguous"]
    assert result["match_candidate_count"].tolist() == [0, 0, 2]
    assert result.loc[2, "match_note"] == "Multiple canonical name-and-position candidates"
    assert result.loc[:, ["player_id", "nflverse_player_name", "nflverse_team"]].isna().all().all()


def test_repeated_snapshots_and_exact_duplicates_do_not_create_ambiguity():
    reference = players([
        ("rb-1", "Runner", "NE", "RB"), ("rb-1", "Runner", "NE", "RB"),
    ])
    result = match_player_prop_odds(odds_rows([{"market_key": "player_rush_yds", "player_name": "Runner"}]), reference)
    assert result.loc[0, ["match_status", "match_candidate_count"]].tolist() == ["matched", 1]


@pytest.mark.parametrize("rows", [
    [("rb-1", "Runner", "NE", "RB"), ("rb-1", "Other", "NE", "RB")],
    [("rb-1", "Runner", "NE", "RB"), ("rb-1", "Runner", "SEA", "RB")],
])
def test_conflicting_same_player_identities_raise(rows):
    with pytest.raises(ValueError, match="Conflicting player-reference identities"):
        match_player_prop_odds(odds_rows([{"market_key": "player_rush_yds", "player_name": "Runner"}]), players(rows))


def test_unresolved_null_id_reference_rows_are_excluded_not_invented():
    result = match_player_prop_odds(
        odds_rows([{"market_key": "player_rush_yds", "player_name": "Runner"}]),
        players([(pd.NA, "Runner", "NE", "RB")]),
    )
    assert result.loc[0, "match_status"] == "unmatched"
    assert pd.isna(result.loc[0, "player_id"])


def test_empty_valid_odds_returns_complete_schema():
    empty = pd.DataFrame(columns=PLAYER_PROP_ODDS_COLUMNS)
    result = match_player_prop_odds(empty, players([("rb-1", "Runner", "NE", "RB")]))
    assert result.empty
    assert result.columns.tolist() == PLAYER_PROP_MATCH_OUTPUT_COLUMNS


@pytest.mark.parametrize(("column", "value"), [("team", ""), ("position", None)])
def test_invalid_candidate_team_or_position_is_rejected(column, value):
    reference = players([("rb-1", "Runner", "NE", "RB")])
    reference.loc[0, column] = value
    with pytest.raises(ValueError, match=column):
        match_player_prop_odds(
            odds_rows([{"market_key": "player_rush_yds", "player_name": "Runner"}]),
            reference,
        )


@pytest.mark.parametrize("value", ["", "  ", None])
def test_blank_sportsbook_name_is_rejected(value):
    with pytest.raises(ValueError, match="player_name"):
        match_player_prop_odds(odds_rows([{"market_key": "player_pass_yds", "player_name": value}]), players([]))


@pytest.mark.parametrize("value", ["", "  ", None, 7])
def test_invalid_candidate_id_or_name_is_rejected(value):
    with pytest.raises(ValueError):
        match_player_prop_odds(
            odds_rows([{"market_key": "player_rush_yds", "player_name": "Runner"}]),
            players([("rb-1", value, "NE", "RB")]),
        )
    if value is not None:
        with pytest.raises(ValueError):
            match_player_prop_odds(
                odds_rows([{"market_key": "player_rush_yds", "player_name": "Runner"}]),
                players([(value, "Runner", "NE", "RB")]),
            )


def test_schema_types_markets_and_duplicate_columns_are_validated():
    with pytest.raises(TypeError): match_player_prop_odds([], players([]))
    with pytest.raises(TypeError): match_player_prop_odds(odds_rows([]), [])
    with pytest.raises(ValueError, match="exactly match"):
        match_player_prop_odds(odds_rows([]).assign(extra=1), players([]))
    unsupported = odds_rows([{"market_key": "player_reception_yds", "player_name": "Player"}])
    with pytest.raises(ValueError, match="unsupported"):
        match_player_prop_odds(unsupported, players([]))
    duplicated = odds_rows([]).copy(); duplicated.columns = list(duplicated.columns[:-1]) + ["price"]
    with pytest.raises(ValueError, match="duplicate columns"):
        match_player_prop_odds(duplicated, players([]))
    with pytest.raises(ValueError, match="missing required columns"):
        match_player_prop_odds(odds_rows([]), players([]).drop(columns="team"))
    duplicate_players = players([]).copy()
    duplicate_players.columns = ["player_id", "player_name", "team", "team"]
    with pytest.raises(ValueError, match="duplicate columns"):
        match_player_prop_odds(odds_rows([]), duplicate_players)


def test_output_is_deterministic_and_inputs_are_never_mutated():
    odds = odds_rows([{"market_key": "player_rush_yds", "player_name": "Twin Runner"}])
    reference = players([("rb-2", "Twin Runner", "SEA", "RB"), ("rb-1", "Twin Runner", "NE", "RB")])
    odds_before, reference_before = odds.copy(deep=True), reference.copy(deep=True)
    forward = match_player_prop_odds(odds, reference)
    reverse = match_player_prop_odds(odds, reference.iloc[::-1])
    pd.testing.assert_frame_equal(forward, reverse)
    pd.testing.assert_frame_equal(odds, odds_before)
    pd.testing.assert_frame_equal(reference, reference_before)
    bad = odds.copy(deep=True)
    bad["price"] = bad["price"].astype("object")
    bad.loc[0, "price"] = True
    bad_before = bad.copy(deep=True)
    with pytest.raises(ValueError, match="price"):
        match_player_prop_odds(bad, reference)
    pd.testing.assert_frame_equal(bad, bad_before)


def test_output_preserves_odds_row_order_and_resets_a_nondefault_index():
    odds = odds_rows([
        {"market_key": "player_rush_yds", "player_name": "Second"},
        {"market_key": "player_rush_yds", "player_name": "First"},
    ])
    odds.index = [9, 3]
    result = match_player_prop_odds(odds, players([
        ("rb-1", "First", "NE", "RB"), ("rb-2", "Second", "NE", "RB"),
    ]))
    assert result.index.tolist() == [0, 1]
    assert result["player_name"].tolist() == ["Second", "First"]
