"""Tests for MLB starting-pitcher identification and extraction."""

from data.fetch_mlb_starters import (
    extract_game_starters,
    find_starting_pitcher,
)
import pytest


def test_find_starting_pitcher():
    team_boxscore = {
        "pitchers": [100, 200],
        "players": {
            "ID100": {
                "person": {"fullName": "Starting Pitcher"},
                "stats": {
                    "pitching": {
                        "gamesStarted": 1,
                    }
                },
            },
            "ID200": {
                "person": {"fullName": "Relief Pitcher"},
                "stats": {
                    "pitching": {
                        "gamesStarted": 0,
                    }
                },
            },
        },
    }

    result = find_starting_pitcher(team_boxscore)

    assert result == {
        "pitcher_id": 100,
        "pitcher_name": "Starting Pitcher",
    }

def test_find_starting_pitcher_raises_when_no_starter_exists():
    team_boxscore = {
        "pitchers": [100, 200],
        "players": {
            "ID100": {
                "person": {"fullName": "Relief Pitcher One"},
                "stats": {"pitching": {"gamesStarted": 0}},
            },
            "ID200": {
                "person": {"fullName": "Relief Pitcher Two"},
                "stats": {"pitching": {"gamesStarted": 0}},
            },
        },
    }

    with pytest.raises(ValueError, match="No starting pitcher found"):
        find_starting_pitcher(team_boxscore)

def test_extract_game_starters():
    game = {
        "game_pk": 123456,
        "game_date": "2026-04-01",
    }

    boxscore = {
        "teams": {
            "away": {
                "pitchers": [100],
                "players": {
                    "ID100": {
                        "person": {"fullName": "Away Starter"},
                        "stats": {"pitching": {"gamesStarted": 1}},
                    }
                },
            },
            "home": {
                "pitchers": [200],
                "players": {
                    "ID200": {
                        "person": {"fullName": "Home Starter"},
                        "stats": {"pitching": {"gamesStarted": 1}},
                    }
                },
            },
        }
    }

    result = extract_game_starters(game, boxscore)

    assert result == [
        {
            "game_pk": 123456,
            "game_date": "2026-04-01",
            "pitcher_id": 100,
            "pitcher_name": "Away Starter",
            "home_away": "away",
        },
        {
            "game_pk": 123456,
            "game_date": "2026-04-01",
            "pitcher_id": 200,
            "pitcher_name": "Home Starter",
            "home_away": "home",
        },
    ]