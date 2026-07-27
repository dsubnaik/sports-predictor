"""Fetch completed MLB games and identify each team's official starting pitcher.

The dataset builder uses this module to avoid treating bulk relievers or other
pitching appearances as starts. MLB's boxscore pitching stats mark the official
starter with gamesStarted == 1 for each team.
"""

import json
from urllib.parse import urlencode
from urllib.request import urlopen

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
BOXSCORE_URL = "https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"


def fetch_completed_games(start_date: str, end_date: str) -> list[dict]:
    """Return final regular-season MLB games in the requested date range."""
    params = {
        "sportId": 1,
        "startDate": start_date,
        "endDate": end_date,
        "gameType": "R",
    }

    query_string = urlencode(params)

    with urlopen(f"{SCHEDULE_URL}?{query_string}") as response:
        data = json.load(response)

    completed_games = []

    for date_record in data["dates"]:
        for game in date_record["games"]:
            if game["status"]["detailedState"] == "Final":
                completed_games.append(
                    {
                        "game_pk": game["gamePk"],
                        "game_date": game["officialDate"],
                    }
                )

    return completed_games


def fetch_boxscore(game_pk: int) -> dict:
    """Fetch the MLB Stats API boxscore for a single game."""
    url = BOXSCORE_URL.format(game_pk=game_pk)

    with urlopen(url) as response:
        data = json.load(response)

    return data


def find_starting_pitcher(team_boxscore: dict) -> dict:
    """Find the official starting pitcher in one team's boxscore.

    The pitcher list can include relievers, so the starter is selected from the
    pitching stats instead of relying on list position.
    """
    for pitcher_id in team_boxscore["pitchers"]:
        player_key = f"ID{pitcher_id}"
        pitcher = team_boxscore["players"][player_key]

        pitching_stats = pitcher.get("stats", {}).get("pitching", {})

        if pitching_stats.get("gamesStarted") == 1:
            return {
                "pitcher_id": pitcher_id,
                "pitcher_name": pitcher["person"]["fullName"],
            }

    raise ValueError("No starting pitcher found")


def extract_game_starters(game: dict, boxscore: dict) -> list[dict]:
    """Return the away and home starting pitchers for one completed game."""
    starters = []

    for home_away in ["away", "home"]:
        team_boxscore = boxscore["teams"][home_away]
        starter = find_starting_pitcher(team_boxscore)

        starters.append(
            {
                "game_pk": game["game_pk"],
                "game_date": game["game_date"],
                "pitcher_id": starter["pitcher_id"],
                "pitcher_name": starter["pitcher_name"],
                "home_away": home_away,
            }
        )

    return starters


def fetch_mlb_starters(start_date: str, end_date: str) -> list[dict]:
    """Fetch official starters for every completed game in the date range."""
    games = fetch_completed_games(start_date, end_date)
    all_starters = []

    for game in games:
        boxscore = fetch_boxscore(game["game_pk"])
        game_starters = extract_game_starters(game, boxscore)
        all_starters.extend(game_starters)

    return all_starters
