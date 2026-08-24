"""Fetch and normalize MLB pitcher strikeout lines from The Odds API."""

import os

import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('ODDS_API_KEY')


def get_event_ids():
    """Return the MLB events available from The Odds API for the current day."""

    url = 'https://api.the-odds-api.com/v4/sports/baseball_mlb/events'
    params = {'apiKey': API_KEY}
    response = requests.get(url, params=params)
    return response.json()


def fetch_strikeout_lines():
    """Fetch pitcher strikeout prop markets for each listed MLB event.

    This function returns the raw API payloads. parse_lines() owns the narrowing
    from nested bookmaker/market data into the app's display format.
    """

    events = get_event_ids()
    results = []
    for event in events:
        event_id = event['id']
        url = f'https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{event_id}/odds'
        params = {
            'apiKey': API_KEY,
            'regions': 'us',
            'markets': 'pitcher_strikeouts',
            'oddsFormat': 'american'
        }
        response = requests.get(url, params=params)
        results.append(response.json())
    return results


def parse_lines(results):
    """Convert raw odds responses into pitcher, line, and matchup records.

    The app currently uses the first bookmaker and first outcome available for a
    game. Games without bookmaker data are skipped because there is no line to
    compare against the model projection.
    """

    lines = []
    for event in results:
        if len(event['bookmakers']) == 0:
            continue
        bookmaker = event['bookmakers'][0]
        outcome = bookmaker['markets'][0]['outcomes'][0]
        lines.append({
            'pitcher': outcome['description'],
            'line': outcome['point'],
            'game': event['away_team'] + ' @ ' + event['home_team']
        })
    return lines


if __name__ == "__main__":
    results = fetch_strikeout_lines()
    lines = parse_lines(results)
    print(lines)
