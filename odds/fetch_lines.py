import requests
import os
from dotenv import load_dotenv

# load environment variables from .env file so API key is never hardcoded
load_dotenv()
API_KEY = os.getenv('ODDS_API_KEY')

# step 1: get all MLB event IDs for today
def get_event_ids():
    
    url = 'https://api.the-odds-api.com/v4/sports/baseball_mlb/events'
    params = {'apiKey': API_KEY}
    response = requests.get(url, params=params)
    return response.json()

# step 2: for each event, fetch pitcher strikeout props
# limited to first 3 events to save API credits (500/month on free tier)
def fetch_strikeout_lines():
    
    events = get_event_ids()
    results = []
    for event in events[:3]:
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

# step 3: parse the raw nested API response into a clean list
# skips games with no bookmaker data
# extracts pitcher name, line, and matchup from the first available bookmaker
def parse_lines(results):
    
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