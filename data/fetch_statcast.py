#fetch_statcast.py
#pulls pitch-level statcast data for MLB pitchers and aggregates it per-start level

# pulls data pitch by pitch leve data
from pybaseball import statcast_pitcher
from pybaseball import playerid_lookup
import pandas as pd

#player_id is MLB's internal ID for each pitcher, ex SOnny Gray is 543243
#start_date and end_date are string in 'YYYY-MM-DD' format
#return a raw DataFrame, possibly thousands of row, one for every pitch
def fetch_pitcher_statcast(player_id, start_date, end_date):

    '''
    Pulls every pitch thrown by a pitcher between two dates.
    Returns a DataFrame where each row is one pitch.
    '''

    df = statcast_pitcher(start_date, end_date, player_id)
    return df


def aggregate_to_starts(df):

    """
    Takes pitch-level data and collapses it to one row per start.
    Each row summarizes what happened across all pithes in that game.
    """

    agg = df.groupby('game_pk').agg(
        game_date=('game_date', 'first'),
        pitcher_name=('player_name', 'first'),
        strikeouts=('events', lambda x: (x=='strikeout').sum()),
        swinging_strikes=('description', lambda x: (x == 'swinging_strike').sum()),
        total_pitches=('release_speed', 'count'),
        avg_velocity=('release_speed', 'mean'),
        avg_spin_rate=('release_spin_rate', 'mean'),
    ).reset_index()

    # swinging strike rate = swinging strikes / total pitches
    agg['swstr_pct'] = agg['swinging_strikes'] / agg['total_pitches']

    return agg

def get_player_id(player_name):

    #splits at the first space so it handles player with three names e
    # ex. Luis Garcia Jr returns ['Luis', 'Garcia Jr']
    parts = player_name.split(' ',1) 
    first = parts[0]
    last = parts[1]
    result = playerid_lookup(last, first)

    #return the MLBAM id from the first result

    return result['key_mlbam'].iloc[0]

if __name__ == "__main__":
    player_id = get_player_id('Aaron Nola')
    print(player_id)
