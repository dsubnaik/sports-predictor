import pandas as pd
import sys
#imports from fetch_statcast.py, brings the data over
sys.path.append('.')
from data.fetch_statcast import fetch_pitcher_statcast, aggregate_to_starts

#rolling features method
def rolling_features(df):

    #sort by function
    df.sort_values('game_date', inplace=True)

    # shift(1) prevents data leakage by excluding current game from its own rolling average
    # rolling(5).mean() calculates the average of the last 5 starts 
    df['rolling_k']=df['strikeouts'].shift(1).rolling(5).mean()

    #swstr=swinging strike rate
    df['rolling_swstr']=df['swstr_pct'].shift(1).rolling(5).mean()

    df['rolling_velocity']=df['avg_velocity'].shift(1).rolling(5).mean()

    df['rolling_pitches']=df['total_pitches'].shift(1).rolling(5).mean()

    return df

if __name__ == "__main__":
    df = fetch_pitcher_statcast(543243, '2023-03-30', '2023-10-01')
    starts = aggregate_to_starts(df)
    features = rolling_features(starts)
    pd.set_option('display.max_rows', None)
    print(features[['game_date', 'strikeouts', 'rolling_k', 'rolling_swstr', 'rolling_velocity', 'rolling_pitches']])



