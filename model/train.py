import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import sys
sys.path.append('.')
from data.fetch_statcast import fetch_pitcher_statcast, aggregate_to_starts, get_player_id
from features.engineer import rolling_features
from pybaseball import statcast

# calls fetch, aggregate, and rolling features functions to prepare data for training
def prepare_data(year):

    start_date =  f'{year}-03-31'
    end_date = f'{year}-12-31'

    # pull all statcast data for opening month to get active pitcher IDs
    sample = statcast(start_dt=f'{year}-03-28', end_dt=f'{year}-04-30')
    # get unique pitcher IDs who have thrown at least 50 pitches
    pitcher_counts = sample.groupby('pitcher')['pitcher'].count()
    active_pitchers = pitcher_counts[pitcher_counts >= 50].index.tolist()

    all_starts = []
    for player_id in active_pitchers:
        try:
            # pull pitch-level Statcast data for the season
            df = fetch_pitcher_statcast(player_id, start_date, end_date)
            # collapse pitch-level data to one row per start
            df = aggregate_to_starts(df)
            # add rolling features with data leakage prevention
            df = rolling_features(df)
            # add this pitcher's starts to the master list
            all_starts.append(df)
        except:
            # skip pitchers who can't be found or have no data
            continue

    # stack all pitcher DataFrames into one combined DataFrame
    combined = pd.concat(all_starts, ignore_index=True)
    # remove rows with NaN (first 5 starts per pitcher have no rolling data)
    combined = combined.dropna()
    return combined
    

def train_model(df):

    #X is features
    X = df[['rolling_k', 'rolling_swstr', 'rolling_velocity', 'rolling_pitches']]

    #y is target
    y = df['strikeouts']
    
    #split X and y into training and test sets
    #random state ensure the split is the same every time it is ran
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #create model and train
    model =XGBRegressor()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"MAE: {mae:.2f}")

    joblib.dump(model, 'models/xgb_model.joblib')
    print("Model saved.")

if __name__ == "__main__":
    df = prepare_data(2026)
    train_model(df)
