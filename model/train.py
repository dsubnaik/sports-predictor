import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import sys
sys.path.append('.')
from data.fetch_statcast import fetch_pitcher_statcast, aggregate_to_starts
from features.engineer import rolling_features

# calls fetch, aggregate, and rolling features functions to prepare data for training
def prepare_data(player_id, start_date, end_date):
    
    df = fetch_pitcher_statcast(player_id, start_date, end_date)

    df = aggregate_to_starts(df)

    df = rolling_features(df)

    return df

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
    df = prepare_data(543243, '2021-01-01', '2023-10-01')
    df = df.dropna()
    train_model(df)
