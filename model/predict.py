import pandas as pd
import joblib
import sys
sys.path.append('.')
from data.fetch_statcast import fetch_pitcher_statcast, aggregate_to_starts
from features.engineer import rolling_features

#load the saved model
def load_model():
    return joblib.load('models/xgb_model.joblib')

def predict_strikeouts(rolling_k, rolling_swstr, rolling_velocity, rolling_pitches):

    model=load_model()

    df = pd.DataFrame({
        'rolling_k': [rolling_k],
        'rolling_swstr': [rolling_swstr],
        'rolling_velocity': [rolling_velocity],
        'rolling_pitches': [rolling_pitches]
        })
    
    return model.predict(df)[0]

if __name__ == "__main__":
    prediction = predict_strikeouts(
        rolling_k=6.8,
        rolling_swstr=0.12,
        rolling_velocity=88.5,
        rolling_pitches=90.0
    )
    print(f"Predicted strikeouts: {prediction:.1f}")