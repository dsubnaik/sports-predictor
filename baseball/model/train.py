"""Legacy model-training workflow for the pitcher strikeout model.

This script predates data/build_pitcher_dataset.py and has not been migrated to
the newer official-starter matching workflow.
"""

import sys

import joblib
import pandas as pd
from pybaseball import statcast
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

sys.path.append('.')

from baseball.data.fetch_statcast import fetch_pitcher_statcast, aggregate_to_starts
from baseball.features.engineer import rolling_features


def prepare_data(year):
    """Build the legacy training frame for a season.

    Active pitchers are identified from an early-season sample, then each
    pitcher's full-season Statcast data is fetched and converted into rolling
    features. This preserves the old training behavior while the official-start
    dataset builder evolves separately.
    """

    start_date = f'{year}-03-31'
    end_date = f'{year}-12-31'

    # The 50-pitch threshold is a legacy shortcut for finding likely active
    # pitchers without fetching every MLB player individually.
    sample = statcast(start_dt=f'{year}-03-28', end_dt=f'{year}-04-30')
    pitcher_counts = sample.groupby('pitcher')['pitcher'].count()
    active_pitchers = pitcher_counts[pitcher_counts >= 50].index.tolist()

    all_starts = []
    for player_id in active_pitchers:
        try:
            df = fetch_pitcher_statcast(player_id, start_date, end_date)
            df = aggregate_to_starts(df)
            df = rolling_features(df)
            all_starts.append(df)
        except:
            # Preserve the legacy behavior: skip pitchers whose Statcast data
            # cannot be fetched or aggregated.
            continue

    combined = pd.concat(all_starts, ignore_index=True)

    # The first five starts for each pitcher lack enough prior starts to create
    # five-game rolling features.
    combined = combined.dropna()
    return combined


def train_model(df):
    """Train and save the XGBoost strikeout regressor."""

    X = df[['rolling_k', 'rolling_swstr', 'rolling_velocity', 'rolling_pitches']]
    y = df['strikeouts']

    # Keep the split deterministic so MAE comparisons are repeatable.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = XGBRegressor()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"MAE: {mae:.2f}")

    joblib.dump(model, 'models/xgb_model.joblib')
    print("Model saved.")


if __name__ == "__main__":
    df = prepare_data(2026)
    print(f"Rows: {len(df)}")
    print(f"Pitchers: {df['pitcher'].nunique()}")
    print(df.groupby('pitcher').size().describe())
    train_model(df)
