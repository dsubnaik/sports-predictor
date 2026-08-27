"""Prediction helpers for the saved pitcher strikeout model."""

import pandas as pd
import joblib

from baseball.config import XGB_MODEL_PATH


def load_model():
    """Load the trained XGBoost model from the ignored models directory."""
    return joblib.load(XGB_MODEL_PATH)


def predict_strikeouts(rolling_k, rolling_swstr, rolling_velocity, rolling_pitches):
    """Predict strikeouts from the four rolling features used at training time.

    The DataFrame column names and order mirror the training feature set. Keep
    these aligned with model/train.py unless the model is retrained.
    """

    model = load_model()

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
