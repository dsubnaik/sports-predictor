import pandas as pd

from sklearn.linear_model import LinearRegression

from baseball.training.evaluate import evaluate_predictions
from baseball.training.split_data import chronological_split


DATA_PATH = "data/processed/pitcher_training_2026.csv"

FEATURES = [
    "rolling_k",
    "rolling_swstr",
    "rolling_velocity",
    "rolling_pitches",
    "opponent_k_rate",
]

data = pd.read_csv(DATA_PATH)

train_data, validation_data, test_data = chronological_split(
    data,
    train_end_date="2026-06-30",
    validation_end_date="2026-07-15",
)

X_train = train_data[FEATURES]
y_train = train_data["strikeouts"]

X_validation = validation_data[FEATURES]
y_validation = validation_data["strikeouts"]

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_validation)

metrics = evaluate_predictions(y_validation, y_pred)
print(metrics)