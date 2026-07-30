import pandas as pd

from training.evaluate import evaluate_predictions
from training.split_data import chronological_split

DATA_PATH = "data/processed/pitcher_training_2026.csv"

data = pd.read_csv(DATA_PATH)

train_data, validation_data, test_data = chronological_split(
    data,
    train_end_date="2026-06-30",
    validation_end_date="2026-07-15",
)

y_true = validation_data["strikeouts"]
y_pred = validation_data["rolling_k"]

metrics = evaluate_predictions(y_true, y_pred)

print(metrics)