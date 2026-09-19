"""A simple, leakage-safe historical-average QB passing-yards baseline.

The primary feature was already constructed point-in-time by the QB dataset
builder, making it an interpretable standard future models must beat on the
same validation and final test rows.  Cold starts are explicitly identified
and receive one fallback fitted from training targets only.  The intended
workflow is to fit on train, develop against validation, and leave the real
test partition untouched until final model comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from football.training.build_qb_passing_yards_dataset import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TARGET_KEY,
)


# The completed builder's first QB-history feature is its season-window
# passing-yards average.  Referencing FEATURE_COLUMNS keeps this baseline tied
# to the established public schema rather than a parallel feature definition.
HISTORICAL_AVERAGE_COLUMN = FEATURE_COLUMNS[0]
PREDICTION_COLUMN = "baseline_prediction"
FALLBACK_USED_COLUMN = "used_training_fallback"


@dataclass(frozen=True)
class QBPassingYardsBaseline:
    """Training-only mean target used when point-in-time QB history is absent."""

    training_target_mean: float


@dataclass(frozen=True)
class RegressionMetrics:
    """Unrounded regression metrics for one aligned prediction set."""

    row_count: int
    mae: float
    rmse: float
    r_squared: float


def fit_qb_passing_yards_baseline(train_dataset: pd.DataFrame) -> QBPassingYardsBaseline:
    """Fit the sole cold-start fallback from finite training targets only."""

    _validate_keyed_data(train_dataset, [TARGET_COLUMN], "training dataset")
    targets = _finite_numeric(train_dataset[TARGET_COLUMN], "training target_passing_yards")
    if targets.empty:
        raise ValueError("Training dataset has no finite target_passing_yards values")

    return QBPassingYardsBaseline(training_target_mean=float(targets.mean()))


def predict_qb_passing_yards_baseline(
    fitted_baseline: QBPassingYardsBaseline,
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Return auditable baseline predictions without inspecting target values.

    The historical season average is used when finite.  Missing values use the
    fitted training-only fallback; infinite or nonnumeric feature values are
    rejected rather than treated as cold starts.
    """

    if not isinstance(fitted_baseline, QBPassingYardsBaseline):
        raise TypeError("fitted_baseline must be a QBPassingYardsBaseline")
    if not np.isfinite(fitted_baseline.training_target_mean):
        raise ValueError("fitted_baseline training_target_mean must be finite")

    _validate_keyed_data(dataset, [HISTORICAL_AVERAGE_COLUMN], "prediction dataset")
    historical_average = _historical_average_values(dataset[HISTORICAL_AVERAGE_COLUMN])
    missing_history = historical_average.isna()
    predictions = historical_average.where(
        ~missing_history, fitted_baseline.training_target_mean
    )

    result_columns = [*TARGET_KEY]
    if TARGET_COLUMN in dataset.columns:
        result_columns.append(TARGET_COLUMN)
    result = dataset.loc[:, result_columns].copy(deep=True)
    result[PREDICTION_COLUMN] = predictions.to_numpy()
    result[FALLBACK_USED_COLUMN] = missing_history.to_numpy(dtype=bool)

    return result.sort_values(TARGET_KEY, kind="mergesort").reset_index(drop=True)


def evaluate_qb_passing_yards_predictions(
    actual_dataset: pd.DataFrame,
    predictions: pd.DataFrame,
) -> RegressionMetrics:
    """Evaluate aligned predictions using MAE, RMSE, and finite-convention R-squared.

    Actual and prediction target keys must match exactly; no rows are dropped.
    For constant actual targets, R-squared is 1.0 for exact predictions and 0.0
    otherwise, matching scikit-learn's default finite behavior.
    """

    _validate_keyed_data(actual_dataset, [TARGET_COLUMN], "actual dataset")
    _validate_keyed_data(predictions, [PREDICTION_COLUMN], "prediction output")
    _validate_matching_keys(actual_dataset, predictions)

    actual_by_key = actual_dataset.loc[:, [*TARGET_KEY, TARGET_COLUMN]].merge(
        predictions.loc[:, [*TARGET_KEY, PREDICTION_COLUMN]],
        on=TARGET_KEY,
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    actual = _finite_numeric(actual_by_key[TARGET_COLUMN], "actual target_passing_yards")
    predicted = _finite_numeric(actual_by_key[PREDICTION_COLUMN], "baseline prediction")
    if len(actual) != len(actual_by_key) or len(predicted) != len(actual_by_key):
        raise ValueError("Actual targets and predictions must be finite for every row")

    residuals = actual.to_numpy() - predicted.to_numpy()
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    total_sum_squares = float(np.sum(np.square(actual.to_numpy() - actual.mean())))
    residual_sum_squares = float(np.sum(np.square(residuals)))
    if total_sum_squares == 0:
        r_squared = 1.0 if residual_sum_squares == 0 else 0.0
    else:
        r_squared = float(1 - residual_sum_squares / total_sum_squares)

    return RegressionMetrics(
        row_count=len(actual_by_key),
        mae=mae,
        rmse=rmse,
        r_squared=r_squared,
    )


def _validate_keyed_data(
    data: pd.DataFrame,
    required_columns: list[str],
    name: str,
) -> None:
    missing_columns = sorted(set([*TARGET_KEY, *required_columns]).difference(data.columns))
    if missing_columns:
        raise ValueError(f"{name.capitalize()} is missing required columns: {missing_columns}")
    if data.empty:
        raise ValueError(f"{name.capitalize()} must not be empty")
    if data[TARGET_KEY].isna().any().any():
        raise ValueError(f"{name.capitalize()} has missing target-key values")
    duplicates = data.loc[data.duplicated(TARGET_KEY, keep=False), TARGET_KEY]
    if not duplicates.empty:
        keys = duplicates.drop_duplicates().sort_values(TARGET_KEY, kind="mergesort")
        raise ValueError(
            f"{name.capitalize()} has duplicate target keys: "
            f"{keys.to_dict(orient='records')}"
        )


def _finite_numeric(values: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric).all():
        raise ValueError(f"{name} values must be finite")
    return numeric.astype(float)


def _historical_average_values(values: pd.Series) -> pd.Series:
    missing = values.isna()
    numeric = pd.to_numeric(values, errors="coerce")
    invalid = ~missing & (numeric.isna() | ~np.isfinite(numeric))
    if invalid.any():
        raise ValueError(
            f"{HISTORICAL_AVERAGE_COLUMN} values must be finite or missing"
        )
    return numeric.astype(float)


def _validate_matching_keys(actual_dataset: pd.DataFrame, predictions: pd.DataFrame) -> None:
    actual_keys = set(map(tuple, actual_dataset[TARGET_KEY].to_numpy()))
    prediction_keys = set(map(tuple, predictions[TARGET_KEY].to_numpy()))
    if actual_keys != prediction_keys:
        raise ValueError("Actual dataset and prediction output target keys must match exactly")
