"""Leakage-safe validation comparison and error diagnostics for QB models.

All five completed models are fit only on a supplied chronological training
partition and predict only the supplied validation partition.  The public
orchestrator deliberately never accesses ``dataset_split.test``.  It retains
aggregate immutable diagnostics only; no input or row-level dataframe is kept.

Signed error is always ``prediction - actual``.  Matchup rank is tiered as a
numeric rank only: the builder ranks passing yards allowed descending, so rank
1 has the highest point-in-time passing-yards-allowed average.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from football.training.build_qb_passing_yards_dataset import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TARGET_KEY,
)
from football.training.qb_passing_yards_baseline import (
    PREDICTION_COLUMN as BASELINE_PREDICTION_COLUMN,
    RegressionMetrics,
    evaluate_qb_passing_yards_predictions,
    fit_qb_passing_yards_baseline,
    predict_qb_passing_yards_baseline,
)
from football.training.qb_passing_yards_gradient_boosting import (
    fit_qb_passing_yards_gradient_boosting,
    predict_qb_passing_yards_gradient_boosting,
)
from football.training.qb_passing_yards_linear_regression import (
    fit_qb_passing_yards_linear_regression,
    predict_qb_passing_yards_linear_regression,
)
from football.training.qb_passing_yards_random_forest import (
    fit_qb_passing_yards_random_forest,
    predict_qb_passing_yards_random_forest,
)
from football.training.qb_passing_yards_xgboost import (
    fit_qb_passing_yards_xgboost,
    predict_qb_passing_yards_xgboost,
)
from football.training.split_qb_passing_yards_dataset import (
    QBPassingYardsDatasetSplit,
)


MODEL_NAMES = (
    "Historical average",
    "Linear Regression",
    "Random Forest",
    "sklearn Gradient Boosting",
    "XGBoost",
)
PREDICTION_TIE_TOLERANCE = 1e-12
EXACT_PREDICTION_TOLERANCE = 1e-12
PERCENTILE_METHOD = "linear"


@dataclass(frozen=True)
class ErrorDistributionDiagnostics:
    """Validation error-shape diagnostics using NumPy's ``linear`` percentile."""

    median_absolute_error: float
    percentile_75_absolute_error: float
    percentile_90_absolute_error: float
    maximum_absolute_error: float
    underprediction_count: int
    overprediction_count: int
    approximately_exact_prediction_count: int


@dataclass(frozen=True)
class ModelSliceMetrics:
    """One model's metrics for a feature-defined validation slice."""

    model_name: str
    row_count: int
    metrics: RegressionMetrics | None
    mean_signed_error: float | None


@dataclass(frozen=True)
class ValidationSlice:
    """One point-in-time feature bucket and every model's diagnostics."""

    family_name: str
    bucket_name: str
    row_count: int
    model_metrics: tuple[ModelSliceMetrics, ...]


@dataclass(frozen=True)
class ModelValidationDiagnostics:
    """Aggregate validation metrics and established cold-start subgroups."""

    model_name: str
    metrics: RegressionMetrics
    mean_signed_error: float
    mean_prediction: float
    mean_actual: float
    cold_start_metrics: ModelSliceMetrics
    non_cold_start_metrics: ModelSliceMetrics
    error_distribution: ErrorDistributionDiagnostics


@dataclass(frozen=True)
class RowLevelWinCount:
    """Rows where this model ties for the smallest absolute error."""

    model_name: str
    win_count: int


@dataclass(frozen=True)
class PairwiseMAEDifference:
    """``first_model.mae - second_model.mae``; positive favors second."""

    first_model_name: str
    second_model_name: str
    first_minus_second_mae: float


@dataclass(frozen=True)
class PredictionDisagreement:
    """Mean absolute difference between two aligned validation predictions."""

    first_model_name: str
    second_model_name: str
    mean_absolute_prediction_difference: float


@dataclass(frozen=True)
class QBPassingYardsModelComparison:
    """Immutable train/validation-only comparison with no retained frames."""

    validation_row_count: int
    model_diagnostics: tuple[ModelValidationDiagnostics, ...]
    slices: tuple[ValidationSlice, ...]
    row_level_win_counts: tuple[RowLevelWinCount, ...]
    pairwise_mae_differences: tuple[PairwiseMAEDifference, ...]
    prediction_disagreements: tuple[PredictionDisagreement, ...]


def compare_qb_passing_yards_models(
    dataset_split: QBPassingYardsDatasetSplit,
) -> QBPassingYardsModelComparison:
    """Fit five fixed models on train and compare only canonical validation.

    The chronological test partition is intentionally not read, copied,
    validated, transformed, predicted, or scored. All learning and imputation
    happens inside the existing public fitting APIs using ``dataset_split.train``.
    """

    if not isinstance(dataset_split, QBPassingYardsDatasetSplit):
        raise TypeError("dataset_split must be a QBPassingYardsDatasetSplit")
    validation = _canonical_validation(dataset_split.validation)
    prediction_frames = _fit_and_predict_train_validation(
        dataset_split.train, validation
    )
    _validate_prediction_alignment(validation, prediction_frames)

    actual = _finite_values(
        validation[TARGET_COLUMN], "validation target_passing_yards"
    ).to_numpy(dtype=float)
    cold_start_mask = validation["qb_season_passing_yards_avg"].isna().to_numpy()
    diagnostics = tuple(
        _model_diagnostics(name, validation, predictions, actual, cold_start_mask)
        for name, predictions in prediction_frames.items()
    )
    slices = _build_slices(validation, prediction_frames, actual, cold_start_mask)
    return QBPassingYardsModelComparison(
        validation_row_count=len(validation),
        model_diagnostics=diagnostics,
        slices=slices,
        row_level_win_counts=_row_level_wins(prediction_frames, actual),
        pairwise_mae_differences=_pairwise_mae_differences(diagnostics),
        prediction_disagreements=_prediction_disagreements(prediction_frames),
    )


def _fit_and_predict_train_validation(
    train: pd.DataFrame, validation: pd.DataFrame
) -> dict[str, np.ndarray]:
    baseline = fit_qb_passing_yards_baseline(train)
    linear = fit_qb_passing_yards_linear_regression(train)
    random_forest = fit_qb_passing_yards_random_forest(train)
    gradient_boosting = fit_qb_passing_yards_gradient_boosting(train)
    xgboost = fit_qb_passing_yards_xgboost(train)
    prediction_outputs = {
        "Historical average": predict_qb_passing_yards_baseline(baseline, validation),
        "Linear Regression": predict_qb_passing_yards_linear_regression(linear, validation),
        "Random Forest": predict_qb_passing_yards_random_forest(random_forest, validation),
        "sklearn Gradient Boosting": predict_qb_passing_yards_gradient_boosting(
            gradient_boosting, validation
        ),
        "XGBoost": predict_qb_passing_yards_xgboost(xgboost, validation),
    }
    return {
        name: _prediction_values(name, output, validation)
        for name, output in prediction_outputs.items()
    }


def _canonical_validation(validation: pd.DataFrame) -> pd.DataFrame:
    required = [*TARGET_KEY, TARGET_COLUMN, *FEATURE_COLUMNS]
    if not isinstance(validation, pd.DataFrame):
        raise TypeError("Validation frame must be a pandas DataFrame")
    missing = sorted(set(required).difference(validation.columns))
    if missing:
        raise ValueError(f"Validation frame is missing required columns: {missing}")
    if validation.empty:
        raise ValueError("Validation frame must not be empty")
    if validation[TARGET_KEY].isna().any().any():
        raise ValueError("Validation frame has missing target-key values")
    if validation.duplicated(TARGET_KEY).any():
        raise ValueError("Validation frame has duplicate target keys")
    _finite_values(validation[TARGET_COLUMN], "validation target_passing_yards")
    return validation.sort_values(TARGET_KEY, kind="mergesort").copy(deep=True)


def _prediction_values(
    name: str, predictions: pd.DataFrame, validation: pd.DataFrame
) -> np.ndarray:
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError(f"{name} predictions must be a pandas DataFrame")
    missing = sorted(set([*TARGET_KEY]).difference(predictions.columns))
    if missing:
        raise ValueError(f"{name} predictions are missing target-key columns: {missing}")
    prediction_columns = [column for column in predictions if column.endswith("_prediction")]
    if len(prediction_columns) != 1:
        raise ValueError(f"{name} predictions must contain exactly one prediction column")
    if len(predictions) == 0 or predictions[TARGET_KEY].isna().any().any():
        raise ValueError(f"{name} predictions have missing target-key values")
    if predictions.duplicated(TARGET_KEY).any():
        raise ValueError(f"{name} predictions have duplicate target keys")
    prediction_keys = set(map(tuple, predictions[TARGET_KEY].to_numpy()))
    validation_keys = set(map(tuple, validation[TARGET_KEY].to_numpy()))
    if prediction_keys != validation_keys:
        raise ValueError(f"{name} prediction keys do not match validation keys")
    ordered = predictions.sort_values(TARGET_KEY, kind="mergesort")
    values = _finite_values(ordered[prediction_columns[0]], f"{name} prediction")
    return values.to_numpy(dtype=float)


def _validate_prediction_alignment(
    validation: pd.DataFrame, prediction_frames: dict[str, np.ndarray]
) -> None:
    if tuple(prediction_frames) != MODEL_NAMES:
        raise ValueError("Model prediction names do not match the completed model contract")
    for name, values in prediction_frames.items():
        if len(values) != len(validation):
            raise ValueError(f"{name} prediction count does not match validation rows")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} predictions must be finite")


def _model_diagnostics(
    name: str,
    validation: pd.DataFrame,
    predictions: np.ndarray,
    actual: np.ndarray,
    cold_start_mask: np.ndarray,
) -> ModelValidationDiagnostics:
    metrics = _shared_metrics(validation, predictions)
    signed_errors = predictions - actual
    return ModelValidationDiagnostics(
        model_name=name,
        metrics=metrics,
        mean_signed_error=float(np.mean(signed_errors)),
        mean_prediction=float(np.mean(predictions)),
        mean_actual=float(np.mean(actual)),
        cold_start_metrics=_slice_model_metrics(
            name, validation, predictions, cold_start_mask
        ),
        non_cold_start_metrics=_slice_model_metrics(
            name, validation, predictions, ~cold_start_mask
        ),
        error_distribution=_error_distribution(signed_errors),
    )


def _shared_metrics(validation: pd.DataFrame, predictions: np.ndarray) -> RegressionMetrics:
    output = validation.loc[:, TARGET_KEY].copy(deep=True)
    output[BASELINE_PREDICTION_COLUMN] = predictions
    return evaluate_qb_passing_yards_predictions(validation, output)


def _slice_model_metrics(
    name: str, validation: pd.DataFrame, predictions: np.ndarray, mask: np.ndarray
) -> ModelSliceMetrics:
    row_count = int(mask.sum())
    if row_count == 0:
        return ModelSliceMetrics(name, 0, None, None)
    rows = validation.loc[mask]
    selected_predictions = predictions[mask]
    actual = _finite_values(rows[TARGET_COLUMN], "validation target_passing_yards").to_numpy()
    return ModelSliceMetrics(
        model_name=name,
        row_count=row_count,
        metrics=_shared_metrics(rows, selected_predictions),
        mean_signed_error=float(np.mean(selected_predictions - actual)),
    )


def _build_slices(
    validation: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    actual: np.ndarray,
    cold_start_mask: np.ndarray,
) -> tuple[ValidationSlice, ...]:
    del actual  # Metrics are evaluated through the existing shared evaluator.
    families = (
        ("cold_start_status", (("cold_start", cold_start_mask), ("non_cold_start", ~cold_start_mask))),
        ("qb_season_history_depth", _history_depth_buckets(validation["qb_season_history_games"])),
        ("qb_recent_history_depth", _recent_depth_buckets(validation["qb_last3_history_games"])),
        ("defense_season_history_depth", _history_depth_buckets(validation["defense_season_history_games"])),
        ("defense_matchup_rank_tier", _matchup_rank_buckets(validation["defense_matchup_rank"])),
    )
    reports = []
    for family_name, buckets in families:
        for bucket_name, mask in buckets:
            mask_array = np.asarray(mask, dtype=bool)
            reports.append(
                ValidationSlice(
                    family_name=family_name,
                    bucket_name=bucket_name,
                    row_count=int(mask_array.sum()),
                    model_metrics=tuple(
                        _slice_model_metrics(name, validation, values, mask_array)
                        for name, values in predictions.items()
                    ),
                )
            )
    return tuple(reports)


def _history_depth_buckets(values: pd.Series) -> tuple[tuple[str, np.ndarray], ...]:
    numeric = _finite_values(values, "history sample size").to_numpy()
    if (numeric < 0).any() or np.any(numeric % 1):
        raise ValueError("History sample sizes must be nonnegative whole numbers")
    return (
        ("0_games", numeric == 0),
        ("1_to_3_games", (numeric >= 1) & (numeric <= 3)),
        ("4_to_8_games", (numeric >= 4) & (numeric <= 8)),
        ("9_plus_games", numeric >= 9),
    )


def _recent_depth_buckets(values: pd.Series) -> tuple[tuple[str, np.ndarray], ...]:
    numeric = _finite_values(values, "recent history sample size").to_numpy()
    if (numeric < 0).any() or np.any(numeric % 1) or (numeric > 3).any():
        raise ValueError("Recent history sample sizes must be whole numbers from zero through three")
    return tuple((f"{count}_games" if count != 1 else "1_game", numeric == count) for count in range(4))


def _matchup_rank_buckets(values: pd.Series) -> tuple[tuple[str, np.ndarray], ...]:
    numeric = pd.to_numeric(values, errors="coerce")
    invalid = ~values.isna() & (numeric.isna() | ~np.isfinite(numeric))
    if invalid.any():
        raise ValueError("Defense matchup rank must be finite or missing")
    populated = numeric.notna().to_numpy()
    numbers = numeric.fillna(0).to_numpy(dtype=float)
    if ((numbers[populated] < 1) | (numbers[populated] > 32) | (numbers[populated] % 1 != 0)).any():
        raise ValueError("Defense matchup rank must be a whole number from one through 32 or missing")
    return (
        ("1_to_10", populated & (numbers >= 1) & (numbers <= 10)),
        ("11_to_22", populated & (numbers >= 11) & (numbers <= 22)),
        ("23_to_32", populated & (numbers >= 23) & (numbers <= 32)),
        ("missing", ~populated),
    )


def _error_distribution(signed_errors: np.ndarray) -> ErrorDistributionDiagnostics:
    absolute_errors = np.abs(signed_errors)
    return ErrorDistributionDiagnostics(
        median_absolute_error=float(np.percentile(absolute_errors, 50, method=PERCENTILE_METHOD)),
        percentile_75_absolute_error=float(np.percentile(absolute_errors, 75, method=PERCENTILE_METHOD)),
        percentile_90_absolute_error=float(np.percentile(absolute_errors, 90, method=PERCENTILE_METHOD)),
        maximum_absolute_error=float(np.max(absolute_errors)),
        underprediction_count=int((signed_errors < -EXACT_PREDICTION_TOLERANCE).sum()),
        overprediction_count=int((signed_errors > EXACT_PREDICTION_TOLERANCE).sum()),
        approximately_exact_prediction_count=int((absolute_errors <= EXACT_PREDICTION_TOLERANCE).sum()),
    )


def _row_level_wins(
    predictions: dict[str, np.ndarray], actual: np.ndarray
) -> tuple[RowLevelWinCount, ...]:
    matrix = np.vstack(tuple(predictions.values()))
    absolute_errors = np.abs(matrix - actual)
    minimums = absolute_errors.min(axis=0)
    return tuple(
        RowLevelWinCount(
            name,
            int(np.isclose(absolute_errors[index], minimums, rtol=0.0, atol=PREDICTION_TIE_TOLERANCE).sum()),
        )
        for index, name in enumerate(predictions)
    )


def _pairwise_mae_differences(
    diagnostics: tuple[ModelValidationDiagnostics, ...]
) -> tuple[PairwiseMAEDifference, ...]:
    return tuple(
        PairwiseMAEDifference(first.model_name, second.model_name, first.metrics.mae - second.metrics.mae)
        for first, second in combinations(diagnostics, 2)
    )


def _prediction_disagreements(
    predictions: dict[str, np.ndarray]
) -> tuple[PredictionDisagreement, ...]:
    return tuple(
        PredictionDisagreement(first_name, second_name, float(np.mean(np.abs(first - second))))
        for (first_name, first), (second_name, second) in combinations(predictions.items(), 2)
    )


def _finite_values(values: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric).all():
        raise ValueError(f"{name} values must be finite")
    return numeric.astype(float)
