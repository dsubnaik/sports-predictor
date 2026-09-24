"""Leakage-safe Linear Regression training for QB passing yards.

The completed QB dataset has already constructed every predictive feature at a
pregame point in time.  This module fits its median imputation and ordinary
Linear Regression estimator on a supplied training partition only, then uses
that fitted pipeline for validation prediction.  The chronological test
partition is deliberately not accessed by :func:`train_and_validate_qb_passing_yards_linear_regression`.

The intended workflow is train fitting, validation comparison, and one final
test evaluation only after model decisions are complete.  No preprocessing
statistics, fallback values, or coefficients are fitted from validation or
test rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from football.training.build_qb_passing_yards_dataset import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TARGET_KEY,
)
from football.training.qb_passing_yards_baseline import (
    HISTORICAL_AVERAGE_COLUMN,
    PREDICTION_COLUMN as BASELINE_PREDICTION_COLUMN,
    RegressionMetrics,
    evaluate_qb_passing_yards_predictions,
    fit_qb_passing_yards_baseline,
    predict_qb_passing_yards_baseline,
)
from football.training.split_qb_passing_yards_dataset import (
    QBPassingYardsDatasetSplit,
)


PREDICTION_COLUMN = "linear_regression_prediction"
COLD_START_COLUMN = "is_qb_history_cold_start"

# These are derived from the builder's public feature contract rather than
# maintained as another hand-written model feature list.
BINARY_FEATURE_COLUMNS = tuple(
    feature for feature in FEATURE_COLUMNS if feature.endswith("_missing_history")
)
NUMERIC_FEATURE_COLUMNS = tuple(
    feature for feature in FEATURE_COLUMNS if feature not in BINARY_FEATURE_COLUMNS
)


@dataclass(frozen=True)
class NumericImputationSummary:
    """One training-fitted median used for a numeric predictive feature."""

    feature_name: str
    training_median: float


@dataclass(frozen=True)
class LinearRegressionCoefficient:
    """One transformed-feature coefficient; magnitude is not feature importance."""

    transformed_feature_name: str
    coefficient: float


@dataclass(frozen=True)
class QBPassingYardsLinearRegressionModel:
    """A fitted training-only preprocessing pipeline and Linear Regression model.

    ``pipeline`` intentionally contains both transformations and the estimator,
    preventing callers from predicting without the training-fitted imputation.
    It retains no input dataframe or design matrix.
    """

    pipeline: Pipeline = field(repr=False, compare=False)
    feature_columns: tuple[str, ...]
    numeric_feature_columns: tuple[str, ...]
    binary_feature_columns: tuple[str, ...]
    numeric_imputation_summaries: tuple[NumericImputationSummary, ...]
    coefficients: tuple[LinearRegressionCoefficient, ...]
    intercept: float
    training_row_count: int


@dataclass(frozen=True)
class QBPassingYardsLinearRegressionEvaluation:
    """Validation-only metrics and explicit QB-history cold-start subgroups."""

    overall_metrics: RegressionMetrics
    cold_start_row_count: int
    cold_start_row_rate: float
    non_cold_start_row_count: int
    non_cold_start_row_rate: float
    cold_start_metrics: RegressionMetrics | None
    non_cold_start_metrics: RegressionMetrics | None


@dataclass(frozen=True)
class HistoricalAverageComparison:
    """Validation comparison where positive improvements favor Linear Regression."""

    baseline_metrics: RegressionMetrics
    mae_improvement: float
    rmse_improvement: float
    r_squared_improvement: float
    beats_baseline_on_mae: bool


@dataclass(frozen=True)
class QBPassingYardsLinearRegressionValidation:
    """Train/validation-only Linear Regression result with no retained dataframes."""

    model: QBPassingYardsLinearRegressionModel
    validation: QBPassingYardsLinearRegressionEvaluation
    historical_average_comparison: HistoricalAverageComparison


def fit_qb_passing_yards_linear_regression(
    training_frame: pd.DataFrame,
) -> QBPassingYardsLinearRegressionModel:
    """Fit median preprocessing and default ordinary Linear Regression on train only.

    Numeric point-in-time features may be missing for legitimate cold starts and
    receive their training-partition median.  Existing boolean missing-history
    flags are passed through unchanged, so the estimator can distinguish those
    imputed rows.  No scaling is used: ordinary unregularized Linear Regression
    does not require it for predictive correctness.
    """

    _validate_frame(training_frame, "training", require_target=True)
    canonical_training = training_frame.sort_values(TARGET_KEY, kind="mergesort")
    features = _model_features(canonical_training)
    targets = _finite_targets(canonical_training[TARGET_COLUMN], "training")
    _reject_all_missing_numeric_features(features, "training")

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), list(NUMERIC_FEATURE_COLUMNS)),
            ("binary", "passthrough", list(BINARY_FEATURE_COLUMNS)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("linear_regression", LinearRegression()),
        ]
    )
    pipeline.fit(features, targets)

    transformed_names = tuple(
        str(name)
        for name in pipeline.named_steps["preprocessing"].get_feature_names_out()
    )
    if len(transformed_names) != len(set(transformed_names)):
        raise ValueError("Linear Regression preprocessing produced duplicate feature names")
    estimator = pipeline.named_steps["linear_regression"]
    coefficients = np.asarray(estimator.coef_, dtype=float).reshape(-1)
    if len(coefficients) != len(transformed_names):
        raise ValueError(
            "Linear Regression coefficient count does not match transformed feature count"
        )
    if not np.isfinite(coefficients).all() or not np.isfinite(estimator.intercept_):
        raise ValueError("Linear Regression fitted non-finite coefficients or intercept")

    imputer = pipeline.named_steps["preprocessing"].named_transformers_["numeric"]
    medians = np.asarray(imputer.statistics_, dtype=float)
    if len(medians) != len(NUMERIC_FEATURE_COLUMNS) or not np.isfinite(medians).all():
        raise ValueError("Training numeric imputation statistics must be finite")

    return QBPassingYardsLinearRegressionModel(
        pipeline=pipeline,
        feature_columns=tuple(FEATURE_COLUMNS),
        numeric_feature_columns=NUMERIC_FEATURE_COLUMNS,
        binary_feature_columns=BINARY_FEATURE_COLUMNS,
        numeric_imputation_summaries=tuple(
            NumericImputationSummary(feature_name, float(median))
            for feature_name, median in zip(NUMERIC_FEATURE_COLUMNS, medians, strict=True)
        ),
        coefficients=tuple(
            LinearRegressionCoefficient(name, float(coefficient))
            for name, coefficient in zip(transformed_names, coefficients, strict=True)
        ),
        intercept=float(estimator.intercept_),
        training_row_count=len(canonical_training),
    )


def predict_qb_passing_yards_linear_regression(
    model: QBPassingYardsLinearRegressionModel,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return one aligned prediction per input key without reading target values.

    Output preserves input row order, allowing direct audit against the supplied
    target keys.  The cold-start flag uses the same missing season-history
    condition as the historical-average baseline; it does not imply a separate
    model fallback.
    """

    if not isinstance(model, QBPassingYardsLinearRegressionModel):
        raise TypeError("model must be a QBPassingYardsLinearRegressionModel")
    _validate_model(model)
    _validate_frame(frame, "prediction", require_target=False)
    features = _model_features(frame)
    predictions = np.asarray(model.pipeline.predict(features), dtype=float).reshape(-1)
    if len(predictions) != len(frame):
        raise ValueError("Linear Regression prediction count does not match input row count")
    if not np.isfinite(predictions).all():
        raise ValueError("Linear Regression predictions must be finite")

    result_columns = [*TARGET_KEY]
    if TARGET_COLUMN in frame.columns:
        result_columns.append(TARGET_COLUMN)
    result = frame.loc[:, result_columns].copy(deep=True)
    result[PREDICTION_COLUMN] = predictions
    result[COLD_START_COLUMN] = frame[HISTORICAL_AVERAGE_COLUMN].isna().to_numpy(
        dtype=bool
    )
    return result.reset_index(drop=True)


def evaluate_qb_passing_yards_linear_regression(
    model: QBPassingYardsLinearRegressionModel,
    validation_frame: pd.DataFrame,
) -> QBPassingYardsLinearRegressionEvaluation:
    """Predict and score validation data with the shared regression evaluator."""

    _validate_frame(validation_frame, "validation", require_target=True)
    canonical_validation = validation_frame.sort_values(TARGET_KEY, kind="mergesort")
    predictions = predict_qb_passing_yards_linear_regression(model, canonical_validation)
    overall = _evaluate_predictions(canonical_validation, predictions)
    cold_start_mask = predictions[COLD_START_COLUMN]
    cold_start_count = int(cold_start_mask.sum())
    non_cold_start_count = len(predictions) - cold_start_count
    return QBPassingYardsLinearRegressionEvaluation(
        overall_metrics=overall,
        cold_start_row_count=cold_start_count,
        cold_start_row_rate=cold_start_count / len(predictions),
        non_cold_start_row_count=non_cold_start_count,
        non_cold_start_row_rate=non_cold_start_count / len(predictions),
        cold_start_metrics=_subgroup_metrics(
            canonical_validation, predictions, cold_start_mask
        ),
        non_cold_start_metrics=_subgroup_metrics(
            canonical_validation, predictions, ~cold_start_mask
        ),
    )


def train_and_validate_qb_passing_yards_linear_regression(
    dataset_split: QBPassingYardsDatasetSplit,
) -> QBPassingYardsLinearRegressionValidation:
    """Fit train only and compare validation only; never access the test partition."""

    if not isinstance(dataset_split, QBPassingYardsDatasetSplit):
        raise TypeError("dataset_split must be a QBPassingYardsDatasetSplit")

    model = fit_qb_passing_yards_linear_regression(dataset_split.train)
    validation = evaluate_qb_passing_yards_linear_regression(model, dataset_split.validation)
    baseline = fit_qb_passing_yards_baseline(dataset_split.train)
    canonical_validation = dataset_split.validation.sort_values(
        TARGET_KEY, kind="mergesort"
    )
    baseline_predictions = predict_qb_passing_yards_baseline(
        baseline, canonical_validation
    )
    baseline_metrics = evaluate_qb_passing_yards_predictions(
        canonical_validation, baseline_predictions
    )
    return QBPassingYardsLinearRegressionValidation(
        model=model,
        validation=validation,
        historical_average_comparison=HistoricalAverageComparison(
            baseline_metrics=baseline_metrics,
            mae_improvement=baseline_metrics.mae - validation.overall_metrics.mae,
            rmse_improvement=baseline_metrics.rmse - validation.overall_metrics.rmse,
            r_squared_improvement=(
                validation.overall_metrics.r_squared - baseline_metrics.r_squared
            ),
            beats_baseline_on_mae=validation.overall_metrics.mae < baseline_metrics.mae,
        ),
    )


def _validate_frame(data: pd.DataFrame, name: str, *, require_target: bool) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"{name.capitalize()} frame must be a pandas DataFrame")
    required = [*TARGET_KEY, *FEATURE_COLUMNS]
    if require_target:
        required.append(TARGET_COLUMN)
    missing = sorted(set(required).difference(data.columns))
    if missing:
        raise ValueError(f"{name.capitalize()} frame is missing required columns: {missing}")
    if data.empty:
        raise ValueError(f"{name.capitalize()} frame must not be empty")
    if data[TARGET_KEY].isna().any().any():
        raise ValueError(f"{name.capitalize()} frame has missing target-key values")
    if data.duplicated(TARGET_KEY).any():
        raise ValueError(f"{name.capitalize()} frame has duplicate target keys")
    _validate_feature_values(data, name)
    if require_target:
        _finite_targets(data[TARGET_COLUMN], name)


def _validate_feature_values(data: pd.DataFrame, name: str) -> None:
    for feature in NUMERIC_FEATURE_COLUMNS:
        values = data[feature]
        numeric = pd.to_numeric(values, errors="coerce")
        invalid = ~values.isna() & (numeric.isna() | ~np.isfinite(numeric))
        if invalid.any():
            raise ValueError(
                f"{name.capitalize()} frame has non-finite predictive values in {feature}"
            )
    for feature in BINARY_FEATURE_COLUMNS:
        values = data[feature]
        if not pd.api.types.is_bool_dtype(values) or values.isna().any():
            raise ValueError(
                f"{name.capitalize()} frame has invalid missing-history flags in {feature}"
            )
    for flag in BINARY_FEATURE_COLUMNS:
        prefix = flag.removesuffix("_missing_history")
        average_features = [
            feature
            for feature in FEATURE_COLUMNS
            if feature.startswith(f"{prefix}_season_passing_yards")
        ]
        if len(average_features) != 1 or not data[flag].eq(
            data[average_features[0]].isna()
        ).all():
            raise ValueError(
                f"{name.capitalize()} frame has inconsistent missing-history flags"
            )


def _model_features(data: pd.DataFrame) -> pd.DataFrame:
    features = data.loc[:, FEATURE_COLUMNS].copy(deep=True)
    for feature in NUMERIC_FEATURE_COLUMNS:
        features[feature] = pd.to_numeric(features[feature], errors="raise").astype(float)
    return features


def _finite_targets(values: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric).all():
        raise ValueError(f"{name.capitalize()} frame has non-finite target_passing_yards values")
    return numeric.astype(float)


def _reject_all_missing_numeric_features(features: pd.DataFrame, name: str) -> None:
    all_missing = [feature for feature in NUMERIC_FEATURE_COLUMNS if features[feature].isna().all()]
    if all_missing:
        raise ValueError(
            f"{name.capitalize()} frame has numeric predictive features with no usable values: "
            f"{all_missing}"
        )


def _validate_model(model: QBPassingYardsLinearRegressionModel) -> None:
    if model.feature_columns != tuple(FEATURE_COLUMNS):
        raise ValueError("Linear Regression model feature contract does not match QB dataset")
    if model.numeric_feature_columns != NUMERIC_FEATURE_COLUMNS:
        raise ValueError("Linear Regression model numeric feature contract does not match QB dataset")
    if model.binary_feature_columns != BINARY_FEATURE_COLUMNS:
        raise ValueError("Linear Regression model binary feature contract does not match QB dataset")
    if model.training_row_count < 1 or not np.isfinite(model.intercept):
        raise ValueError("Linear Regression model has invalid fitted metadata")


def _evaluate_predictions(
    actual_frame: pd.DataFrame, predictions: pd.DataFrame
) -> RegressionMetrics:
    evaluator_predictions = predictions.loc[:, TARGET_KEY].copy(deep=True)
    evaluator_predictions[BASELINE_PREDICTION_COLUMN] = predictions[
        PREDICTION_COLUMN
    ].to_numpy()
    return evaluate_qb_passing_yards_predictions(actual_frame, evaluator_predictions)


def _subgroup_metrics(
    validation_frame: pd.DataFrame,
    predictions: pd.DataFrame,
    mask: pd.Series,
) -> RegressionMetrics | None:
    if not mask.any():
        return None
    keys = predictions.loc[mask, TARGET_KEY]
    validation_rows = validation_frame.merge(
        keys, on=TARGET_KEY, how="inner", validate="one_to_one", sort=False
    )
    return _evaluate_predictions(validation_rows, predictions.loc[mask])
