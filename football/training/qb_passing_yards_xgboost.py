"""Fixed, leakage-safe XGBoost benchmark for QB passing yards.

The completed dataset supplies pregame, point-in-time features. This module
fits training-only median imputation and one predetermined XGBoost
configuration, then evaluates validation. Its train-and-validate entry point
intentionally never accesses the chronological test partition.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

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
from football.training.qb_passing_yards_gradient_boosting import (
    evaluate_qb_passing_yards_gradient_boosting,
    fit_qb_passing_yards_gradient_boosting,
)
from football.training.qb_passing_yards_linear_regression import (
    NumericImputationSummary,
    evaluate_qb_passing_yards_linear_regression,
    fit_qb_passing_yards_linear_regression,
)
from football.training.qb_passing_yards_random_forest import (
    evaluate_qb_passing_yards_random_forest,
    fit_qb_passing_yards_random_forest,
)
from football.training.split_qb_passing_yards_dataset import (
    QBPassingYardsDatasetSplit,
)


PREDICTION_COLUMN = "xgboost_prediction"
COLD_START_COLUMN = "is_qb_history_cold_start"
IMPORTANCE_TYPE = "gain"

# These categories are derived from the builder's sole public model-input
# contract rather than maintained as a separate hand-written feature list.
BINARY_FEATURE_COLUMNS = tuple(
    feature for feature in FEATURE_COLUMNS if feature.endswith("_missing_history")
)
NUMERIC_FEATURE_COLUMNS = tuple(
    feature for feature in FEATURE_COLUMNS if feature not in BINARY_FEATURE_COLUMNS
)
HISTORY_SAMPLE_SIZE_COLUMNS = tuple(
    feature for feature in FEATURE_COLUMNS if feature.endswith("_history_games")
)


@dataclass(frozen=True)
class XGBoostParameters:
    """The one predetermined, non-tuned CPU XGBoost configuration."""

    objective: str = "reg:squarederror"
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 3
    min_child_weight: int = 1
    gamma: float = 0.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    tree_method: str = "hist"
    random_state: int = 42
    n_jobs: int = 1
    verbosity: int = 0
    importance_type: str = IMPORTANCE_TYPE


@dataclass(frozen=True)
class XGBoostFeatureImportance:
    """One normalized XGBoost gain importance, not causal feature value."""

    feature_name: str
    importance: float


@dataclass(frozen=True)
class QBPassingYardsXGBoostModel:
    """Training-only preprocessing and XGBoost estimator, with no frames."""

    pipeline: Pipeline = field(repr=False, compare=False)
    parameters: XGBoostParameters
    feature_columns: tuple[str, ...]
    numeric_feature_columns: tuple[str, ...]
    binary_feature_columns: tuple[str, ...]
    numeric_imputation_summaries: tuple[NumericImputationSummary, ...]
    importance_type: str
    feature_importances: tuple[XGBoostFeatureImportance, ...]
    training_row_count: int


@dataclass(frozen=True)
class QBPassingYardsXGBoostEvaluation:
    """Validation metrics and established QB-history cold-start groups."""

    overall_metrics: RegressionMetrics
    cold_start_row_count: int
    cold_start_row_rate: float
    non_cold_start_row_count: int
    non_cold_start_row_rate: float
    cold_start_metrics: RegressionMetrics | None
    non_cold_start_metrics: RegressionMetrics | None


@dataclass(frozen=True)
class XGBoostComparison:
    """Metrics relative to one model; positive differences favor XGBoost."""

    comparison_model_name: str
    comparison_metrics: RegressionMetrics
    mae_improvement: float
    rmse_improvement: float
    r_squared_improvement: float
    xgboost_wins_on_mae: bool


@dataclass(frozen=True)
class QBPassingYardsXGBoostValidation:
    """Complete train/validation-only benchmark result without dataframes."""

    model: QBPassingYardsXGBoostModel
    validation: QBPassingYardsXGBoostEvaluation
    historical_average_comparison: XGBoostComparison
    linear_regression_comparison: XGBoostComparison
    random_forest_comparison: XGBoostComparison
    gradient_boosting_comparison: XGBoostComparison


def fit_qb_passing_yards_xgboost(
    training_frame: pd.DataFrame,
) -> QBPassingYardsXGBoostModel:
    """Fit fixed XGBoost parameters with training-only median imputation.

    Permitted cold-start numeric history is median-imputed from training rows.
    Existing missing-history flags pass through unchanged. No scaling is used,
    and validation/test values cannot affect the fitted pipeline.
    """

    _validate_frame(training_frame, "training", require_target=True)
    canonical_training = training_frame.sort_values(TARGET_KEY, kind="mergesort")
    features = _model_features(canonical_training)
    targets = _finite_targets(canonical_training[TARGET_COLUMN], "training")
    _reject_all_missing_numeric_features(features, "training")

    parameters = XGBoostParameters()
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), list(NUMERIC_FEATURE_COLUMNS)),
            ("binary", "passthrough", list(BINARY_FEATURE_COLUMNS)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    estimator = XGBRegressor(
        objective=parameters.objective,
        n_estimators=parameters.n_estimators,
        learning_rate=parameters.learning_rate,
        max_depth=parameters.max_depth,
        min_child_weight=parameters.min_child_weight,
        gamma=parameters.gamma,
        subsample=parameters.subsample,
        colsample_bytree=parameters.colsample_bytree,
        reg_alpha=parameters.reg_alpha,
        reg_lambda=parameters.reg_lambda,
        tree_method=parameters.tree_method,
        random_state=parameters.random_state,
        n_jobs=parameters.n_jobs,
        verbosity=parameters.verbosity,
        importance_type=parameters.importance_type,
    )
    pipeline = Pipeline(steps=[("preprocessing", preprocessor), ("xgboost", estimator)])
    pipeline.fit(features, targets)

    fitted_preprocessor = pipeline.named_steps["preprocessing"]
    fitted_estimator = pipeline.named_steps["xgboost"]
    transformed_names = tuple(
        str(name) for name in fitted_preprocessor.get_feature_names_out()
    )
    if len(transformed_names) != len(set(transformed_names)):
        raise ValueError("XGBoost preprocessing produced duplicate feature names")
    imputer = fitted_preprocessor.named_transformers_["numeric"]
    medians = np.asarray(imputer.statistics_, dtype=float)
    if len(medians) != len(NUMERIC_FEATURE_COLUMNS) or not np.isfinite(medians).all():
        raise ValueError("Training numeric imputation statistics must be finite")

    raw_importances = np.asarray(
        fitted_estimator.feature_importances_, dtype=float
    ).reshape(-1)
    importances = _normalized_gain_importances(transformed_names, raw_importances)
    ordered_importances = tuple(
        XGBoostFeatureImportance(name, float(importance))
        for name, importance in sorted(
            zip(transformed_names, importances, strict=True),
            key=lambda item: (-float(item[1]), item[0]),
        )
    )
    return QBPassingYardsXGBoostModel(
        pipeline=pipeline,
        parameters=parameters,
        feature_columns=tuple(FEATURE_COLUMNS),
        numeric_feature_columns=NUMERIC_FEATURE_COLUMNS,
        binary_feature_columns=BINARY_FEATURE_COLUMNS,
        numeric_imputation_summaries=tuple(
            NumericImputationSummary(name, float(median))
            for name, median in zip(NUMERIC_FEATURE_COLUMNS, medians, strict=True)
        ),
        importance_type=parameters.importance_type,
        feature_importances=ordered_importances,
        training_row_count=len(canonical_training),
    )


def predict_qb_passing_yards_xgboost(
    model: QBPassingYardsXGBoostModel,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return one finite prediction per supplied key without target reads."""

    if not isinstance(model, QBPassingYardsXGBoostModel):
        raise TypeError("model must be a QBPassingYardsXGBoostModel")
    _validate_model(model)
    _validate_frame(frame, "prediction", require_target=False)
    predictions = np.asarray(model.pipeline.predict(_model_features(frame)), dtype=float)
    if len(predictions) != len(frame):
        raise ValueError("XGBoost prediction count does not match input row count")
    if not np.isfinite(predictions).all():
        raise ValueError("XGBoost predictions must be finite")

    result_columns = [*TARGET_KEY]
    if TARGET_COLUMN in frame.columns:
        result_columns.append(TARGET_COLUMN)
    result = frame.loc[:, result_columns].copy(deep=True)
    result[PREDICTION_COLUMN] = predictions
    result[COLD_START_COLUMN] = frame[HISTORICAL_AVERAGE_COLUMN].isna().to_numpy(
        dtype=bool
    )
    return result.reset_index(drop=True)


def evaluate_qb_passing_yards_xgboost(
    model: QBPassingYardsXGBoostModel,
    validation_frame: pd.DataFrame,
) -> QBPassingYardsXGBoostEvaluation:
    """Score validation only through the existing public regression evaluator."""

    _validate_frame(validation_frame, "validation", require_target=True)
    canonical_validation = validation_frame.sort_values(TARGET_KEY, kind="mergesort")
    predictions = predict_qb_passing_yards_xgboost(model, canonical_validation)
    cold_start_mask = predictions[COLD_START_COLUMN]
    cold_start_count = int(cold_start_mask.sum())
    non_cold_start_count = len(predictions) - cold_start_count
    return QBPassingYardsXGBoostEvaluation(
        overall_metrics=_evaluate_predictions(canonical_validation, predictions),
        cold_start_row_count=cold_start_count,
        cold_start_row_rate=cold_start_count / len(predictions),
        non_cold_start_row_count=non_cold_start_count,
        non_cold_start_row_rate=non_cold_start_count / len(predictions),
        cold_start_metrics=_subgroup_metrics(canonical_validation, predictions, cold_start_mask),
        non_cold_start_metrics=_subgroup_metrics(
            canonical_validation, predictions, ~cold_start_mask
        ),
    )


def train_and_validate_qb_passing_yards_xgboost(
    dataset_split: QBPassingYardsDatasetSplit,
) -> QBPassingYardsXGBoostValidation:
    """Fit and compare train/validation only; this never accesses test."""

    if not isinstance(dataset_split, QBPassingYardsDatasetSplit):
        raise TypeError("dataset_split must be a QBPassingYardsDatasetSplit")

    model = fit_qb_passing_yards_xgboost(dataset_split.train)
    validation = evaluate_qb_passing_yards_xgboost(model, dataset_split.validation)
    canonical_validation = dataset_split.validation.sort_values(TARGET_KEY, kind="mergesort")

    baseline = fit_qb_passing_yards_baseline(dataset_split.train)
    baseline_metrics = evaluate_qb_passing_yards_predictions(
        canonical_validation,
        predict_qb_passing_yards_baseline(baseline, canonical_validation),
    )
    linear_metrics = evaluate_qb_passing_yards_linear_regression(
        fit_qb_passing_yards_linear_regression(dataset_split.train), canonical_validation
    ).overall_metrics
    random_forest_metrics = evaluate_qb_passing_yards_random_forest(
        fit_qb_passing_yards_random_forest(dataset_split.train), canonical_validation
    ).overall_metrics
    gradient_boosting_metrics = evaluate_qb_passing_yards_gradient_boosting(
        fit_qb_passing_yards_gradient_boosting(dataset_split.train), canonical_validation
    ).overall_metrics

    return QBPassingYardsXGBoostValidation(
        model=model,
        validation=validation,
        historical_average_comparison=_comparison("Historical average", baseline_metrics, validation.overall_metrics),
        linear_regression_comparison=_comparison("Linear Regression", linear_metrics, validation.overall_metrics),
        random_forest_comparison=_comparison("Random Forest", random_forest_metrics, validation.overall_metrics),
        gradient_boosting_comparison=_comparison("Gradient Boosting", gradient_boosting_metrics, validation.overall_metrics),
    )


def _comparison(
    name: str, comparison: RegressionMetrics, xgboost: RegressionMetrics
) -> XGBoostComparison:
    return XGBoostComparison(
        comparison_model_name=name,
        comparison_metrics=comparison,
        mae_improvement=comparison.mae - xgboost.mae,
        rmse_improvement=comparison.rmse - xgboost.rmse,
        r_squared_improvement=xgboost.r_squared - comparison.r_squared,
        xgboost_wins_on_mae=xgboost.mae < comparison.mae,
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
            raise ValueError(f"{name.capitalize()} frame has non-finite predictive values in {feature}")
    for feature in HISTORY_SAMPLE_SIZE_COLUMNS:
        values = pd.to_numeric(data[feature], errors="coerce")
        invalid = values.isna() | ~np.isfinite(values) | values.lt(0) | values.mod(1).ne(0)
        if invalid.any():
            raise ValueError(f"{name.capitalize()} frame has invalid history sample sizes in {feature}")
    for feature in BINARY_FEATURE_COLUMNS:
        values = data[feature]
        if not pd.api.types.is_bool_dtype(values) or values.isna().any():
            raise ValueError(f"{name.capitalize()} frame has invalid missing-history flags in {feature}")
        prefix = feature.removesuffix("_missing_history")
        average_features = [
            candidate for candidate in FEATURE_COLUMNS
            if candidate.startswith(f"{prefix}_season_passing_yards")
        ]
        history_feature = f"{prefix}_season_history_games"
        if (
            len(average_features) != 1
            or not values.eq(data[average_features[0]].isna()).all()
            or not values.eq(data[history_feature].eq(0)).all()
        ):
            raise ValueError(f"{name.capitalize()} frame has inconsistent missing-history flags")


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
    all_missing = [
        feature for feature in NUMERIC_FEATURE_COLUMNS if features[feature].isna().all()
    ]
    if all_missing:
        raise ValueError(
            f"{name.capitalize()} frame has numeric predictive features with no usable values: {all_missing}"
        )


def _normalized_gain_importances(
    feature_names: tuple[str, ...], importances: np.ndarray
) -> np.ndarray:
    if len(feature_names) != len(importances):
        raise ValueError("XGBoost importance count does not match transformed feature count")
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("XGBoost preprocessing produced duplicate feature names")
    if not np.isfinite(importances).all() or (importances < 0).any():
        raise ValueError("XGBoost feature importances must be finite and nonnegative")
    importance_sum = float(importances.sum())
    if importance_sum <= 0:
        raise ValueError("XGBoost gain feature importances must have a positive sum")
    normalized = importances / importance_sum
    if not np.isclose(float(normalized.sum()), 1.0, rtol=1e-9, atol=1e-12):
        raise ValueError("Normalized XGBoost gain feature importances must sum to one")
    return normalized


def _validate_model(model: QBPassingYardsXGBoostModel) -> None:
    if model.feature_columns != tuple(FEATURE_COLUMNS):
        raise ValueError("XGBoost model feature contract does not match QB dataset")
    if model.numeric_feature_columns != NUMERIC_FEATURE_COLUMNS:
        raise ValueError("XGBoost model numeric feature contract does not match QB dataset")
    if model.binary_feature_columns != BINARY_FEATURE_COLUMNS:
        raise ValueError("XGBoost model binary feature contract does not match QB dataset")
    if model.importance_type != IMPORTANCE_TYPE or model.training_row_count < 1:
        raise ValueError("XGBoost model has invalid fitted metadata")


def _evaluate_predictions(actual_frame: pd.DataFrame, predictions: pd.DataFrame) -> RegressionMetrics:
    evaluator_predictions = predictions.loc[:, TARGET_KEY].copy(deep=True)
    evaluator_predictions[BASELINE_PREDICTION_COLUMN] = predictions[PREDICTION_COLUMN].to_numpy()
    return evaluate_qb_passing_yards_predictions(actual_frame, evaluator_predictions)


def _subgroup_metrics(
    validation_frame: pd.DataFrame, predictions: pd.DataFrame, mask: pd.Series
) -> RegressionMetrics | None:
    if not mask.any():
        return None
    keys = predictions.loc[mask, TARGET_KEY]
    validation_rows = validation_frame.merge(
        keys, on=TARGET_KEY, how="inner", validate="one_to_one", sort=False
    )
    return _evaluate_predictions(validation_rows, predictions.loc[mask])
