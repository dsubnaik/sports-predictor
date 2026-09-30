"""Fixed, leakage-safe Random Forest benchmark for QB passing yards.

The completed dataset supplies point-in-time features. This module fits median
imputation and one predetermined Random Forest configuration using a supplied
training partition only, then evaluates a supplied validation partition. Its
train-and-validate entry point deliberately never accesses the chronological
test partition.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
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
from football.training.qb_passing_yards_linear_regression import (
    NumericImputationSummary,
    evaluate_qb_passing_yards_linear_regression,
    fit_qb_passing_yards_linear_regression,
)
from football.training.split_qb_passing_yards_dataset import (
    QBPassingYardsDatasetSplit,
)


PREDICTION_COLUMN = "random_forest_prediction"
COLD_START_COLUMN = "is_qb_history_cold_start"

# Derive categories from the established public builder contract. No separate
# hand-maintained model feature list is introduced.
BINARY_FEATURE_COLUMNS = tuple(
    feature for feature in FEATURE_COLUMNS if feature.endswith("_missing_history")
)
NUMERIC_FEATURE_COLUMNS = tuple(
    feature for feature in FEATURE_COLUMNS if feature not in BINARY_FEATURE_COLUMNS
)


@dataclass(frozen=True)
class RandomForestParameters:
    """The one predetermined, non-tuned Random Forest benchmark configuration."""

    n_estimators: int = 500
    random_state: int = 42
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: float = 1.0
    bootstrap: bool = True
    n_jobs: int = 1


@dataclass(frozen=True)
class RandomForestFeatureImportance:
    """One impurity-based importance; it is not causal or standalone value."""

    feature_name: str
    importance: float


@dataclass(frozen=True)
class QBPassingYardsRandomForestModel:
    """Training-only preprocessing and estimator together, with no input frames."""

    pipeline: Pipeline = field(repr=False, compare=False)
    parameters: RandomForestParameters
    feature_columns: tuple[str, ...]
    numeric_feature_columns: tuple[str, ...]
    binary_feature_columns: tuple[str, ...]
    numeric_imputation_summaries: tuple[NumericImputationSummary, ...]
    feature_importances: tuple[RandomForestFeatureImportance, ...]
    training_row_count: int


@dataclass(frozen=True)
class QBPassingYardsRandomForestEvaluation:
    """Validation-only metrics and the established QB-history cold-start groups."""

    overall_metrics: RegressionMetrics
    cold_start_row_count: int
    cold_start_row_rate: float
    non_cold_start_row_count: int
    non_cold_start_row_rate: float
    cold_start_metrics: RegressionMetrics | None
    non_cold_start_metrics: RegressionMetrics | None


@dataclass(frozen=True)
class RandomForestComparison:
    """Metrics relative to one comparison model; positive differences favor RF."""

    comparison_model_name: str
    comparison_metrics: RegressionMetrics
    mae_improvement: float
    rmse_improvement: float
    r_squared_improvement: float
    random_forest_wins_on_mae: bool


@dataclass(frozen=True)
class QBPassingYardsRandomForestValidation:
    """Complete train/validation-only RF benchmark result without dataframes."""

    model: QBPassingYardsRandomForestModel
    validation: QBPassingYardsRandomForestEvaluation
    historical_average_comparison: RandomForestComparison
    linear_regression_comparison: RandomForestComparison


def fit_qb_passing_yards_random_forest(
    training_frame: pd.DataFrame,
) -> QBPassingYardsRandomForestModel:
    """Fit fixed RF parameters and training-only median preprocessing.

    Numeric historical values may be absent for legitimate cold starts and are
    median-imputed from training rows. Existing boolean missing-history flags
    pass through unchanged. No scaling is used or needed for this tree model.
    """

    _validate_frame(training_frame, "training", require_target=True)
    canonical_training = training_frame.sort_values(TARGET_KEY, kind="mergesort")
    features = _model_features(canonical_training)
    targets = _finite_targets(canonical_training[TARGET_COLUMN], "training")
    _reject_all_missing_numeric_features(features, "training")

    parameters = RandomForestParameters()
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), list(NUMERIC_FEATURE_COLUMNS)),
            ("binary", "passthrough", list(BINARY_FEATURE_COLUMNS)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    estimator = RandomForestRegressor(
        n_estimators=parameters.n_estimators,
        random_state=parameters.random_state,
        max_depth=parameters.max_depth,
        min_samples_split=parameters.min_samples_split,
        min_samples_leaf=parameters.min_samples_leaf,
        max_features=parameters.max_features,
        bootstrap=parameters.bootstrap,
        n_jobs=parameters.n_jobs,
    )
    pipeline = Pipeline(
        steps=[("preprocessing", preprocessor), ("random_forest", estimator)]
    )
    pipeline.fit(features, targets)

    transformed_names = tuple(
        str(name)
        for name in pipeline.named_steps["preprocessing"].get_feature_names_out()
    )
    if len(transformed_names) != len(set(transformed_names)):
        raise ValueError("Random Forest preprocessing produced duplicate feature names")

    imputer = pipeline.named_steps["preprocessing"].named_transformers_["numeric"]
    medians = np.asarray(imputer.statistics_, dtype=float)
    if len(medians) != len(NUMERIC_FEATURE_COLUMNS) or not np.isfinite(medians).all():
        raise ValueError("Training numeric imputation statistics must be finite")

    importances = np.asarray(
        pipeline.named_steps["random_forest"].feature_importances_, dtype=float
    ).reshape(-1)
    _validate_feature_importances(transformed_names, importances)
    ordered_importances = tuple(
        RandomForestFeatureImportance(name, float(importance))
        for name, importance in sorted(
            zip(transformed_names, importances, strict=True),
            key=lambda item: (-float(item[1]), item[0]),
        )
    )
    return QBPassingYardsRandomForestModel(
        pipeline=pipeline,
        parameters=parameters,
        feature_columns=tuple(FEATURE_COLUMNS),
        numeric_feature_columns=NUMERIC_FEATURE_COLUMNS,
        binary_feature_columns=BINARY_FEATURE_COLUMNS,
        numeric_imputation_summaries=tuple(
            NumericImputationSummary(name, float(median))
            for name, median in zip(NUMERIC_FEATURE_COLUMNS, medians, strict=True)
        ),
        feature_importances=ordered_importances,
        training_row_count=len(canonical_training),
    )


def predict_qb_passing_yards_random_forest(
    model: QBPassingYardsRandomForestModel,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return one finite prediction per supplied target key without target reads."""

    if not isinstance(model, QBPassingYardsRandomForestModel):
        raise TypeError("model must be a QBPassingYardsRandomForestModel")
    _validate_model(model)
    _validate_frame(frame, "prediction", require_target=False)
    predictions = np.asarray(model.pipeline.predict(_model_features(frame)), dtype=float)
    if len(predictions) != len(frame):
        raise ValueError("Random Forest prediction count does not match input row count")
    if not np.isfinite(predictions).all():
        raise ValueError("Random Forest predictions must be finite")

    result_columns = [*TARGET_KEY]
    if TARGET_COLUMN in frame.columns:
        result_columns.append(TARGET_COLUMN)
    result = frame.loc[:, result_columns].copy(deep=True)
    result[PREDICTION_COLUMN] = predictions
    result[COLD_START_COLUMN] = frame[HISTORICAL_AVERAGE_COLUMN].isna().to_numpy(
        dtype=bool
    )
    return result.reset_index(drop=True)


def evaluate_qb_passing_yards_random_forest(
    model: QBPassingYardsRandomForestModel,
    validation_frame: pd.DataFrame,
) -> QBPassingYardsRandomForestEvaluation:
    """Score validation only through the existing shared regression evaluator."""

    _validate_frame(validation_frame, "validation", require_target=True)
    canonical_validation = validation_frame.sort_values(TARGET_KEY, kind="mergesort")
    predictions = predict_qb_passing_yards_random_forest(model, canonical_validation)
    cold_start_mask = predictions[COLD_START_COLUMN]
    cold_start_count = int(cold_start_mask.sum())
    non_cold_start_count = len(predictions) - cold_start_count
    return QBPassingYardsRandomForestEvaluation(
        overall_metrics=_evaluate_predictions(canonical_validation, predictions),
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


def train_and_validate_qb_passing_yards_random_forest(
    dataset_split: QBPassingYardsDatasetSplit,
) -> QBPassingYardsRandomForestValidation:
    """Fit and compare on train/validation only; intentionally never access test."""

    if not isinstance(dataset_split, QBPassingYardsDatasetSplit):
        raise TypeError("dataset_split must be a QBPassingYardsDatasetSplit")

    model = fit_qb_passing_yards_random_forest(dataset_split.train)
    validation = evaluate_qb_passing_yards_random_forest(model, dataset_split.validation)
    canonical_validation = dataset_split.validation.sort_values(
        TARGET_KEY, kind="mergesort"
    )

    baseline = fit_qb_passing_yards_baseline(dataset_split.train)
    baseline_metrics = evaluate_qb_passing_yards_predictions(
        canonical_validation,
        predict_qb_passing_yards_baseline(baseline, canonical_validation),
    )
    linear_model = fit_qb_passing_yards_linear_regression(dataset_split.train)
    linear_metrics = evaluate_qb_passing_yards_linear_regression(
        linear_model, canonical_validation
    ).overall_metrics

    return QBPassingYardsRandomForestValidation(
        model=model,
        validation=validation,
        historical_average_comparison=_comparison(
            "Historical average", baseline_metrics, validation.overall_metrics
        ),
        linear_regression_comparison=_comparison(
            "Linear Regression", linear_metrics, validation.overall_metrics
        ),
    )


def _comparison(
    name: str, comparison: RegressionMetrics, random_forest: RegressionMetrics
) -> RandomForestComparison:
    return RandomForestComparison(
        comparison_model_name=name,
        comparison_metrics=comparison,
        mae_improvement=comparison.mae - random_forest.mae,
        rmse_improvement=comparison.rmse - random_forest.rmse,
        r_squared_improvement=random_forest.r_squared - comparison.r_squared,
        random_forest_wins_on_mae=random_forest.mae < comparison.mae,
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
        prefix = feature.removesuffix("_missing_history")
        average_features = [
            candidate
            for candidate in FEATURE_COLUMNS
            if candidate.startswith(f"{prefix}_season_passing_yards")
        ]
        if len(average_features) != 1 or not values.eq(
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
    all_missing = [
        feature for feature in NUMERIC_FEATURE_COLUMNS if features[feature].isna().all()
    ]
    if all_missing:
        raise ValueError(
            f"{name.capitalize()} frame has numeric predictive features with no usable values: "
            f"{all_missing}"
        )


def _validate_feature_importances(
    feature_names: tuple[str, ...], importances: np.ndarray
) -> None:
    if len(feature_names) != len(importances):
        raise ValueError("Random Forest importance count does not match transformed feature count")
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("Random Forest preprocessing produced duplicate feature names")
    if not np.isfinite(importances).all() or (importances < 0).any():
        raise ValueError("Random Forest feature importances must be finite and nonnegative")
    if not np.isclose(float(importances.sum()), 1.0, rtol=1e-9, atol=1e-12):
        raise ValueError("Random Forest feature importances must sum to one")


def _validate_model(model: QBPassingYardsRandomForestModel) -> None:
    if model.feature_columns != tuple(FEATURE_COLUMNS):
        raise ValueError("Random Forest model feature contract does not match QB dataset")
    if model.numeric_feature_columns != NUMERIC_FEATURE_COLUMNS:
        raise ValueError("Random Forest model numeric feature contract does not match QB dataset")
    if model.binary_feature_columns != BINARY_FEATURE_COLUMNS:
        raise ValueError("Random Forest model binary feature contract does not match QB dataset")
    if model.training_row_count < 1:
        raise ValueError("Random Forest model has invalid fitted metadata")


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
