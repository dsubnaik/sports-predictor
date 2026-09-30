"""Fixed, leakage-safe Gradient Boosting benchmark for QB passing yards.

The completed dataset supplies pregame, point-in-time features.  This module
fits median imputation and one predetermined Gradient Boosting configuration
on a supplied training partition only, then evaluates validation.  Its
train-and-validate entry point intentionally never accesses the held-out test
partition.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
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
from football.training.qb_passing_yards_random_forest import (
    evaluate_qb_passing_yards_random_forest,
    fit_qb_passing_yards_random_forest,
)
from football.training.split_qb_passing_yards_dataset import (
    QBPassingYardsDatasetSplit,
)


PREDICTION_COLUMN = "gradient_boosting_prediction"
COLD_START_COLUMN = "is_qb_history_cold_start"

# Derive categories from the established builder contract, rather than keeping
# a second model-maintained list of predictive inputs.
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
class GradientBoostingParameters:
    """The one predetermined, non-tuned Gradient Boosting configuration."""

    loss: str = "squared_error"
    learning_rate: float = 0.05
    n_estimators: int = 200
    subsample: float = 1.0
    criterion: str = "friedman_mse"
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_depth: int = 3
    max_features: None = None
    random_state: int = 42


@dataclass(frozen=True)
class GradientBoostingFeatureImportance:
    """One impurity-based importance; it is not causal or standalone value."""

    feature_name: str
    importance: float


@dataclass(frozen=True)
class QBPassingYardsGradientBoostingModel:
    """Training-only preprocessing and estimator together, with no dataframes."""

    pipeline: Pipeline = field(repr=False, compare=False)
    parameters: GradientBoostingParameters
    feature_columns: tuple[str, ...]
    numeric_feature_columns: tuple[str, ...]
    binary_feature_columns: tuple[str, ...]
    numeric_imputation_summaries: tuple[NumericImputationSummary, ...]
    feature_importances: tuple[GradientBoostingFeatureImportance, ...]
    training_row_count: int


@dataclass(frozen=True)
class QBPassingYardsGradientBoostingEvaluation:
    """Validation metrics with the established QB-history cold-start groups."""

    overall_metrics: RegressionMetrics
    cold_start_row_count: int
    cold_start_row_rate: float
    non_cold_start_row_count: int
    non_cold_start_row_rate: float
    cold_start_metrics: RegressionMetrics | None
    non_cold_start_metrics: RegressionMetrics | None


@dataclass(frozen=True)
class GradientBoostingComparison:
    """Metrics relative to one model; positive differences favor boosting."""

    comparison_model_name: str
    comparison_metrics: RegressionMetrics
    mae_improvement: float
    rmse_improvement: float
    r_squared_improvement: float
    gradient_boosting_wins_on_mae: bool


@dataclass(frozen=True)
class QBPassingYardsGradientBoostingValidation:
    """Complete train/validation-only benchmark result without dataframes."""

    model: QBPassingYardsGradientBoostingModel
    validation: QBPassingYardsGradientBoostingEvaluation
    historical_average_comparison: GradientBoostingComparison
    linear_regression_comparison: GradientBoostingComparison
    random_forest_comparison: GradientBoostingComparison


def fit_qb_passing_yards_gradient_boosting(
    training_frame: pd.DataFrame,
) -> QBPassingYardsGradientBoostingModel:
    """Fit fixed boosting parameters and training-only median preprocessing.

    Permitted cold-start numeric history is median-imputed from training rows.
    Existing boolean missing-history flags pass through unchanged.  No scaling
    is used and no training row is dropped.
    """

    _validate_frame(training_frame, "training", require_target=True)
    canonical_training = training_frame.sort_values(TARGET_KEY, kind="mergesort")
    features = _model_features(canonical_training)
    targets = _finite_targets(canonical_training[TARGET_COLUMN], "training")
    _reject_all_missing_numeric_features(features, "training")

    parameters = GradientBoostingParameters()
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), list(NUMERIC_FEATURE_COLUMNS)),
            ("binary", "passthrough", list(BINARY_FEATURE_COLUMNS)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    estimator = GradientBoostingRegressor(
        loss=parameters.loss,
        learning_rate=parameters.learning_rate,
        n_estimators=parameters.n_estimators,
        subsample=parameters.subsample,
        criterion=parameters.criterion,
        min_samples_split=parameters.min_samples_split,
        min_samples_leaf=parameters.min_samples_leaf,
        max_depth=parameters.max_depth,
        max_features=parameters.max_features,
        random_state=parameters.random_state,
    )
    pipeline = Pipeline(
        steps=[("preprocessing", preprocessor), ("gradient_boosting", estimator)]
    )
    pipeline.fit(features, targets)

    transformed_names = tuple(
        str(name)
        for name in pipeline.named_steps["preprocessing"].get_feature_names_out()
    )
    if len(transformed_names) != len(set(transformed_names)):
        raise ValueError("Gradient Boosting preprocessing produced duplicate feature names")

    imputer = pipeline.named_steps["preprocessing"].named_transformers_["numeric"]
    medians = np.asarray(imputer.statistics_, dtype=float)
    if len(medians) != len(NUMERIC_FEATURE_COLUMNS) or not np.isfinite(medians).all():
        raise ValueError("Training numeric imputation statistics must be finite")

    importances = np.asarray(
        pipeline.named_steps["gradient_boosting"].feature_importances_, dtype=float
    ).reshape(-1)
    _validate_feature_importances(transformed_names, importances)
    ordered_importances = tuple(
        GradientBoostingFeatureImportance(name, float(importance))
        for name, importance in sorted(
            zip(transformed_names, importances, strict=True),
            key=lambda item: (-float(item[1]), item[0]),
        )
    )
    return QBPassingYardsGradientBoostingModel(
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


def predict_qb_passing_yards_gradient_boosting(
    model: QBPassingYardsGradientBoostingModel,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return one finite prediction per target key without using target values."""

    if not isinstance(model, QBPassingYardsGradientBoostingModel):
        raise TypeError("model must be a QBPassingYardsGradientBoostingModel")
    _validate_model(model)
    _validate_frame(frame, "prediction", require_target=False)
    predictions = np.asarray(model.pipeline.predict(_model_features(frame)), dtype=float)
    if len(predictions) != len(frame):
        raise ValueError("Gradient Boosting prediction count does not match input row count")
    if not np.isfinite(predictions).all():
        raise ValueError("Gradient Boosting predictions must be finite")

    result_columns = [*TARGET_KEY]
    if TARGET_COLUMN in frame.columns:
        result_columns.append(TARGET_COLUMN)
    result = frame.loc[:, result_columns].copy(deep=True)
    result[PREDICTION_COLUMN] = predictions
    result[COLD_START_COLUMN] = frame[HISTORICAL_AVERAGE_COLUMN].isna().to_numpy(
        dtype=bool
    )
    return result.reset_index(drop=True)


def evaluate_qb_passing_yards_gradient_boosting(
    model: QBPassingYardsGradientBoostingModel,
    validation_frame: pd.DataFrame,
) -> QBPassingYardsGradientBoostingEvaluation:
    """Score validation only through the public shared regression evaluator."""

    _validate_frame(validation_frame, "validation", require_target=True)
    canonical_validation = validation_frame.sort_values(TARGET_KEY, kind="mergesort")
    predictions = predict_qb_passing_yards_gradient_boosting(model, canonical_validation)
    cold_start_mask = predictions[COLD_START_COLUMN]
    cold_start_count = int(cold_start_mask.sum())
    non_cold_start_count = len(predictions) - cold_start_count
    return QBPassingYardsGradientBoostingEvaluation(
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


def train_and_validate_qb_passing_yards_gradient_boosting(
    dataset_split: QBPassingYardsDatasetSplit,
) -> QBPassingYardsGradientBoostingValidation:
    """Fit and compare on train/validation only; this never accesses test."""

    if not isinstance(dataset_split, QBPassingYardsDatasetSplit):
        raise TypeError("dataset_split must be a QBPassingYardsDatasetSplit")

    model = fit_qb_passing_yards_gradient_boosting(dataset_split.train)
    validation = evaluate_qb_passing_yards_gradient_boosting(
        model, dataset_split.validation
    )
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
    random_forest_model = fit_qb_passing_yards_random_forest(dataset_split.train)
    random_forest_metrics = evaluate_qb_passing_yards_random_forest(
        random_forest_model, canonical_validation
    ).overall_metrics

    return QBPassingYardsGradientBoostingValidation(
        model=model,
        validation=validation,
        historical_average_comparison=_comparison(
            "Historical average", baseline_metrics, validation.overall_metrics
        ),
        linear_regression_comparison=_comparison(
            "Linear Regression", linear_metrics, validation.overall_metrics
        ),
        random_forest_comparison=_comparison(
            "Random Forest", random_forest_metrics, validation.overall_metrics
        ),
    )


def _comparison(
    name: str, comparison: RegressionMetrics, gradient_boosting: RegressionMetrics
) -> GradientBoostingComparison:
    return GradientBoostingComparison(
        comparison_model_name=name,
        comparison_metrics=comparison,
        mae_improvement=comparison.mae - gradient_boosting.mae,
        rmse_improvement=comparison.rmse - gradient_boosting.rmse,
        r_squared_improvement=gradient_boosting.r_squared - comparison.r_squared,
        gradient_boosting_wins_on_mae=gradient_boosting.mae < comparison.mae,
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
    for feature in HISTORY_SAMPLE_SIZE_COLUMNS:
        values = pd.to_numeric(data[feature], errors="coerce")
        invalid = (
            values.isna()
            | ~np.isfinite(values)
            | values.lt(0)
            | values.mod(1).ne(0)
        )
        if invalid.any():
            raise ValueError(
                f"{name.capitalize()} frame has invalid history sample sizes in {feature}"
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
        history_feature = f"{prefix}_season_history_games"
        if not values.eq(data[history_feature].eq(0)).all():
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
        raise ValueError(
            "Gradient Boosting importance count does not match transformed feature count"
        )
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("Gradient Boosting preprocessing produced duplicate feature names")
    if not np.isfinite(importances).all() or (importances < 0).any():
        raise ValueError(
            "Gradient Boosting feature importances must be finite and nonnegative"
        )
    if not np.isclose(float(importances.sum()), 1.0, rtol=1e-9, atol=1e-12):
        raise ValueError("Gradient Boosting feature importances must sum to one")


def _validate_model(model: QBPassingYardsGradientBoostingModel) -> None:
    if model.feature_columns != tuple(FEATURE_COLUMNS):
        raise ValueError("Gradient Boosting model feature contract does not match QB dataset")
    if model.numeric_feature_columns != NUMERIC_FEATURE_COLUMNS:
        raise ValueError(
            "Gradient Boosting model numeric feature contract does not match QB dataset"
        )
    if model.binary_feature_columns != BINARY_FEATURE_COLUMNS:
        raise ValueError(
            "Gradient Boosting model binary feature contract does not match QB dataset"
        )
    if model.training_row_count < 1:
        raise ValueError("Gradient Boosting model has invalid fitted metadata")


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
