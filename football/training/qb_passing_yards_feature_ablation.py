"""Leakage-safe QB-history versus defense-feature validation ablation.

This module changes only the fixed, public input feature group.  Each completed
estimator keeps its established parameters, training-only median imputation,
and deterministic seed.  It fits on ``split.train`` and predicts only
``split.validation``; the public orchestration intentionally never reads
``split.test``.

Positive MAE/RMSE improvements always mean that the second group in a
comparison is better: ``first - second``.  R-squared improvement is
``second - first``.  The paired bootstrap resamples entire ``game_id`` clusters
and estimates QB-history-only MAE minus combined-feature MAE, so positive
values favor adding defense.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from football.training.build_qb_passing_yards_dataset import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TARGET_KEY,
)
from football.training.qb_passing_yards_baseline import (
    HISTORICAL_AVERAGE_COLUMN,
    PREDICTION_COLUMN,
    RegressionMetrics,
    evaluate_qb_passing_yards_predictions,
)
from football.training.qb_passing_yards_gradient_boosting import (
    GradientBoostingParameters,
)
from football.training.qb_passing_yards_model_comparison import PERCENTILE_METHOD
from football.training.qb_passing_yards_random_forest import RandomForestParameters
from football.training.qb_passing_yards_xgboost import XGBoostParameters
from football.training.split_qb_passing_yards_dataset import QBPassingYardsDatasetSplit


QB_HISTORY_ONLY = "QB_HISTORY_ONLY"
DEFENSE_ONLY = "DEFENSE_ONLY"
QB_AND_DEFENSE = "QB_AND_DEFENSE"
FEATURE_GROUP_NAMES = (QB_HISTORY_ONLY, DEFENSE_ONLY, QB_AND_DEFENSE)

QB_HISTORY_FEATURE_COLUMNS = (
    "qb_season_passing_yards_avg",
    "qb_last3_passing_yards_avg",
    "qb_season_passing_attempts_avg",
    "qb_last3_passing_attempts_avg",
    "qb_season_history_games",
    "qb_last3_history_games",
    "qb_missing_history",
)
DEFENSE_FEATURE_COLUMNS = (
    "defense_season_passing_yards_allowed_avg",
    "defense_last3_passing_yards_allowed_avg",
    "defense_season_passing_attempts_allowed_avg",
    "defense_season_history_games",
    "defense_last3_history_games",
    "defense_missing_history",
    "defense_matchup_rank",
)
FEATURE_GROUP_COLUMNS = {
    QB_HISTORY_ONLY: QB_HISTORY_FEATURE_COLUMNS,
    DEFENSE_ONLY: DEFENSE_FEATURE_COLUMNS,
    QB_AND_DEFENSE: tuple(FEATURE_COLUMNS),
}
ESTIMATOR_NAMES = (
    "Linear Regression",
    "Random Forest",
    "sklearn Gradient Boosting",
    "XGBoost",
)
DEFAULT_BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 42


@dataclass(frozen=True)
class FeatureAblationPopulationMetrics:
    """One population's metrics; signed error is prediction minus actual."""

    row_count: int
    metrics: RegressionMetrics | None
    mean_signed_error: float | None
    median_absolute_error: float | None
    percentile_75_absolute_error: float | None
    percentile_90_absolute_error: float | None


@dataclass(frozen=True)
class FeatureAblationVariant:
    """One fixed estimator evaluated with exactly one public feature group."""

    estimator_name: str
    feature_group_name: str
    feature_columns: tuple[str, ...]
    feature_count: int
    validation: FeatureAblationPopulationMetrics
    cold_start: FeatureAblationPopulationMetrics
    non_cold_start: FeatureAblationPopulationMetrics


@dataclass(frozen=True)
class FeatureGroupComparison:
    """Comparison where positive improvements favor ``second_feature_group``."""

    estimator_name: str
    first_feature_group: str
    second_feature_group: str
    mae_improvement_for_second_group: float
    rmse_improvement_for_second_group: float
    r_squared_improvement_for_second_group: float
    second_group_wins_on_mae: bool


@dataclass(frozen=True)
class PairedDefenseMAEBootstrap:
    """Game-cluster bootstrap for QB-history-only MAE minus combined MAE."""

    estimator_name: str
    point_estimate: float
    lower_bound: float
    upper_bound: float
    replicate_count: int
    seed: int
    cluster_count: int
    percentile_method: str


@dataclass(frozen=True)
class EstimatorFeatureAblation:
    """All three feature variants and fixed comparisons for one estimator."""

    estimator_name: str
    variants: tuple[FeatureAblationVariant, ...]
    qb_history_to_combined: FeatureGroupComparison
    defense_to_combined: FeatureGroupComparison
    qb_history_to_defense: FeatureGroupComparison
    adding_defense_bootstrap: PairedDefenseMAEBootstrap


@dataclass(frozen=True)
class QBPassingYardsFeatureAblation:
    """Immutable train/validation-only feature-ablation result with no frames."""

    validation_row_count: int
    bootstrap_replicates: int
    bootstrap_seed: int
    estimator_results: tuple[EstimatorFeatureAblation, ...]


def analyze_qb_passing_yards_feature_ablation(
    dataset_split: QBPassingYardsDatasetSplit,
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
) -> QBPassingYardsFeatureAblation:
    """Run fixed feature groups on train/validation only.

    All median imputers and estimators fit exclusively on the canonicalized
    training partition.  The test property is deliberately never accessed.
    ``bootstrap_replicates`` is exposed for smaller synthetic tests; the
    production default is 2,000 game-cluster replicates using seed 42 and
    NumPy's deterministic ``linear`` percentile method.
    """

    if not isinstance(dataset_split, QBPassingYardsDatasetSplit):
        raise TypeError("dataset_split must be a QBPassingYardsDatasetSplit")
    if isinstance(bootstrap_replicates, bool) or not isinstance(bootstrap_replicates, int) or bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be a positive integer")
    _validate_feature_contract()
    train = _canonical_frame(dataset_split.train, "training", require_target=True)
    validation = _canonical_frame(dataset_split.validation, "validation", require_target=True)
    actual = _finite(validation[TARGET_COLUMN], "validation target_passing_yards").to_numpy(dtype=float)
    cold_start_mask = validation[HISTORICAL_AVERAGE_COLUMN].isna().to_numpy(dtype=bool)

    results = []
    for estimator_name in ESTIMATOR_NAMES:
        predictions = {
            group: _fit_predict(estimator_name, columns, train, validation)
            for group, columns in FEATURE_GROUP_COLUMNS.items()
        }
        variants = tuple(
            _variant(estimator_name, group, columns, validation, actual, predictions[group], cold_start_mask)
            for group, columns in FEATURE_GROUP_COLUMNS.items()
        )
        by_group = {variant.feature_group_name: variant for variant in variants}
        results.append(
            EstimatorFeatureAblation(
                estimator_name=estimator_name,
                variants=variants,
                qb_history_to_combined=_comparison(
                    estimator_name, by_group[QB_HISTORY_ONLY], by_group[QB_AND_DEFENSE]
                ),
                defense_to_combined=_comparison(
                    estimator_name, by_group[DEFENSE_ONLY], by_group[QB_AND_DEFENSE]
                ),
                qb_history_to_defense=_comparison(
                    estimator_name, by_group[QB_HISTORY_ONLY], by_group[DEFENSE_ONLY]
                ),
                adding_defense_bootstrap=_paired_bootstrap(
                    estimator_name,
                    validation["game_id"].to_numpy(),
                    actual,
                    predictions[QB_HISTORY_ONLY],
                    predictions[QB_AND_DEFENSE],
                    bootstrap_replicates,
                ),
            )
        )
    return QBPassingYardsFeatureAblation(
        validation_row_count=len(validation),
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=BOOTSTRAP_SEED,
        estimator_results=tuple(results),
    )


def _validate_feature_contract() -> None:
    expected = (*QB_HISTORY_FEATURE_COLUMNS, *DEFENSE_FEATURE_COLUMNS)
    if tuple(FEATURE_COLUMNS) != expected:
        raise ValueError(
            "QB dataset FEATURE_COLUMNS drifted from the fixed feature-ablation contract"
        )
    if len(set(expected)) != len(expected) or len(expected) != 14:
        raise ValueError("Feature-ablation groups must contain 14 unique public features")


def _canonical_frame(frame: pd.DataFrame, name: str, *, require_target: bool) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name.capitalize()} frame must be a pandas DataFrame")
    required = [*TARGET_KEY, *FEATURE_COLUMNS]
    if require_target:
        required.append(TARGET_COLUMN)
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{name.capitalize()} frame is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"{name.capitalize()} frame must not be empty")
    if frame[TARGET_KEY].isna().any().any():
        raise ValueError(f"{name.capitalize()} frame has missing target-key values")
    if frame.duplicated(TARGET_KEY).any():
        raise ValueError(f"{name.capitalize()} frame has duplicate target keys")
    _finite(frame[TARGET_COLUMN], f"{name} target_passing_yards")
    _validate_feature_values(frame, name)
    return frame.sort_values(TARGET_KEY, kind="mergesort").copy(deep=True).reset_index(drop=True)


def _validate_feature_values(frame: pd.DataFrame, name: str) -> None:
    for column in FEATURE_COLUMNS:
        values = frame[column]
        if column.endswith("_missing_history"):
            if not pd.api.types.is_bool_dtype(values) or values.isna().any():
                raise ValueError(f"{name.capitalize()} frame has invalid missing-history flags in {column}")
            season_average = {
                "qb_missing_history": "qb_season_passing_yards_avg",
                "defense_missing_history": "defense_season_passing_yards_allowed_avg",
            }[column]
            season_history = column.removesuffix("_missing_history") + "_season_history_games"
            if (
                not frame[column].eq(frame[season_average].isna()).all()
                or not frame[column].eq(frame[season_history].eq(0)).all()
            ):
                raise ValueError(f"{name.capitalize()} frame has inconsistent missing-history flags")
        else:
            numeric = pd.to_numeric(values, errors="coerce")
            invalid = ~values.isna() & (numeric.isna() | ~np.isfinite(numeric))
            if invalid.any():
                raise ValueError(f"{name.capitalize()} frame has non-finite predictive values in {column}")
            if column.endswith("_history_games"):
                invalid_history = numeric.isna() | ~np.isfinite(numeric) | numeric.lt(0) | numeric.mod(1).ne(0)
                if invalid_history.any():
                    raise ValueError(f"{name.capitalize()} frame has invalid history sample sizes in {column}")


def _fit_predict(
    estimator_name: str,
    columns: tuple[str, ...],
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> np.ndarray:
    numeric_columns = tuple(column for column in columns if not column.endswith("_missing_history"))
    binary_columns = tuple(column for column in columns if column.endswith("_missing_history"))
    train_features = _features(train, columns, numeric_columns)
    validation_features = _features(validation, columns, numeric_columns)
    if any(train_features[column].isna().all() for column in numeric_columns):
        raise ValueError("Training frame has selected numeric features with no usable values")
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), list(numeric_columns)),
            ("binary", "passthrough", list(binary_columns)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    pipeline = Pipeline(
        steps=[("preprocessing", preprocessor), ("estimator", _estimator(estimator_name))]
    )
    pipeline.fit(train_features, _finite(train[TARGET_COLUMN], "training target_passing_yards"))
    predictions = np.asarray(pipeline.predict(validation_features), dtype=float).reshape(-1)
    if len(predictions) != len(validation) or not np.isfinite(predictions).all():
        raise ValueError(f"{estimator_name} predictions must be finite and aligned to validation rows")
    return predictions


def _features(frame: pd.DataFrame, columns: tuple[str, ...], numeric_columns: tuple[str, ...]) -> pd.DataFrame:
    result = frame.loc[:, columns].copy(deep=True)
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
    return result


def _estimator(name: str):
    if name == "Linear Regression":
        return LinearRegression()
    if name == "Random Forest":
        parameters = RandomForestParameters()
        return RandomForestRegressor(
            n_estimators=parameters.n_estimators, random_state=parameters.random_state,
            max_depth=parameters.max_depth, min_samples_split=parameters.min_samples_split,
            min_samples_leaf=parameters.min_samples_leaf, max_features=parameters.max_features,
            bootstrap=parameters.bootstrap, n_jobs=parameters.n_jobs,
        )
    if name == "sklearn Gradient Boosting":
        parameters = GradientBoostingParameters()
        return GradientBoostingRegressor(
            loss=parameters.loss, learning_rate=parameters.learning_rate,
            n_estimators=parameters.n_estimators, subsample=parameters.subsample,
            criterion=parameters.criterion, min_samples_split=parameters.min_samples_split,
            min_samples_leaf=parameters.min_samples_leaf, max_depth=parameters.max_depth,
            max_features=parameters.max_features, random_state=parameters.random_state,
        )
    if name == "XGBoost":
        parameters = XGBoostParameters()
        return XGBRegressor(
            objective=parameters.objective, n_estimators=parameters.n_estimators,
            learning_rate=parameters.learning_rate, max_depth=parameters.max_depth,
            min_child_weight=parameters.min_child_weight, gamma=parameters.gamma,
            subsample=parameters.subsample, colsample_bytree=parameters.colsample_bytree,
            reg_alpha=parameters.reg_alpha, reg_lambda=parameters.reg_lambda,
            tree_method=parameters.tree_method, random_state=parameters.random_state,
            n_jobs=parameters.n_jobs, verbosity=parameters.verbosity,
            importance_type=parameters.importance_type,
        )
    raise ValueError(f"Unknown ablation estimator: {name}")


def _variant(
    estimator_name: str, group: str, columns: tuple[str, ...], validation: pd.DataFrame,
    actual: np.ndarray, predictions: np.ndarray, cold_start_mask: np.ndarray,
) -> FeatureAblationVariant:
    return FeatureAblationVariant(
        estimator_name=estimator_name,
        feature_group_name=group,
        feature_columns=columns,
        feature_count=len(columns),
        validation=_population_metrics(validation, actual, predictions, np.ones(len(validation), dtype=bool)),
        cold_start=_population_metrics(validation, actual, predictions, cold_start_mask),
        non_cold_start=_population_metrics(validation, actual, predictions, ~cold_start_mask),
    )


def _population_metrics(
    validation: pd.DataFrame, actual: np.ndarray, predictions: np.ndarray, mask: np.ndarray
) -> FeatureAblationPopulationMetrics:
    count = int(mask.sum())
    if count == 0:
        return FeatureAblationPopulationMetrics(0, None, None, None, None, None)
    subset = validation.loc[mask, :]
    values = predictions[mask]
    metrics = _metrics(subset, values)
    signed = values - actual[mask]
    absolute = np.abs(signed)
    return FeatureAblationPopulationMetrics(
        count, metrics, float(np.mean(signed)), float(np.percentile(absolute, 50, method=PERCENTILE_METHOD)),
        float(np.percentile(absolute, 75, method=PERCENTILE_METHOD)),
        float(np.percentile(absolute, 90, method=PERCENTILE_METHOD)),
    )


def _metrics(actual_frame: pd.DataFrame, predictions: np.ndarray) -> RegressionMetrics:
    output = actual_frame.loc[:, TARGET_KEY].copy(deep=True)
    output[PREDICTION_COLUMN] = predictions
    return evaluate_qb_passing_yards_predictions(actual_frame, output)


def _comparison(
    estimator_name: str, first: FeatureAblationVariant, second: FeatureAblationVariant
) -> FeatureGroupComparison:
    first_metrics = first.validation.metrics
    second_metrics = second.validation.metrics
    if first_metrics is None or second_metrics is None:
        raise ValueError("Nonempty validation variants must have metrics")
    return FeatureGroupComparison(
        estimator_name, first.feature_group_name, second.feature_group_name,
        first_metrics.mae - second_metrics.mae,
        first_metrics.rmse - second_metrics.rmse,
        second_metrics.r_squared - first_metrics.r_squared,
        second_metrics.mae < first_metrics.mae,
    )


def _paired_bootstrap(
    estimator_name: str, game_ids: np.ndarray, actual: np.ndarray,
    qb_history_predictions: np.ndarray, combined_predictions: np.ndarray,
    replicate_count: int,
) -> PairedDefenseMAEBootstrap:
    clusters = tuple(sorted(pd.unique(game_ids).tolist(), key=str))
    if not clusters:
        raise ValueError("Validation bootstrap requires at least one game_id cluster")
    positions = tuple(np.flatnonzero(game_ids == game_id) for game_id in clusters)
    history_errors = np.abs(qb_history_predictions - actual)
    combined_errors = np.abs(combined_predictions - actual)
    point = float(np.mean(history_errors) - np.mean(combined_errors))
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.empty(replicate_count, dtype=float)
    for index in range(replicate_count):
        chosen = generator.integers(0, len(positions), size=len(positions))
        row_indices = np.concatenate(tuple(positions[item] for item in chosen))
        values[index] = float(np.mean(history_errors[row_indices]) - np.mean(combined_errors[row_indices]))
    bounds = np.percentile(values, [2.5, 97.5], method=PERCENTILE_METHOD)
    return PairedDefenseMAEBootstrap(
        estimator_name, point, float(bounds[0]), float(bounds[1]), replicate_count,
        BOOTSTRAP_SEED, len(clusters), PERCENTILE_METHOD,
    )


def _finite(values: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric).all():
        raise ValueError(f"{name} values must be finite")
    return numeric.astype(float)
