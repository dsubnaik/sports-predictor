"""Leakage-safe defensive-strength representation experiment for QB passing yards.

Rank 1 is the highest point-in-time mean passing yards allowed, because the
builder ranks historical defensive passing yards allowed in descending order.
This module uses numeric rank labels only, never qualitative strength labels.
It fits only ``split.train`` and analyzes only ``split.validation``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from football.training.build_qb_passing_yards_dataset import FEATURE_COLUMNS, TARGET_COLUMN, TARGET_KEY
from football.training.qb_passing_yards_baseline import HISTORICAL_AVERAGE_COLUMN, PREDICTION_COLUMN, RegressionMetrics, evaluate_qb_passing_yards_predictions
from football.training.qb_passing_yards_feature_ablation import (
    BOOTSTRAP_SEED,
    DEFAULT_BOOTSTRAP_REPLICATES,
    ESTIMATOR_NAMES,
    FeatureAblationPopulationMetrics,
    QB_HISTORY_FEATURE_COLUMNS,
    _estimator,
    _finite,
)
from football.training.qb_passing_yards_model_comparison import PERCENTILE_METHOD
from football.training.split_qb_passing_yards_dataset import QBPassingYardsDatasetSplit


QB_ONLY_REFERENCE = "QB_ONLY_REFERENCE"
CONTINUOUS_DEFENSE = "CONTINUOUS_DEFENSE"
RANK_ONLY = "RANK_ONLY"
TIER_ONLY = "TIER_ONLY"
CONTINUOUS_PLUS_RANK = "CONTINUOUS_PLUS_RANK"
CONTINUOUS_PLUS_TIER = "CONTINUOUS_PLUS_TIER"
REPRESENTATION_NAMES = (
    QB_ONLY_REFERENCE, CONTINUOUS_DEFENSE, RANK_ONLY, TIER_ONLY,
    CONTINUOUS_PLUS_RANK, CONTINUOUS_PLUS_TIER,
)
DEFENSE_AVAILABILITY_COLUMNS = (
    "defense_season_history_games", "defense_last3_history_games", "defense_missing_history",
)
CONTINUOUS_DEFENSE_COLUMNS = (
    "defense_season_passing_yards_allowed_avg",
    "defense_last3_passing_yards_allowed_avg",
    "defense_season_passing_attempts_allowed_avg",
)
RANK_COLUMN = "defense_matchup_rank"
TIER_COLUMNS = (
    "defense_matchup_rank_tier_1_10",
    "defense_matchup_rank_tier_11_22",
    "defense_matchup_rank_tier_23_32",
    "defense_matchup_rank_tier_missing",
)
LINEAR_REFERENCE_TIER = "defense_matchup_rank_tier_missing"


@dataclass(frozen=True)
class DefensiveRepresentationVariant:
    """One estimator/representation result with actual transformed inputs."""

    estimator_name: str
    representation_name: str
    raw_input_feature_names: tuple[str, ...]
    transformed_feature_names: tuple[str, ...]
    raw_feature_count: int
    transformed_feature_count: int
    validation: FeatureAblationPopulationMetrics
    cold_start: FeatureAblationPopulationMetrics
    non_cold_start: FeatureAblationPopulationMetrics


@dataclass(frozen=True)
class DefensiveRepresentationComparison:
    """Explicit comparison whose documented sign favors ``favored_representation``."""

    estimator_name: str
    comparison_name: str
    favored_representation: str
    mae_improvement: float
    rmse_improvement: float
    r_squared_improvement: float


@dataclass(frozen=True)
class DefensiveRepresentationBootstrap:
    """Game-cluster paired MAE diagnostic with a comparison-specific sign."""

    estimator_name: str
    comparison_name: str
    favored_representation: str
    point_estimate: float
    lower_bound: float
    upper_bound: float
    replicate_count: int
    seed: int
    cluster_count: int
    percentile_method: str


@dataclass(frozen=True)
class EstimatorDefensiveRepresentation:
    """Six fixed representations and their comparisons for one estimator."""

    estimator_name: str
    variants: tuple[DefensiveRepresentationVariant, ...]
    continuous_defense_value: DefensiveRepresentationComparison
    rank_beyond_continuous: DefensiveRepresentationComparison
    tiers_beyond_continuous: DefensiveRepresentationComparison
    rank_versus_tiers: DefensiveRepresentationComparison
    rank_beyond_continuous_bootstrap: DefensiveRepresentationBootstrap
    tiers_beyond_continuous_bootstrap: DefensiveRepresentationBootstrap
    rank_versus_tiers_bootstrap: DefensiveRepresentationBootstrap


@dataclass(frozen=True)
class QBPassingYardsDefensiveRepresentationAnalysis:
    """Immutable train/validation-only defensive-representation report."""

    validation_row_count: int
    bootstrap_replicates: int
    bootstrap_seed: int
    rank_direction: str
    linear_regression_reference_tier: str
    estimator_results: tuple[EstimatorDefensiveRepresentation, ...]


def analyze_qb_passing_yards_defensive_representation(
    dataset_split: QBPassingYardsDatasetSplit, *, bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
) -> QBPassingYardsDefensiveRepresentationAnalysis:
    """Fit fixed variants on train and evaluate validation only, never test.

    Tier indicators derive solely from the already point-in-time rank in each
    row. ``tier_missing`` is omitted for Linear Regression as its intercept's
    reference category; trees receive all four boolean tier indicators.
    """
    if not isinstance(dataset_split, QBPassingYardsDatasetSplit):
        raise TypeError("dataset_split must be a QBPassingYardsDatasetSplit")
    if isinstance(bootstrap_replicates, bool) or not isinstance(bootstrap_replicates, int) or bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be a positive integer")
    _validate_contract()
    train = _canonical(dataset_split.train, "training")
    validation = _canonical(dataset_split.validation, "validation")
    train = _with_tiers(train)
    validation = _with_tiers(validation)
    actual = _finite(validation[TARGET_COLUMN], "validation target_passing_yards").to_numpy(dtype=float)
    cold = validation[HISTORICAL_AVERAGE_COLUMN].isna().to_numpy(dtype=bool)
    results = []
    for estimator_name in ESTIMATOR_NAMES:
        columns = {name: _columns(name, estimator_name) for name in REPRESENTATION_NAMES}
        predicted = {name: _fit_predict(estimator_name, selected, train, validation) for name, selected in columns.items()}
        variants = tuple(_variant(estimator_name, name, columns[name], validation, actual, predicted[name], cold) for name in REPRESENTATION_NAMES)
        by_name = {item.representation_name: item for item in variants}
        results.append(EstimatorDefensiveRepresentation(
            estimator_name, variants,
            _comparison(estimator_name, "continuous_defense_value", by_name[QB_ONLY_REFERENCE], by_name[CONTINUOUS_DEFENSE], CONTINUOUS_DEFENSE),
            _comparison(estimator_name, "rank_beyond_continuous", by_name[CONTINUOUS_DEFENSE], by_name[CONTINUOUS_PLUS_RANK], CONTINUOUS_PLUS_RANK),
            _comparison(estimator_name, "tiers_beyond_continuous", by_name[CONTINUOUS_DEFENSE], by_name[CONTINUOUS_PLUS_TIER], CONTINUOUS_PLUS_TIER),
            _comparison(estimator_name, "rank_versus_tiers", by_name[CONTINUOUS_PLUS_RANK], by_name[CONTINUOUS_PLUS_TIER], CONTINUOUS_PLUS_RANK, tier_vs_rank=True),
            _bootstrap(estimator_name, "rank_beyond_continuous", CONTINUOUS_PLUS_RANK, validation["game_id"].to_numpy(), actual, predicted[CONTINUOUS_DEFENSE], predicted[CONTINUOUS_PLUS_RANK], bootstrap_replicates),
            _bootstrap(estimator_name, "tiers_beyond_continuous", CONTINUOUS_PLUS_TIER, validation["game_id"].to_numpy(), actual, predicted[CONTINUOUS_DEFENSE], predicted[CONTINUOUS_PLUS_TIER], bootstrap_replicates),
            _bootstrap(estimator_name, "rank_versus_tiers", CONTINUOUS_PLUS_RANK, validation["game_id"].to_numpy(), actual, predicted[CONTINUOUS_PLUS_RANK], predicted[CONTINUOUS_PLUS_TIER], bootstrap_replicates, tier_vs_rank=True),
        ))
    return QBPassingYardsDefensiveRepresentationAnalysis(
        len(validation), bootstrap_replicates, BOOTSTRAP_SEED,
        "rank 1 is the highest point-in-time mean passing yards allowed", LINEAR_REFERENCE_TIER, tuple(results),
    )


def _validate_contract() -> None:
    expected = (*QB_HISTORY_FEATURE_COLUMNS, *CONTINUOUS_DEFENSE_COLUMNS, *DEFENSE_AVAILABILITY_COLUMNS, RANK_COLUMN)
    if tuple(FEATURE_COLUMNS) != expected:
        raise ValueError("QB dataset FEATURE_COLUMNS drifted from the defensive-representation contract")


def _canonical(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    required = [*TARGET_KEY, TARGET_COLUMN, *FEATURE_COLUMNS]
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name.capitalize()} frame must be a pandas DataFrame")
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{name.capitalize()} frame is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"{name.capitalize()} frame must not be empty")
    if frame[TARGET_KEY].isna().any().any() or frame.duplicated(TARGET_KEY).any():
        raise ValueError(f"{name.capitalize()} frame must have unique, nonmissing target keys")
    _finite(frame[TARGET_COLUMN], f"{name} target_passing_yards")
    _validate_features(frame, name)
    return frame.sort_values(TARGET_KEY, kind="mergesort").copy(deep=True).reset_index(drop=True)


def _validate_features(frame: pd.DataFrame, name: str) -> None:
    for column in FEATURE_COLUMNS:
        values = frame[column]
        if column == RANK_COLUMN:
            continue
        if column.endswith("_missing_history"):
            if not pd.api.types.is_bool_dtype(values) or values.isna().any():
                raise ValueError(f"{name.capitalize()} frame has invalid missing-history flags in {column}")
            season_average = {
                "qb_missing_history": "qb_season_passing_yards_avg",
                "defense_missing_history": "defense_season_passing_yards_allowed_avg",
            }[column]
            season_history = column.removesuffix("_missing_history") + "_season_history_games"
            if not values.eq(frame[season_average].isna()).all() or not values.eq(frame[season_history].eq(0)).all():
                raise ValueError(f"{name.capitalize()} frame has inconsistent missing-history flags")
        else:
            numeric = pd.to_numeric(values, errors="coerce")
            invalid = ~values.isna() & (numeric.isna() | ~np.isfinite(numeric))
            if invalid.any():
                raise ValueError(f"{name.capitalize()} frame has non-finite predictive values in {column}")
            if column.endswith("_history_games") and (numeric.isna() | numeric.lt(0) | numeric.mod(1).ne(0)).any():
                raise ValueError(f"{name.capitalize()} frame has invalid history sample sizes in {column}")
    rank = pd.to_numeric(frame[RANK_COLUMN], errors="coerce")
    invalid_rank = ~frame[RANK_COLUMN].isna() & (rank.isna() | ~np.isfinite(rank) | rank.lt(1) | rank.gt(32) | rank.mod(1).ne(0))
    if invalid_rank.any():
        raise ValueError("Defense matchup rank must be missing or a finite integer from 1 through 32")


def _with_tiers(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy(deep=True)
    rank = pd.to_numeric(result[RANK_COLUMN], errors="coerce")
    result[TIER_COLUMNS[0]] = rank.between(1, 10, inclusive="both").astype(bool)
    result[TIER_COLUMNS[1]] = rank.between(11, 22, inclusive="both").astype(bool)
    result[TIER_COLUMNS[2]] = rank.between(23, 32, inclusive="both").astype(bool)
    result[TIER_COLUMNS[3]] = rank.isna().astype(bool)
    active = result.loc[:, TIER_COLUMNS].sum(axis=1)
    if not active.eq(1).all():
        raise ValueError("Defense matchup rank tiers must have exactly one active indicator")
    return result


def _columns(representation: str, estimator_name: str) -> tuple[str, ...]:
    base = (*QB_HISTORY_FEATURE_COLUMNS, *DEFENSE_AVAILABILITY_COLUMNS)
    mapping = {
        QB_ONLY_REFERENCE: QB_HISTORY_FEATURE_COLUMNS,
        CONTINUOUS_DEFENSE: (*base, *CONTINUOUS_DEFENSE_COLUMNS),
        RANK_ONLY: (*base, RANK_COLUMN),
        TIER_ONLY: (*base, *TIER_COLUMNS),
        CONTINUOUS_PLUS_RANK: tuple(FEATURE_COLUMNS),
        CONTINUOUS_PLUS_TIER: (*base, *CONTINUOUS_DEFENSE_COLUMNS, *TIER_COLUMNS),
    }
    columns = tuple(mapping[representation])
    if estimator_name == "Linear Regression" and representation in (TIER_ONLY, CONTINUOUS_PLUS_TIER):
        return tuple(column for column in columns if column != LINEAR_REFERENCE_TIER)
    return columns


def _fit_predict(estimator_name: str, columns: tuple[str, ...], train: pd.DataFrame, validation: pd.DataFrame) -> np.ndarray:
    numeric = tuple(column for column in columns if not column.endswith("_missing_history") and column not in TIER_COLUMNS)
    binary = tuple(column for column in columns if column.endswith("_missing_history") or column in TIER_COLUMNS)
    train_features = _features(train, columns, numeric)
    validation_features = _features(validation, columns, numeric)
    if any(train_features[column].isna().all() for column in numeric):
        raise ValueError("Training frame has selected numeric features with no usable values")
    pipeline = Pipeline([("preprocessing", ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), list(numeric)), ("binary", "passthrough", list(binary)),
    ], remainder="drop", verbose_feature_names_out=False)), ("estimator", _estimator(estimator_name))])
    pipeline.fit(train_features, _finite(train[TARGET_COLUMN], "training target_passing_yards"))
    values = np.asarray(pipeline.predict(validation_features), dtype=float).reshape(-1)
    if len(values) != len(validation) or not np.isfinite(values).all():
        raise ValueError(f"{estimator_name} predictions must be finite and aligned to validation rows")
    return values


def _features(frame: pd.DataFrame, columns: tuple[str, ...], numeric: tuple[str, ...]) -> pd.DataFrame:
    result = frame.loc[:, columns].copy(deep=True)
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
    return result


def _variant(estimator: str, name: str, columns: tuple[str, ...], validation: pd.DataFrame, actual: np.ndarray, predicted: np.ndarray, cold: np.ndarray) -> DefensiveRepresentationVariant:
    numeric = tuple(column for column in columns if not column.endswith("_missing_history") and column not in TIER_COLUMNS)
    binary = tuple(column for column in columns if column.endswith("_missing_history") or column in TIER_COLUMNS)
    transformed = (*numeric, *binary)
    return DefensiveRepresentationVariant(estimator, name, columns, transformed, len(columns), len(transformed), _population(validation, actual, predicted, np.ones(len(validation), dtype=bool)), _population(validation, actual, predicted, cold), _population(validation, actual, predicted, ~cold))


def _population(validation: pd.DataFrame, actual: np.ndarray, predicted: np.ndarray, mask: np.ndarray) -> FeatureAblationPopulationMetrics:
    count = int(mask.sum())
    if count == 0:
        return FeatureAblationPopulationMetrics(0, None, None, None, None, None)
    subset, values = validation.loc[mask, :], predicted[mask]
    output = subset.loc[:, TARGET_KEY].copy(deep=True)
    output[PREDICTION_COLUMN] = values
    metrics = evaluate_qb_passing_yards_predictions(subset, output)
    signed = values - actual[mask]
    absolute = np.abs(signed)
    return FeatureAblationPopulationMetrics(count, metrics, float(np.mean(signed)), float(np.percentile(absolute, 50, method=PERCENTILE_METHOD)), float(np.percentile(absolute, 75, method=PERCENTILE_METHOD)), float(np.percentile(absolute, 90, method=PERCENTILE_METHOD)))


def _comparison(estimator: str, name: str, first: DefensiveRepresentationVariant, second: DefensiveRepresentationVariant, favored: str, *, tier_vs_rank: bool = False) -> DefensiveRepresentationComparison:
    a, b = first.validation.metrics, second.validation.metrics
    if a is None or b is None:
        raise ValueError("Nonempty validation variants must have metrics")
    if tier_vs_rank:
        return DefensiveRepresentationComparison(estimator, name, favored, b.mae - a.mae, b.rmse - a.rmse, a.r_squared - b.r_squared)
    return DefensiveRepresentationComparison(estimator, name, favored, a.mae - b.mae, a.rmse - b.rmse, b.r_squared - a.r_squared)


def _bootstrap(estimator: str, name: str, favored: str, games: np.ndarray, actual: np.ndarray, first: np.ndarray, second: np.ndarray, repeats: int, *, tier_vs_rank: bool = False) -> DefensiveRepresentationBootstrap:
    clusters = tuple(sorted(pd.unique(games).tolist(), key=str))
    if not clusters:
        raise ValueError("Validation bootstrap requires at least one game_id cluster")
    indices = tuple(np.flatnonzero(games == game) for game in clusters)
    first_error, second_error = np.abs(first - actual), np.abs(second - actual)
    def difference(rows):
        return float(np.mean(second_error[rows]) - np.mean(first_error[rows])) if tier_vs_rank else float(np.mean(first_error[rows]) - np.mean(second_error[rows]))
    generator, values = np.random.default_rng(BOOTSTRAP_SEED), np.empty(repeats, dtype=float)
    all_rows = np.arange(len(actual))
    for index in range(repeats):
        chosen = generator.integers(0, len(indices), size=len(indices))
        values[index] = difference(np.concatenate(tuple(indices[item] for item in chosen)))
    bounds = np.percentile(values, [2.5, 97.5], method=PERCENTILE_METHOD)
    return DefensiveRepresentationBootstrap(estimator, name, favored, difference(all_rows), float(bounds[0]), float(bounds[1]), repeats, BOOTSTRAP_SEED, len(clusters), PERCENTILE_METHOD)
