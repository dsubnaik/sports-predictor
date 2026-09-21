"""Structured, test-blind quality audit for QB passing-yards modeling data.

The audit accepts an already-built chronological split.  It validates dataset
structure and point-in-time feature integrity for all partitions, but reads
target values only from train and validation: the training fallback is fit on
train, and baseline metrics are calculated on validation.  Test targets are
intentionally never selected or summarized, keeping the final test partition
outcome-blind until final model comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

from football.training.build_qb_passing_yards_dataset import (
    FEATURE_COLUMNS,
    OUTPUT_COLUMNS,
    TARGET_COLUMN,
    TARGET_KEY,
)
from football.training.qb_passing_yards_baseline import (
    FALLBACK_USED_COLUMN,
    RegressionMetrics,
    evaluate_qb_passing_yards_predictions,
    fit_qb_passing_yards_baseline,
    predict_qb_passing_yards_baseline,
)
from football.training.split_qb_passing_yards_dataset import (
    QBPassingYardsDatasetSplit,
)


QB_SAMPLE_SIZE_COLUMNS = ["qb_season_history_games", "qb_last3_history_games"]
DEFENSE_SAMPLE_SIZE_COLUMNS = [
    "defense_season_history_games",
    "defense_last3_history_games",
]
MISSING_HISTORY_PAIRS = [
    ("qb_season_passing_yards_avg", "qb_season_history_games", "qb_missing_history"),
    ("defense_season_passing_yards_allowed_avg", "defense_season_history_games", "defense_missing_history"),
]
HISTORY_AVERAGE_COLUMNS = [
    "qb_season_passing_yards_avg",
    "qb_last3_passing_yards_avg",
    "defense_season_passing_yards_allowed_avg",
    "defense_last3_passing_yards_allowed_avg",
]


@dataclass(frozen=True)
class SampleSizeSummary:
    """Distribution and zero-history count for one history sample-size field."""

    feature_name: str
    minimum: int
    median: float
    maximum: int
    zero_history_count: int


@dataclass(frozen=True)
class PartitionSummary:
    """Outcome-blind partition structure and point-in-time feature diagnostics."""

    name: str
    row_count: int
    unique_player_count: int
    unique_game_count: int
    minimum_period: tuple[int, int]
    maximum_period: tuple[int, int]
    missing_qb_season_history_count: int
    missing_qb_season_history_rate: float
    missing_qb_last3_history_count: int
    missing_qb_last3_history_rate: float
    missing_defense_season_history_count: int
    missing_defense_season_history_rate: float
    missing_defense_last3_history_count: int
    missing_defense_last3_history_rate: float
    sample_size_summaries: tuple[SampleSizeSummary, ...]
    zero_qb_history_count: int
    zero_defense_history_count: int
    negative_qb_historical_average_count: int
    negative_defense_historical_average_count: int
    permitted_cold_start_count: int
    invalid_sample_size_count: int


@dataclass(frozen=True)
class TrainingTargetSummary:
    """Finite train-target distribution; standard deviation uses sample ddof=1."""

    row_count: int
    mean: float
    standard_deviation: float | None
    minimum: float
    percentile_25: float
    median: float
    percentile_75: float
    maximum: float


@dataclass(frozen=True)
class TrainingFeatureSummary:
    """One predictive feature's training-only completeness and numeric range."""

    feature_name: str
    feature_category: str
    dtype: str
    non_missing_count: int
    missing_count: int
    missing_rate: float
    finite_numeric_count: int | None
    minimum: float | None
    median: float | None
    maximum: float | None


@dataclass(frozen=True)
class ValidationBaselineAudit:
    """Training-only baseline fit and validation-only performance diagnostics."""

    training_fallback_value: float
    overall_metrics: RegressionMetrics
    fallback_row_count: int
    fallback_row_rate: float
    non_fallback_row_count: int
    non_fallback_row_rate: float
    fallback_metrics: RegressionMetrics | None
    non_fallback_metrics: RegressionMetrics | None


@dataclass(frozen=True)
class QBPassingYardsDatasetAudit:
    """Complete scalar audit report; intentionally retains no input dataframes."""

    partition_summaries: tuple[PartitionSummary, ...]
    training_target_summary: TrainingTargetSummary
    training_feature_summaries: tuple[TrainingFeatureSummary, ...]
    validation_baseline: ValidationBaselineAudit


def audit_qb_passing_yards_dataset(
    dataset_split: QBPassingYardsDatasetSplit,
) -> QBPassingYardsDatasetAudit:
    """Return a deterministic, test-outcome-blind audit of a completed QB split."""

    if not isinstance(dataset_split, QBPassingYardsDatasetSplit):
        raise TypeError("dataset_split must be a QBPassingYardsDatasetSplit")

    partitions = (
        ("train", dataset_split.train, True),
        ("validation", dataset_split.validation, True),
        ("test", dataset_split.test, False),
    )
    _validate_partition_schemas(partitions)
    for name, data, inspect_targets in partitions:
        _validate_partition(data, name, inspect_targets)
    _validate_cross_partition_structure(dataset_split)

    partition_summaries = tuple(
        _partition_summary(name, data) for name, data, _ in partitions
    )
    training_target_summary = _training_target_summary(dataset_split.train)
    training_feature_summaries = tuple(
        _training_feature_summary(dataset_split.train, feature)
        for feature in FEATURE_COLUMNS
    )
    validation_baseline = _validation_baseline_audit(
        dataset_split.train, dataset_split.validation
    )
    return QBPassingYardsDatasetAudit(
        partition_summaries=partition_summaries,
        training_target_summary=training_target_summary,
        training_feature_summaries=training_feature_summaries,
        validation_baseline=validation_baseline,
    )


def _validate_partition_schemas(
    partitions: tuple[tuple[str, pd.DataFrame, bool], ...],
) -> None:
    reference_columns: list[str] | None = None
    reference_dtypes: pd.Series | None = None
    for name, data, _ in partitions:
        missing = sorted(set(OUTPUT_COLUMNS).difference(data.columns))
        if missing:
            raise ValueError(f"{name} partition is missing required columns: {missing}")
        if reference_columns is None:
            reference_columns = data.columns.tolist()
            reference_dtypes = data.dtypes
        elif data.columns.tolist() != reference_columns or not data.dtypes.equals(reference_dtypes):
            raise ValueError("QB dataset split partitions have inconsistent schemas")


def _validate_partition(data: pd.DataFrame, name: str, inspect_targets: bool) -> None:
    if data.empty:
        raise ValueError(f"{name} partition must not be empty")
    if data[TARGET_KEY].isna().any().any():
        raise ValueError(f"{name} partition has missing target-key values")
    if data.duplicated(TARGET_KEY).any():
        raise ValueError(f"{name} partition has duplicate target keys")
    _validate_periods(data, name)
    _validate_feature_integrity(data, name)
    if inspect_targets:
        _finite_targets(data[TARGET_COLUMN], name)


def _validate_periods(data: pd.DataFrame, name: str) -> None:
    for column in ("season", "week"):
        for value in data[column]:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} partition has invalid {column} values")
            if not np.isfinite(value) or int(value) != value or value < 1:
                raise ValueError(f"{name} partition has invalid {column} values")


def _validate_feature_integrity(data: pd.DataFrame, name: str) -> None:
    for feature in FEATURE_COLUMNS:
        values = data[feature]
        if pd.api.types.is_bool_dtype(values):
            if values.isna().any():
                raise ValueError(f"{name} partition has invalid missing-history flags")
            continue
        numeric = pd.to_numeric(values, errors="coerce")
        invalid = ~values.isna() & (numeric.isna() | ~np.isfinite(numeric))
        if invalid.any():
            raise ValueError(f"{name} partition has non-finite predictive values")

    for feature, _, flag in MISSING_HISTORY_PAIRS:
        values = data[flag]
        if not pd.api.types.is_bool_dtype(values) or values.isna().any():
            raise ValueError(f"{name} partition has invalid missing-history flags")

    for feature in [*QB_SAMPLE_SIZE_COLUMNS, *DEFENSE_SAMPLE_SIZE_COLUMNS]:
        numeric = pd.to_numeric(data[feature], errors="coerce")
        invalid = numeric.isna() | ~np.isfinite(numeric) | (numeric < 0) | (numeric % 1 != 0)
        if invalid.any():
            raise ValueError(f"{name} partition has invalid historical sample sizes")

    for average, sample_size, flag in MISSING_HISTORY_PAIRS:
        zero_history = data[sample_size].astype(int).eq(0)
        if not data[flag].eq(zero_history).all() or not data[average].isna().eq(zero_history).all():
            raise ValueError(f"{name} partition has inconsistent missing-history flags")


def _validate_cross_partition_structure(split: QBPassingYardsDatasetSplit) -> None:
    partitions = (split.train, split.validation, split.test)
    key_sets = [set(map(tuple, data[TARGET_KEY].to_numpy())) for data in partitions]
    if key_sets[0] & key_sets[1] or key_sets[0] & key_sets[2] or key_sets[1] & key_sets[2]:
        raise ValueError("QB dataset split partitions have overlapping target keys")

    periods = [(_period_range(data)[0], _period_range(data)[1]) for data in partitions]
    if not periods[0][1] < periods[1][0]:
        raise ValueError("train partition must chronologically precede validation")
    if not periods[1][1] < periods[2][0]:
        raise ValueError("validation partition must chronologically precede test")


def _partition_summary(name: str, data: pd.DataFrame) -> PartitionSummary:
    row_count = len(data)
    summaries = tuple(_sample_size_summary(data, feature) for feature in [
        *QB_SAMPLE_SIZE_COLUMNS,
        *DEFENSE_SAMPLE_SIZE_COLUMNS,
    ])
    qb_season_missing = int(data["qb_season_passing_yards_avg"].isna().sum())
    qb_last3_missing = int(data["qb_last3_passing_yards_avg"].isna().sum())
    defense_season_missing = int(data["defense_season_passing_yards_allowed_avg"].isna().sum())
    defense_last3_missing = int(data["defense_last3_passing_yards_allowed_avg"].isna().sum())
    return PartitionSummary(
        name=name,
        row_count=row_count,
        unique_player_count=int(data["player_id"].nunique()),
        unique_game_count=int(data["game_id"].nunique()),
        minimum_period=_period_range(data)[0],
        maximum_period=_period_range(data)[1],
        missing_qb_season_history_count=qb_season_missing,
        missing_qb_season_history_rate=qb_season_missing / row_count,
        missing_qb_last3_history_count=qb_last3_missing,
        missing_qb_last3_history_rate=qb_last3_missing / row_count,
        missing_defense_season_history_count=defense_season_missing,
        missing_defense_season_history_rate=defense_season_missing / row_count,
        missing_defense_last3_history_count=defense_last3_missing,
        missing_defense_last3_history_rate=defense_last3_missing / row_count,
        sample_size_summaries=summaries,
        zero_qb_history_count=int(data["qb_season_history_games"].eq(0).sum()),
        zero_defense_history_count=int(data["defense_season_history_games"].eq(0).sum()),
        negative_qb_historical_average_count=_negative_average_count(data, HISTORY_AVERAGE_COLUMNS[:2]),
        negative_defense_historical_average_count=_negative_average_count(data, HISTORY_AVERAGE_COLUMNS[2:]),
        permitted_cold_start_count=qb_season_missing,
        invalid_sample_size_count=0,
    )


def _sample_size_summary(data: pd.DataFrame, feature: str) -> SampleSizeSummary:
    values = data[feature].astype(int)
    return SampleSizeSummary(
        feature_name=feature,
        minimum=int(values.min()),
        median=float(values.median()),
        maximum=int(values.max()),
        zero_history_count=int(values.eq(0).sum()),
    )


def _training_target_summary(data: pd.DataFrame) -> TrainingTargetSummary:
    values = _finite_targets(data[TARGET_COLUMN], "train")
    return TrainingTargetSummary(
        row_count=len(values),
        mean=float(values.mean()),
        standard_deviation=float(values.std(ddof=1)) if len(values) > 1 else None,
        minimum=float(values.min()),
        percentile_25=float(values.quantile(0.25)),
        median=float(values.median()),
        percentile_75=float(values.quantile(0.75)),
        maximum=float(values.max()),
    )


def _training_feature_summary(data: pd.DataFrame, feature: str) -> TrainingFeatureSummary:
    values = data[feature]
    missing_count = int(values.isna().sum())
    if pd.api.types.is_bool_dtype(values):
        return TrainingFeatureSummary(
            feature_name=feature,
            feature_category="boolean",
            dtype=str(values.dtype),
            non_missing_count=len(values) - missing_count,
            missing_count=missing_count,
            missing_rate=missing_count / len(values),
            finite_numeric_count=None,
            minimum=None,
            median=None,
            maximum=None,
        )
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    return TrainingFeatureSummary(
        feature_name=feature,
        feature_category="numeric",
        dtype=str(values.dtype),
        non_missing_count=len(values) - missing_count,
        missing_count=missing_count,
        missing_rate=missing_count / len(values),
        finite_numeric_count=len(numeric),
        minimum=float(numeric.min()) if not numeric.empty else None,
        median=float(numeric.median()) if not numeric.empty else None,
        maximum=float(numeric.max()) if not numeric.empty else None,
    )


def _validation_baseline_audit(
    train: pd.DataFrame, validation: pd.DataFrame
) -> ValidationBaselineAudit:
    baseline = fit_qb_passing_yards_baseline(train)
    predictions = predict_qb_passing_yards_baseline(baseline, validation)
    overall = evaluate_qb_passing_yards_predictions(validation, predictions)
    fallback_mask = predictions[FALLBACK_USED_COLUMN]
    fallback_count = int(fallback_mask.sum())
    non_fallback_count = len(predictions) - fallback_count
    return ValidationBaselineAudit(
        training_fallback_value=baseline.training_target_mean,
        overall_metrics=overall,
        fallback_row_count=fallback_count,
        fallback_row_rate=fallback_count / len(predictions),
        non_fallback_row_count=non_fallback_count,
        non_fallback_row_rate=non_fallback_count / len(predictions),
        fallback_metrics=_subgroup_metrics(validation, predictions, fallback_mask),
        non_fallback_metrics=_subgroup_metrics(validation, predictions, ~fallback_mask),
    )


def _subgroup_metrics(
    validation: pd.DataFrame, predictions: pd.DataFrame, mask: pd.Series
) -> RegressionMetrics | None:
    if not mask.any():
        return None
    keys = predictions.loc[mask, TARGET_KEY]
    validation_rows = validation.merge(keys, on=TARGET_KEY, how="inner", validate="one_to_one")
    prediction_rows = predictions.loc[mask]
    return evaluate_qb_passing_yards_predictions(validation_rows, prediction_rows)


def _finite_targets(values: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric).all():
        raise ValueError(f"{name} partition has non-finite target_passing_yards values")
    return numeric.astype(float)


def _period_range(data: pd.DataFrame) -> tuple[tuple[int, int], tuple[int, int]]:
    periods = [(int(season), int(week)) for season, week in zip(data["season"], data["week"], strict=True)]
    return min(periods), max(periods)


def _negative_average_count(data: pd.DataFrame, columns: list[str]) -> int:
    return int(sum(data[column].lt(0).sum() for column in columns))
