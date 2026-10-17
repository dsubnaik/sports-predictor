"""Protected-holdout-safe structural audit for RB rushing-yards datasets.

Validation and test diagnostics select only identifiers, diagnostics, and
pregame features. Their target column is intentionally neither required nor
read. Only the training target is summarized.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

from football.training.build_rb_rushing_yards_dataset import (
    FEATURE_COLUMNS,
    METADATA_COLUMNS,
    TARGET_COLUMN,
    TARGET_KEY,
)
from football.training.split_rb_rushing_yards_dataset import RBRushingYardsDatasetSplit


ALLOWED_PARTITION_COLUMNS = [*METADATA_COLUMNS, *FEATURE_COLUMNS]
HISTORY_COUNT_COLUMNS = [
    "rb_season_history_games", "rb_last3_history_games",
    "defense_season_history_games", "defense_last3_history_games",
]
SHARE_COLUMNS = [
    "rb_season_carry_share_avg", "rb_last3_carry_share_avg",
    "rb_season_opportunity_share_avg", "rb_last3_opportunity_share_avg",
]
WORKLOAD_AND_SHARE_COLUMNS = [
    "rb_season_opportunities_avg", "rb_last3_opportunities_avg", *SHARE_COLUMNS,
]


@dataclass(frozen=True)
class FeatureDiagnostic:
    feature_name: str
    missing_count: int
    missing_rate: float
    nonfinite_count: int


@dataclass(frozen=True)
class NumericDistribution:
    feature_name: str
    finite_count: int
    minimum: float | None
    percentile_25: float | None
    median: float | None
    percentile_75: float | None
    maximum: float | None


@dataclass(frozen=True)
class TeamChangeCounts:
    changed_count: int
    unchanged_count: int
    unknown_count: int
    invalid_count: int


@dataclass(frozen=True)
class SliceCounts:
    missing_player_history_count: int
    available_player_history_count: int
    available_history_under_3_games_count: int
    available_history_at_least_3_games_count: int
    missing_last3_opportunity_share_count: int
    last3_opportunity_share_under_030_count: int
    last3_opportunity_share_at_least_030_count: int


@dataclass(frozen=True)
class RBPartitionAudit:
    name: str
    row_count: int
    unique_player_count: int
    unique_game_count: int
    minimum_period: tuple[int, int] | None
    maximum_period: tuple[int, int] | None
    feature_diagnostics: tuple[FeatureDiagnostic, ...]
    player_missing_history_count: int
    defense_missing_history_count: int
    player_zero_history_count: int
    defense_zero_history_count: int
    player_low_history_count: int
    defense_low_history_count: int
    invalid_history_count: int
    out_of_range_share_count: int
    team_change_counts: TeamChangeCounts
    slice_counts: SliceCounts
    workload_and_share_distributions: tuple[NumericDistribution, ...]


@dataclass(frozen=True)
class TrainingFeatureSummary:
    feature_name: str
    feature_category: str
    dtype: str
    non_missing_count: int
    missing_count: int
    nonfinite_count: int | None
    distribution: NumericDistribution | None


@dataclass(frozen=True)
class TrainingTargetSummary:
    """Finite training targets; standard deviation is sample ddof=1 or None."""

    row_count: int
    mean: float | None
    standard_deviation: float | None
    minimum: float | None
    percentile_25: float | None
    median: float | None
    percentile_75: float | None
    maximum: float | None
    zero_count: int
    negative_count: int


@dataclass(frozen=True)
class RBRushingYardsDatasetAudit:
    """Scalar report retaining no dataframe or validation/test target values."""

    partition_audits: tuple[RBPartitionAudit, ...]
    training_feature_summaries: tuple[TrainingFeatureSummary, ...]
    training_target_summary: TrainingTargetSummary


def audit_rb_rushing_yards_dataset(
    dataset_split: RBRushingYardsDatasetSplit,
) -> RBRushingYardsDatasetAudit:
    """Return deterministic structural diagnostics for an existing RB split.

    This audit reports anomalies; it never corrects, filters, imputes, or
    treats the report as proof that the feature builder is leakage-free.
    """
    if not isinstance(dataset_split, RBRushingYardsDatasetSplit):
        raise TypeError("dataset_split must be an RBRushingYardsDatasetSplit")

    partitions = (
        ("train", dataset_split.train),
        ("validation", dataset_split.validation),
        ("test", dataset_split.test),
    )
    allowed = tuple((name, _allowed_partition(data, name)) for name, data in partitions)
    _validate_partitions(allowed)
    _validate_cross_partition_keys(allowed)

    train = dataset_split.train
    _validate_training_target(train)
    return RBRushingYardsDatasetAudit(
        partition_audits=tuple(_partition_audit(name, data) for name, data in allowed),
        training_feature_summaries=tuple(
            _training_feature_summary(train, feature) for feature in FEATURE_COLUMNS
        ),
        training_target_summary=_training_target_summary(train),
    )


def _allowed_partition(data: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"{name} partition must be a pandas DataFrame")
    missing = sorted(set(ALLOWED_PARTITION_COLUMNS).difference(data.columns))
    if missing:
        raise ValueError(f"{name} partition is missing required columns: {missing}")
    # This selection deliberately excludes TARGET_COLUMN for every partition.
    return data.loc[:, ALLOWED_PARTITION_COLUMNS]


def _validate_partitions(partitions: tuple[tuple[str, pd.DataFrame], ...]) -> None:
    for name, data in partitions:
        if data.columns[data.columns.duplicated()].any():
            raise ValueError(f"{name} partition has duplicate columns")
        if data[TARGET_KEY].isna().any().any():
            raise ValueError(f"{name} partition has missing target-key values")
        if data.duplicated(TARGET_KEY).any():
            raise ValueError(f"{name} partition has duplicate target keys")
        for column in ("season", "week"):
            for value in data[column]:
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise ValueError(f"{name} partition has invalid {column} values")
                if not np.isfinite(value) or int(value) != value or value < 1:
                    raise ValueError(f"{name} partition has invalid {column} values")


def _validate_cross_partition_keys(partitions: tuple[tuple[str, pd.DataFrame], ...]) -> None:
    keys = [set(map(tuple, data[TARGET_KEY].to_numpy())) for _, data in partitions]
    if keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2]:
        raise ValueError("RB dataset split partitions have overlapping target keys")
    ranges = [(name, _period_range(data)) for name, data in partitions]
    for earlier_index, (earlier_name, (_, earlier_maximum)) in enumerate(ranges):
        if earlier_maximum is None:
            continue
        for later_name, (later_minimum, _) in ranges[earlier_index + 1:]:
            if later_minimum is not None and not earlier_maximum < later_minimum:
                raise ValueError(
                    f"{earlier_name} partition must chronologically precede {later_name}"
                )


def _validate_training_target(data: pd.DataFrame) -> None:
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"train partition is missing required columns: ['{TARGET_COLUMN}']")
    values = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    if values.isna().any() or not np.isfinite(values).all():
        raise ValueError("train partition has non-finite target_rushing_yards values")


def _partition_audit(name: str, data: pd.DataFrame) -> RBPartitionAudit:
    row_count = len(data)
    player_history = _numeric(data["rb_season_history_games"])
    defense_history = _numeric(data["defense_season_history_games"])
    player_missing = _true_count(data["rb_missing_history"])
    defense_missing = _true_count(data["defense_missing_history"])
    last3_share = _numeric(data["rb_last3_opportunity_share_avg"])
    available = ~data["rb_missing_history"].isna() & data["rb_missing_history"].eq(False)
    return RBPartitionAudit(
        name=name,
        row_count=row_count,
        unique_player_count=int(data["player_id"].nunique()),
        unique_game_count=int(data["game_id"].nunique()),
        minimum_period=_period_range(data)[0],
        maximum_period=_period_range(data)[1],
        feature_diagnostics=tuple(_feature_diagnostic(data, feature) for feature in FEATURE_COLUMNS),
        player_missing_history_count=player_missing,
        defense_missing_history_count=defense_missing,
        player_zero_history_count=int(player_history.eq(0).sum()),
        defense_zero_history_count=int(defense_history.eq(0).sum()),
        player_low_history_count=int((player_history.gt(0) & player_history.lt(3)).sum()),
        defense_low_history_count=int((defense_history.gt(0) & defense_history.lt(3)).sum()),
        invalid_history_count=sum(_invalid_history_count(data[column]) for column in HISTORY_COUNT_COLUMNS),
        out_of_range_share_count=sum(_out_of_range_share_count(data[column]) for column in SHARE_COLUMNS),
        team_change_counts=_team_change_counts(data["player_team_changed"]),
        slice_counts=SliceCounts(
            missing_player_history_count=player_missing,
            available_player_history_count=int(available.sum()),
            available_history_under_3_games_count=int(
                (available & player_history.gt(0) & player_history.lt(3)).sum()
            ),
            available_history_at_least_3_games_count=int(
                (available & player_history.ge(3)).sum()
            ),
            missing_last3_opportunity_share_count=int(data["rb_last3_opportunity_share_avg"].isna().sum()),
            last3_opportunity_share_under_030_count=int(last3_share.lt(0.30).sum()),
            last3_opportunity_share_at_least_030_count=int(last3_share.ge(0.30).sum()),
        ),
        workload_and_share_distributions=tuple(
            _distribution(data[feature], feature) for feature in WORKLOAD_AND_SHARE_COLUMNS
        ),
    )


def _feature_diagnostic(data: pd.DataFrame, feature: str) -> FeatureDiagnostic:
    values = data[feature]
    numeric = _numeric(values)
    return FeatureDiagnostic(
        feature_name=feature,
        missing_count=int(values.isna().sum()),
        missing_rate=_rate(int(values.isna().sum()), len(values)),
        nonfinite_count=int((~values.isna() & numeric.notna() & ~np.isfinite(numeric)).sum()),
    )


def _training_feature_summary(data: pd.DataFrame, feature: str) -> TrainingFeatureSummary:
    values = data[feature]
    if pd.api.types.is_bool_dtype(values):
        return TrainingFeatureSummary(feature, "boolean", str(values.dtype), int(values.notna().sum()), int(values.isna().sum()), None, None)
    numeric = _numeric(values)
    nonfinite = int((~values.isna() & numeric.notna() & ~np.isfinite(numeric)).sum())
    return TrainingFeatureSummary(
        feature, "numeric", str(values.dtype), int(values.notna().sum()), int(values.isna().sum()),
        nonfinite, _distribution(values, feature),
    )


def _training_target_summary(data: pd.DataFrame) -> TrainingTargetSummary:
    values = pd.to_numeric(data[TARGET_COLUMN], errors="raise").astype(float)
    distribution = _distribution(values, TARGET_COLUMN)
    return TrainingTargetSummary(
        row_count=len(values), mean=float(values.mean()) if len(values) else None,
        standard_deviation=float(values.std(ddof=1)) if len(values) > 1 else None,
        minimum=distribution.minimum, percentile_25=distribution.percentile_25,
        median=distribution.median, percentile_75=distribution.percentile_75,
        maximum=distribution.maximum, zero_count=int(values.eq(0).sum()),
        negative_count=int(values.lt(0).sum()),
    )


def _distribution(values: pd.Series, name: str) -> NumericDistribution:
    numeric = _numeric(values)
    finite = numeric[np.isfinite(numeric)]
    return NumericDistribution(
        name, len(finite), float(finite.min()) if len(finite) else None,
        float(finite.quantile(0.25)) if len(finite) else None,
        float(finite.median()) if len(finite) else None,
        float(finite.quantile(0.75)) if len(finite) else None,
        float(finite.max()) if len(finite) else None,
    )


def _numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").astype(float)


def _period_range(data: pd.DataFrame) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    if data.empty:
        return None, None
    periods = list(zip(data["season"].astype(int), data["week"].astype(int), strict=True))
    return min(periods), max(periods)


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def _true_count(values: pd.Series) -> int:
    return int(values.eq(True).sum())


def _invalid_history_count(values: pd.Series) -> int:
    numeric = _numeric(values)
    invalid = numeric.isna() | ~np.isfinite(numeric) | numeric.lt(0) | (numeric % 1 != 0)
    return int(invalid.sum())


def _out_of_range_share_count(values: pd.Series) -> int:
    numeric = _numeric(values)
    return int((np.isfinite(numeric) & (numeric.lt(0) | numeric.gt(1))).sum())


def _team_change_counts(values: pd.Series) -> TeamChangeCounts:
    valid = values.map(lambda value: isinstance(value, (bool, np.bool_)))
    return TeamChangeCounts(
        changed_count=int((valid & values.eq(True)).sum()),
        unchanged_count=int((valid & values.eq(False)).sum()),
        unknown_count=int(values.isna().sum()), invalid_count=int((~values.isna() & ~valid).sum()),
    )
