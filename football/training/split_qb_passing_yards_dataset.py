"""Chronologically partition the completed QB passing-yards training dataset.

This module only assigns already-built rows to temporal partitions.  It does
not recalculate point-in-time features or inspect target values.  Any learned
preprocessing, fallback values, feature decisions, and model fitting must be
fit using the training partition only.  Validation supports those future
decisions; the test partition remains untouched until final comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

from football.training.build_qb_passing_yards_dataset import (
    OUTPUT_COLUMNS,
    TARGET_KEY,
)


SORT_COLUMNS = ["season", "week", "game_id", "player_id"]


@dataclass(frozen=True)
class QBPassingYardsDatasetSplit:
    """Independent chronological train, validation, and test dataframes."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_qb_passing_yards_dataset(
    dataset: pd.DataFrame,
    *,
    validation_start: tuple[int, int],
    test_start: tuple[int, int],
) -> QBPassingYardsDatasetSplit:
    """Split a built QB dataset using inclusive validation and test starts.

    ``(season, week)`` pairs compare lexicographically.  Rows before
    ``validation_start`` are train rows; rows from ``validation_start`` up to
    (but excluding) ``test_start`` are validation rows; rows at or after
    ``test_start`` are test rows.  The starts must be positive integer-like
    pairs and ``validation_start`` must be earlier than ``test_start``.
    """

    validation_boundary = _validate_boundary(validation_start, "validation_start")
    test_boundary = _validate_boundary(test_start, "test_start")
    if validation_boundary >= test_boundary:
        raise ValueError("validation_start must be chronologically earlier than test_start")

    _validate_dataset(dataset)
    data = dataset.copy(deep=True)
    temporal_pairs = list(zip(data["season"], data["week"], strict=True))

    train_mask = [pair < validation_boundary for pair in temporal_pairs]
    validation_mask = [
        validation_boundary <= pair < test_boundary for pair in temporal_pairs
    ]
    test_mask = [pair >= test_boundary for pair in temporal_pairs]

    train = _ordered_copy(data.loc[train_mask])
    validation = _ordered_copy(data.loc[validation_mask])
    test = _ordered_copy(data.loc[test_mask])
    _validate_nonempty_partitions(train, validation, test)

    return QBPassingYardsDatasetSplit(
        train=train,
        validation=validation,
        test=test,
    )


def _validate_boundary(value: object, name: str) -> tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{name} must be a (season, week) tuple")

    season = _validate_temporal_value(value[0], f"{name} season")
    week = _validate_temporal_value(value[1], f"{name} week")
    return season, week


def _validate_dataset(dataset: pd.DataFrame) -> None:
    missing_columns = sorted(set(OUTPUT_COLUMNS).difference(dataset.columns))
    if missing_columns:
        raise ValueError(
            "QB passing-yards training dataset is missing required columns: "
            f"{missing_columns}"
        )

    for column in ("season", "week"):
        for value in dataset[column]:
            _validate_temporal_value(value, f"dataset {column}")

    if dataset[TARGET_KEY].isna().any().any():
        raise ValueError("QB passing-yards training dataset has missing target-key values")

    duplicates = dataset.loc[dataset.duplicated(TARGET_KEY, keep=False), TARGET_KEY]
    if not duplicates.empty:
        keys = (
            duplicates.drop_duplicates()
            .sort_values(SORT_COLUMNS, kind="mergesort")
            .to_dict(orient="records")
        )
        raise ValueError(f"QB passing-yards training dataset has duplicate target keys: {keys}")


def _validate_temporal_value(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive integer-like value")
    if not np.isfinite(value) or int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive integer-like value")
    return int(value)


def _ordered_copy(data: pd.DataFrame) -> pd.DataFrame:
    return data.sort_values(SORT_COLUMNS, kind="mergesort").copy(deep=True).reset_index(
        drop=True
    )


def _validate_nonempty_partitions(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    empty_names = [
        name
        for name, partition in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        )
        if partition.empty
    ]
    if empty_names:
        raise ValueError(
            "Chronological split produced empty partition(s): "
            f"{', '.join(empty_names)}"
        )
