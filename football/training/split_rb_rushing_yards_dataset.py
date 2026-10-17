"""Chronologically partition an already-built RB rushing-yards dataset.

This module only assigns rows to temporal partitions.  It does not recalculate
features, inspect target values, impute missing values, or fit a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

from football.training.build_rb_rushing_yards_dataset import OUTPUT_COLUMNS, TARGET_KEY


DEFAULT_MINIMUM_TARGET_SEASON = 2021
DEFAULT_VALIDATION_START = (2025, 1)
DEFAULT_TEST_START = (2026, 1)
SORT_COLUMNS = ["season", "week", "game_id", "player_id"]


@dataclass(frozen=True)
class RBRushingYardsDatasetSplit:
    """Independent chronological train, validation, and test dataframes."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_rb_rushing_yards_dataset(
    dataset: pd.DataFrame,
    *,
    minimum_target_season: int = DEFAULT_MINIMUM_TARGET_SEASON,
    validation_start: tuple[int, int] = DEFAULT_VALIDATION_START,
    test_start: tuple[int, int] = DEFAULT_TEST_START,
) -> RBRushingYardsDatasetSplit:
    """Split RB rows by ``(season, week)`` without changing their values.

    Rows before ``minimum_target_season`` are excluded.  Eligible rows before
    ``validation_start`` are train rows; rows from ``validation_start`` up to
    (but excluding) ``test_start`` are validation rows; later rows are test
    rows.  Empty inputs and partitions are valid and preserve input columns.
    """

    minimum_season = _validate_temporal_value(
        minimum_target_season, "minimum_target_season"
    )
    validation_boundary = _validate_boundary(validation_start, "validation_start")
    test_boundary = _validate_boundary(test_start, "test_start")
    if validation_boundary >= test_boundary:
        raise ValueError("validation_start must be chronologically earlier than test_start")

    _validate_dataset(dataset)
    data = dataset.copy(deep=True)
    eligible = data.loc[data["season"] >= minimum_season].copy(deep=True)
    temporal_pairs = list(zip(eligible["season"], eligible["week"], strict=True))

    train = _ordered_copy(eligible.loc[[pair < validation_boundary for pair in temporal_pairs]])
    validation = _ordered_copy(
        eligible.loc[
            [validation_boundary <= pair < test_boundary for pair in temporal_pairs]
        ]
    )
    test = _ordered_copy(eligible.loc[[pair >= test_boundary for pair in temporal_pairs]])

    return RBRushingYardsDatasetSplit(train=train, validation=validation, test=test)


def _validate_boundary(value: object, name: str) -> tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{name} must be a (season, week) tuple")
    return (
        _validate_temporal_value(value[0], f"{name} season"),
        _validate_temporal_value(value[1], f"{name} week"),
    )


def _validate_dataset(dataset: pd.DataFrame) -> None:
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("RB rushing-yards training dataset must be a pandas DataFrame")
    missing_columns = sorted(set(OUTPUT_COLUMNS).difference(dataset.columns))
    if missing_columns:
        raise ValueError(
            "RB rushing-yards training dataset is missing required columns: "
            f"{missing_columns}"
        )
    duplicate_columns = dataset.columns[dataset.columns.duplicated()].tolist()
    if duplicate_columns:
        raise ValueError(
            "RB rushing-yards training dataset has duplicate columns: "
            f"{duplicate_columns}"
        )

    for column in ("season", "week"):
        for value in dataset[column]:
            _validate_temporal_value(value, f"dataset {column}")

    if dataset[TARGET_KEY].isna().any().any():
        raise ValueError("RB rushing-yards training dataset has missing target-key values")
    if dataset.duplicated(TARGET_KEY).any():
        raise ValueError("RB rushing-yards training dataset has duplicate target keys")

    game_chronology = dataset.groupby("game_id", dropna=False)[["season", "week"]].nunique()
    inconsistent_games = game_chronology.loc[
        (game_chronology["season"] > 1) | (game_chronology["week"] > 1)
    ]
    if not inconsistent_games.empty:
        game_ids = sorted(inconsistent_games.index.tolist(), key=str)
        raise ValueError(
            "RB rushing-yards training dataset has inconsistent game chronology: "
            f"{game_ids}"
        )


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
