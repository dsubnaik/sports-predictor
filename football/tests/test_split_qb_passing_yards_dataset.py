import numpy as np
import pandas as pd
import pytest

from football.training.build_qb_passing_yards_dataset import (
    FEATURE_COLUMNS,
    OUTPUT_COLUMNS,
    TARGET_COLUMN,
    TARGET_KEY,
)
from football.training.split_qb_passing_yards_dataset import (
    QBPassingYardsDatasetSplit,
    split_qb_passing_yards_dataset,
)


def _dataset() -> pd.DataFrame:
    rows = []
    for season, week, game_id, player_id, yards in [
        (2025, 1, "2025_01_B", "qb_b", 301.0),
        (2023, 18, "2023_18_A", "qb_a", 198.0),
        (2024, 2, "2024_02_C", "qb_c", 222.0),
        (2025, 2, "2025_02_D", "qb_d", 244.0),
        (2024, 1, "2024_01_B", "qb_b", 211.0),
        (2023, 1, "2023_01_A", "qb_a", 175.0),
    ]:
        row = {
            "season": season,
            "week": week,
            "game_id": game_id,
            "player_id": player_id,
            "player_name": f"{player_id} name",
            "team": "AAA",
            "opponent": "BBB",
            "home_away": "home",
            TARGET_COLUMN: yards,
        }
        row.update({column: float(index) for index, column in enumerate(FEATURE_COLUMNS)})
        rows.append(row)
    data = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    data["extra_audit_column"] = [f"audit-{index}" for index in range(len(data))]
    return data


def _split(data: pd.DataFrame):
    return split_qb_passing_yards_dataset(
        data,
        validation_start=(2024, 1),
        test_start=(2025, 1),
    )


def test_assigns_exact_boundary_rows_to_validation_and_test():
    result = _split(_dataset())

    assert isinstance(result, QBPassingYardsDatasetSplit)
    assert result.train["game_id"].tolist() == ["2023_01_A", "2023_18_A"]
    assert result.validation["game_id"].tolist() == ["2024_01_B", "2024_02_C"]
    assert result.test["game_id"].tolist() == ["2025_01_B", "2025_02_D"]
    assert result.train[["season", "week"]].max().tolist() < [2024, 1]
    assert (result.validation["season"] >= 2024).all()
    assert (result.test["season"] >= 2025).all()


def test_preserves_all_columns_values_and_deterministic_tie_breaking():
    data = _dataset()
    extra = data.iloc[[2]].copy()
    extra["game_id"] = "2024_02_A"
    extra["player_id"] = "qb_a"
    extra[TARGET_COLUMN] = 999.0
    extra[FEATURE_COLUMNS[0]] = 123.5
    data = pd.concat([data, extra], ignore_index=True)

    result = _split(data)

    assert result.validation.columns.tolist() == data.columns.tolist()
    assert result.validation[["season", "week", "game_id", "player_id"]].values.tolist() == [
        [2024, 1, "2024_01_B", "qb_b"],
        [2024, 2, "2024_02_A", "qb_a"],
        [2024, 2, "2024_02_C", "qb_c"],
    ]
    changed = result.validation.loc[result.validation["game_id"] == "2024_02_A"].iloc[0]
    assert changed[TARGET_COLUMN] == 999.0
    assert changed[FEATURE_COLUMNS[0]] == 123.5


def test_shuffled_input_does_not_change_outputs_or_mutate_caller():
    data = _dataset()
    original = data.copy(deep=True)

    expected = _split(data)
    actual = _split(data.sample(frac=1, random_state=7))

    for expected_partition, actual_partition in zip(
        (expected.train, expected.validation, expected.test),
        (actual.train, actual.validation, actual.test),
        strict=True,
    ):
        pd.testing.assert_frame_equal(actual_partition, expected_partition)
    pd.testing.assert_frame_equal(data, original)


def test_partitions_are_independent_and_reconstruct_all_unique_keys():
    data = _dataset()
    result = _split(data)

    result.train.loc[0, "player_name"] = "changed"
    assert data.loc[data["game_id"] == "2023_01_A", "player_name"].iloc[0] == "qb_a name"
    assert "changed" not in result.validation["player_name"].tolist()
    assert "changed" not in result.test["player_name"].tolist()

    all_keys = pd.concat(
        [result.train[TARGET_KEY], result.validation[TARGET_KEY], result.test[TARGET_KEY]],
        ignore_index=True,
    )
    assert not all_keys.duplicated().any()
    assert set(map(tuple, all_keys.to_numpy())) == set(map(tuple, data[TARGET_KEY].to_numpy()))


@pytest.mark.parametrize(
    ("validation_start", "test_start"),
    [((2025, 1), (2025, 1)), ((2025, 2), (2025, 1))],
)
def test_rejects_nonchronological_boundaries(validation_start, test_start):
    with pytest.raises(ValueError, match="chronologically earlier"):
        split_qb_passing_yards_dataset(
            _dataset(), validation_start=validation_start, test_start=test_start
        )


@pytest.mark.parametrize(
    "boundary",
    [True, (True, 1), (2024, False), (2024, 1.5), (np.inf, 1), (2024,), [2024, 1]],
)
def test_rejects_invalid_boundary_values(boundary):
    with pytest.raises(ValueError, match="validation_start"):
        split_qb_passing_yards_dataset(
            _dataset(), validation_start=boundary, test_start=(2025, 1)
        )


@pytest.mark.parametrize(
    ("validation_start", "test_start", "empty_name"),
    [((2023, 1), (2025, 1), "train"), ((2024, 3), (2025, 1), "validation"), ((2024, 1), (2026, 1), "test")],
)
def test_rejects_empty_partitions(validation_start, test_start, empty_name):
    with pytest.raises(ValueError, match=empty_name):
        split_qb_passing_yards_dataset(
            _dataset(), validation_start=validation_start, test_start=test_start
        )


@pytest.mark.parametrize("column", ["season", "week", "game_id", TARGET_COLUMN])
def test_rejects_missing_required_columns(column):
    with pytest.raises(ValueError, match="missing required columns"):
        _split(_dataset().drop(columns=column))


@pytest.mark.parametrize("column", ["season", "week"])
@pytest.mark.parametrize("value", [None, np.nan, np.inf, 2024.5, True])
def test_rejects_invalid_dataset_temporal_values(column, value):
    data = _dataset()
    data[column] = data[column].astype(object)
    data.loc[0, column] = value

    with pytest.raises(ValueError, match=f"dataset {column}"):
        _split(data)


@pytest.mark.parametrize("conflicting", [False, True])
def test_rejects_exact_and_conflicting_duplicate_target_keys(conflicting):
    data = _dataset()
    duplicate = data.iloc[[0]].copy()
    if conflicting:
        duplicate[TARGET_COLUMN] = 999.0
    data = pd.concat([data, duplicate], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate target keys"):
        _split(data)


def test_target_values_do_not_change_membership_or_features():
    data = _dataset()
    changed = data.copy(deep=True)
    changed[TARGET_COLUMN] = np.arange(len(changed)) * 1000.0
    changed[FEATURE_COLUMNS[0]] = np.arange(len(changed)) * -1.0

    original_split = _split(data)
    changed_split = _split(changed)

    for original_partition, changed_partition in zip(
        (original_split.train, original_split.validation, original_split.test),
        (changed_split.train, changed_split.validation, changed_split.test),
        strict=True,
    ):
        assert original_partition[TARGET_KEY].values.tolist() == changed_partition[TARGET_KEY].values.tolist()
    assert changed_split.validation[FEATURE_COLUMNS[0]].tolist() != original_split.validation[FEATURE_COLUMNS[0]].tolist()
