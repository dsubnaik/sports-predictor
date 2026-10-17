import numpy as np
import pandas as pd
import pytest

from football.data.build_running_back_dataset import OUTPUT_COLUMNS as RB_GAME_COLUMNS
from football.training.build_rb_rushing_yards_dataset import (
    FEATURE_COLUMNS,
    OUTPUT_COLUMNS,
    TARGET_COLUMN,
    TARGET_KEY,
    build_rb_rushing_yards_training_dataset,
)
from football.training.split_rb_rushing_yards_dataset import (
    RBRushingYardsDatasetSplit,
    split_rb_rushing_yards_dataset,
)


def _dataset() -> pd.DataFrame:
    rows = []
    for season, week, game_id, player_id, yards in [
        (2020, 17, "old", "old-rb", 12.0),
        (2021, 1, "first", "rb-a", 0.0),
        (2024, 18, "train", "rb-b", 50.0),
        (2025, 1, "validation", "rb-a", -3.0),
        (2025, 8, "shared", "rb-a", 62.0),
        (2025, 8, "shared", "rb-b", 41.0),
        (2026, 1, "test", "rb-c", 80.0),
    ]:
        row = {
            "season": season,
            "week": week,
            "game_id": game_id,
            "player_id": player_id,
            "player_name": player_id,
            "team": "A",
            "opponent": "B",
            "home_away": "home",
            "rb_prior_latest_team": pd.NA,
            "player_team_changed": False,
            TARGET_COLUMN: yards,
        }
        row.update({column: float(index) for index, column in enumerate(FEATURE_COLUMNS)})
        rows.append(row)
    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    result["diagnostic"] = [f"diagnostic-{index}" for index in range(len(result))]
    result.loc[result["game_id"] == "validation", FEATURE_COLUMNS[0]] = np.nan
    return result


def _split(data: pd.DataFrame, **kwargs) -> RBRushingYardsDatasetSplit:
    return split_rb_rushing_yards_dataset(data, **kwargs)


def _keys(partition: pd.DataFrame) -> set[tuple[object, ...]]:
    return set(map(tuple, partition[TARGET_KEY].to_numpy()))


def test_default_boundaries_exclude_pre_minimum_and_include_exact_starts():
    result = _split(_dataset())

    assert isinstance(result, RBRushingYardsDatasetSplit)
    assert result.train["game_id"].tolist() == ["first", "train"]
    assert result.validation["game_id"].tolist() == ["validation", "shared", "shared"]
    assert result.test["game_id"].tolist() == ["test"]
    assert "old" not in pd.concat([result.train, result.validation, result.test])["game_id"].tolist()


def test_custom_within_season_boundaries_are_exact():
    result = _split(
        _dataset(), minimum_target_season=2021, validation_start=(2025, 8), test_start=(2025, 9)
    )

    assert result.train["game_id"].tolist() == ["first", "train", "validation"]
    assert result.validation["game_id"].tolist() == ["shared", "shared"]
    assert result.test["game_id"].tolist() == ["test"]


def test_all_players_in_one_game_remain_together_and_eligible_rows_have_complete_coverage():
    data = _dataset()
    result = _split(data)
    partitions = (result.train, result.validation, result.test)
    shared_partitions = [partition for partition in partitions if "shared" in partition["game_id"].tolist()]

    assert len(shared_partitions) == 1
    assert shared_partitions[0].loc[shared_partitions[0]["game_id"] == "shared", "player_id"].tolist() == ["rb-a", "rb-b"]
    partition_keys = [_keys(partition) for partition in partitions]
    assert not partition_keys[0] & partition_keys[1]
    assert not partition_keys[0] & partition_keys[2]
    assert not partition_keys[1] & partition_keys[2]
    assert set.union(*partition_keys) == _keys(data.loc[data["season"] >= 2021])


def test_membership_depends_only_on_chronology_and_minimum_target_season():
    data = _dataset()
    changed = data.copy(deep=True)
    changed[TARGET_COLUMN] = np.arange(len(changed)) * -100.0
    changed[FEATURE_COLUMNS[0]] = np.nan
    changed["diagnostic"] = "changed"

    original = _split(data)
    actual = _split(changed)
    for expected, partition in zip(
        (original.train, original.validation, original.test),
        (actual.train, actual.validation, actual.test),
        strict=True,
    ):
        assert _keys(expected) == _keys(partition)


def test_zero_negative_targets_missing_features_and_all_columns_are_preserved():
    data = _dataset()
    result = _split(data)

    assert result.validation.columns.tolist() == data.columns.tolist()
    assert result.train.loc[result.train["game_id"] == "first", TARGET_COLUMN].iloc[0] == 0.0
    assert result.validation.loc[result.validation["game_id"] == "validation", TARGET_COLUMN].iloc[0] == -3.0
    assert pd.isna(result.validation.loc[result.validation["game_id"] == "validation", FEATURE_COLUMNS[0]].iloc[0])


def test_shuffled_input_is_deterministic_and_input_is_not_mutated():
    data = _dataset()
    original = data.copy(deep=True)
    expected = _split(data)
    actual = _split(data.sample(frac=1, random_state=11))

    for expected_partition, actual_partition in zip(
        (expected.train, expected.validation, expected.test),
        (actual.train, actual.validation, actual.test),
        strict=True,
    ):
        pd.testing.assert_frame_equal(actual_partition, expected_partition)
    pd.testing.assert_frame_equal(data, original)


def test_empty_input_and_empty_partitions_preserve_schema():
    empty = pd.DataFrame(columns=[*OUTPUT_COLUMNS, "diagnostic"])
    result = _split(empty)
    for partition in (result.train, result.validation, result.test):
        assert partition.empty
        assert partition.columns.tolist() == empty.columns.tolist()

    only_train = _split(_dataset().query("season < 2025 and season >= 2021"))
    assert only_train.validation.empty and only_train.test.empty
    assert only_train.validation.columns.tolist() == OUTPUT_COLUMNS + ["diagnostic"]


@pytest.mark.parametrize(
    ("validation_start", "test_start"),
    [((2025, 1), (2025, 1)), ((2025, 2), (2025, 1))],
)
def test_rejects_reversed_boundaries(validation_start, test_start):
    with pytest.raises(ValueError, match="chronologically earlier"):
        _split(_dataset(), validation_start=validation_start, test_start=test_start)


@pytest.mark.parametrize("boundary", [True, (2025,), [2025, 1], (2025, False), (2025, 1.5)])
def test_rejects_invalid_boundaries(boundary):
    with pytest.raises(ValueError, match="validation_start"):
        _split(_dataset(), validation_start=boundary)


@pytest.mark.parametrize("minimum_target_season", [True, 2020.5, np.inf])
def test_rejects_invalid_minimum_target_season(minimum_target_season):
    with pytest.raises(ValueError, match="minimum_target_season"):
        _split(_dataset(), minimum_target_season=minimum_target_season)


@pytest.mark.parametrize("column", ["season", "week", "game_id", "player_id", TARGET_COLUMN])
def test_rejects_missing_required_columns(column):
    with pytest.raises(ValueError, match="missing required columns"):
        _split(_dataset().drop(columns=column))


@pytest.mark.parametrize("column", ["season", "week"])
@pytest.mark.parametrize("value", [None, np.nan, np.inf, 2025.5, True])
def test_rejects_malformed_chronology(column, value):
    data = _dataset().astype({column: object})
    data.loc[0, column] = value
    with pytest.raises(ValueError, match=f"dataset {column}"):
        _split(data)


def test_rejects_duplicate_keys_and_inconsistent_game_chronology():
    data = _dataset()
    with pytest.raises(ValueError, match="duplicate target keys"):
        _split(pd.concat([data, data.iloc[[0]]], ignore_index=True))

    inconsistent = data.copy(deep=True)
    inconsistent.loc[inconsistent["game_id"] == "shared", "week"] = [8, 9]
    with pytest.raises(ValueError, match="inconsistent game chronology"):
        _split(inconsistent)


def test_rejects_missing_target_key_values():
    data = _dataset()
    data.loc[0, "player_id"] = pd.NA
    with pytest.raises(ValueError, match="missing target-key values"):
        _split(data)


def _rb_game(season, week, game_id, player_id, yards):
    return {
        "season": season, "week": week, "game_id": game_id, "player_id": player_id,
        "player_name": player_id, "team": "A", "opponent": "B", "position": "RB",
        "rushing_attempts": 1, "rushing_yards": yards, "rushing_touchdowns": 0,
        "receptions": 0, "targets": 0, "receiving_yards": 0, "receiving_touchdowns": 0,
    }


def test_integration_with_public_rb_dataset_builder_uses_synthetic_fixtures_only():
    games = [
        _rb_game(2024, 18, "g2024", "rb", 20),
        _rb_game(2025, 1, "g2025", "rb", 0),
        _rb_game(2026, 1, "g2026", "rb", -1),
    ]
    schedule = pd.DataFrame([
        {"season": row["season"], "week": row["week"], "game_id": row["game_id"], "team": "A", "opponent": "B", "home_away": "home"}
        for row in games
    ])
    built = build_rb_rushing_yards_training_dataset(
        pd.DataFrame(games, columns=RB_GAME_COLUMNS), schedule
    )
    result = _split(built)

    assert result.train["game_id"].tolist() == ["g2024"]
    assert result.validation["game_id"].tolist() == ["g2025"]
    assert result.test["game_id"].tolist() == ["g2026"]
