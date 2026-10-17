from dataclasses import fields

import numpy as np
import pandas as pd
import pytest

from football.data.build_running_back_dataset import OUTPUT_COLUMNS as RB_GAME_COLUMNS
from football.training.audit_rb_rushing_yards_dataset import (
    RBRushingYardsDatasetAudit,
    audit_rb_rushing_yards_dataset,
)
from football.training.build_rb_rushing_yards_dataset import (
    FEATURE_COLUMNS, OUTPUT_COLUMNS, TARGET_COLUMN, build_rb_rushing_yards_training_dataset,
)
from football.training.split_rb_rushing_yards_dataset import RBRushingYardsDatasetSplit, split_rb_rushing_yards_dataset


def _row(season, week, game, player, target, history, share, changed=False):
    row = {
        "season": season, "week": week, "game_id": game, "player_id": player,
        "player_name": player, "team": "A", "opponent": "B", "home_away": "home",
        "rb_prior_latest_team": pd.NA, "player_team_changed": changed, TARGET_COLUMN: target,
    }
    row.update({feature: 1.0 for feature in FEATURE_COLUMNS})
    row.update({
        "rb_season_history_games": history, "rb_last3_history_games": history,
        "defense_season_history_games": history, "defense_last3_history_games": history,
        "rb_missing_history": history == 0, "defense_missing_history": history == 0,
        "rb_last3_opportunity_share_avg": share,
    })
    return row


def _split():
    train = pd.DataFrame([
        _row(2024, 1, "tr1", "a", 0.0, 0, np.nan, pd.NA),
        _row(2024, 2, "tr2", "b", -2.0, 2, 0.20, False),
        _row(2024, 3, "tr3", "c", 10.0, 3, 0.30, True),
    ], columns=OUTPUT_COLUMNS)
    validation = pd.DataFrame([_row(2025, 1, "va", "d", 999.0, 1, 0.50)], columns=OUTPUT_COLUMNS)
    test = pd.DataFrame([_row(2026, 1, "te", "e", -999.0, 4, 0.10)], columns=OUTPUT_COLUMNS)
    return RBRushingYardsDatasetSplit(train, validation, test)


def _partition(report, name):
    return next(item for item in report.partition_audits if item.name == name)


def test_reports_structure_missingness_target_summary_and_pregame_slices():
    report = audit_rb_rushing_yards_dataset(_split())
    train = _partition(report, "train")
    assert isinstance(report, RBRushingYardsDatasetAudit)
    assert (train.row_count, train.unique_player_count, train.unique_game_count) == (3, 3, 3)
    assert (train.minimum_period, train.maximum_period) == ((2024, 1), (2024, 3))
    diagnostics = {item.feature_name: item for item in train.feature_diagnostics}
    assert diagnostics["rb_last3_opportunity_share_avg"].missing_count == 1
    assert diagnostics["rb_last3_opportunity_share_avg"].missing_rate == pytest.approx(1 / 3)
    assert train.player_zero_history_count == train.defense_zero_history_count == 1
    assert train.player_low_history_count == train.defense_low_history_count == 1
    assert train.slice_counts.missing_player_history_count == 1
    assert train.slice_counts.available_history_under_3_games_count == 1
    assert train.slice_counts.available_history_at_least_3_games_count == 1
    assert train.slice_counts.missing_last3_opportunity_share_count == 1
    assert train.slice_counts.last3_opportunity_share_under_030_count == 1
    assert train.slice_counts.last3_opportunity_share_at_least_030_count == 1
    assert train.team_change_counts.changed_count == 1
    assert train.team_change_counts.unchanged_count == 1
    assert train.team_change_counts.unknown_count == 1
    target = report.training_target_summary
    assert (target.row_count, target.zero_count, target.negative_count) == (3, 1, 1)
    assert (target.minimum, target.median, target.maximum) == (-2.0, 0.0, 10.0)
    assert target.standard_deviation is not None


def test_reports_infinities_invalid_counts_and_out_of_range_shares_without_mutation():
    split = _split()
    original = tuple(frame.copy(deep=True) for frame in (split.train, split.validation, split.test))
    changed = split.train.copy(deep=True)
    changed.loc[0, "rb_season_opportunities_avg"] = np.inf
    changed.loc[1, "rb_season_history_games"] = -1
    changed.loc[2, "rb_season_carry_share_avg"] = 1.2
    report = audit_rb_rushing_yards_dataset(RBRushingYardsDatasetSplit(changed, split.validation, split.test))
    train = _partition(report, "train")
    diagnostic = next(item for item in train.feature_diagnostics if item.feature_name == "rb_season_opportunities_avg")
    assert diagnostic.nonfinite_count == 1
    assert train.invalid_history_count == 1
    assert train.out_of_range_share_count == 1
    for actual, expected in zip((split.train, split.validation, split.test), original, strict=True):
        pd.testing.assert_frame_equal(actual, expected)


def test_validation_and_test_targets_are_never_required_or_read():
    split = _split()
    expected = audit_rb_rushing_yards_dataset(split)
    validation = split.validation.drop(columns=TARGET_COLUMN)
    test = split.test.drop(columns=TARGET_COLUMN)
    actual = audit_rb_rushing_yards_dataset(RBRushingYardsDatasetSplit(split.train, validation, test))
    assert actual == expected
    changed = RBRushingYardsDatasetSplit(split.train, split.validation.assign(target_rushing_yards=123456.0), split.test.assign(target_rushing_yards=-123456.0))
    assert audit_rb_rushing_yards_dataset(changed) == expected
    assert all("DataFrame" not in str(field.type) for field in fields(type(expected)))


def test_empty_and_single_row_partitions_have_explicit_conventions_and_are_deterministic():
    split = _split()
    empty = split.validation.iloc[0:0].drop(columns=TARGET_COLUMN)
    single = split.train.iloc[[0]].copy()
    report = audit_rb_rushing_yards_dataset(RBRushingYardsDatasetSplit(single, empty, empty.copy()))
    assert report.training_target_summary.standard_deviation is None
    assert _partition(report, "validation").minimum_period is None
    assert _partition(report, "validation").feature_diagnostics[0].missing_rate == 0.0
    shuffled = RBRushingYardsDatasetSplit(split.train.sample(frac=1, random_state=1), split.validation.sample(frac=1, random_state=2), split.test.sample(frac=1, random_state=3))
    assert audit_rb_rushing_yards_dataset(shuffled) == audit_rb_rushing_yards_dataset(split)


def test_all_missing_features_are_reported_without_imputation():
    split = _split()
    all_missing = split.validation.copy()
    all_missing["rb_season_rushing_yards_avg"] = np.nan
    report = audit_rb_rushing_yards_dataset(RBRushingYardsDatasetSplit(split.train, all_missing, split.test))
    diagnostic = next(item for item in _partition(report, "validation").feature_diagnostics if item.feature_name == "rb_season_rushing_yards_avg")
    assert (diagnostic.missing_count, diagnostic.missing_rate, diagnostic.nonfinite_count) == (1, 1.0, 0)


def test_rejects_invalid_type_keys_and_nonfinite_training_target_only():
    split = _split()
    with pytest.raises(TypeError, match="RBRushingYardsDatasetSplit"):
        audit_rb_rushing_yards_dataset(split.train)
    duplicate = pd.concat([split.train, split.train.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate target keys"):
        audit_rb_rushing_yards_dataset(RBRushingYardsDatasetSplit(duplicate, split.validation, split.test))
    invalid = split.train.copy()
    invalid[TARGET_COLUMN] = np.nan
    with pytest.raises(ValueError, match="non-finite target_rushing_yards"):
        audit_rb_rushing_yards_dataset(RBRushingYardsDatasetSplit(invalid, split.validation, split.test))


@pytest.mark.parametrize("test_period", [(2023, 1), (2024, 3)])
def test_rejects_reversed_or_overlapping_train_test_chronology_with_empty_validation(test_period):
    split = _split()
    empty_validation = split.validation.iloc[0:0].drop(columns=TARGET_COLUMN)
    test = split.test.copy()
    test.loc[:, ["season", "week"]] = test_period
    with pytest.raises(ValueError, match="train partition must chronologically precede test"):
        audit_rb_rushing_yards_dataset(RBRushingYardsDatasetSplit(split.train, empty_validation, test))


def test_accepts_ordered_train_test_chronology_with_empty_validation():
    split = _split()
    empty_validation = split.validation.iloc[0:0].drop(columns=TARGET_COLUMN)
    report = audit_rb_rushing_yards_dataset(
        RBRushingYardsDatasetSplit(split.train, empty_validation, split.test)
    )
    assert _partition(report, "test").minimum_period == (2026, 1)


def test_team_change_buckets_are_exclusive_for_booleans_missing_and_invalid_values():
    split = _split()
    train = pd.DataFrame([
        _row(2024, week, f"flag-{week}", f"rb-{week}", 1.0, 1, 0.2, value)
        for week, value in enumerate([True, np.bool_(False), 1, 0, pd.NA, "invalid"], start=1)
    ], columns=OUTPUT_COLUMNS)
    report = audit_rb_rushing_yards_dataset(
        RBRushingYardsDatasetSplit(train, split.validation, split.test)
    )
    counts = _partition(report, "train").team_change_counts
    assert (counts.changed_count, counts.unchanged_count, counts.unknown_count, counts.invalid_count) == (1, 1, 1, 3)
    assert sum((counts.changed_count, counts.unchanged_count, counts.unknown_count, counts.invalid_count)) == len(train)


def test_integration_with_public_builder_and_split_uses_only_synthetic_rows():
    games = []
    for season, game, yards in [(2024, "g24", 20), (2025, "g25", 0), (2026, "g26", -2)]:
        games.append({"season": season, "week": 1, "game_id": game, "player_id": "rb", "player_name": "rb", "team": "A", "opponent": "B", "position": "RB", "rushing_attempts": 1, "rushing_yards": yards, "rushing_touchdowns": 0, "receptions": 0, "targets": 0, "receiving_yards": 0, "receiving_touchdowns": 0})
    schedule = pd.DataFrame([{"season": item["season"], "week": 1, "game_id": item["game_id"], "team": "A", "opponent": "B", "home_away": "home"} for item in games])
    built = build_rb_rushing_yards_training_dataset(pd.DataFrame(games, columns=RB_GAME_COLUMNS), schedule)
    report = audit_rb_rushing_yards_dataset(split_rb_rushing_yards_dataset(built))
    assert [item.row_count for item in report.partition_audits] == [1, 1, 1]
