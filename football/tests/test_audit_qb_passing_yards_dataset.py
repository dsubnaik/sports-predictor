from dataclasses import fields
from importlib import import_module

import numpy as np
import pandas as pd
import pytest

from football.training.audit_qb_passing_yards_dataset import (
    QBPassingYardsDatasetAudit,
    audit_qb_passing_yards_dataset,
)
from football.training.build_qb_passing_yards_dataset import (
    FEATURE_COLUMNS,
    OUTPUT_COLUMNS,
    TARGET_COLUMN,
    TARGET_KEY,
)
from football.training.split_qb_passing_yards_dataset import (
    QBPassingYardsDatasetSplit,
)


audit_module = import_module("football.training.audit_qb_passing_yards_dataset")


def _row(season, week, game_id, player_id, target, qb_average, defense_average):
    row = {
        "season": season, "week": week, "game_id": game_id,
        "player_id": player_id, "player_name": f"{player_id} name",
        "team": "AAA", "opponent": "BBB", "home_away": "home",
        TARGET_COLUMN: target,
    }
    row.update({feature: 1.0 for feature in FEATURE_COLUMNS})
    row["qb_season_passing_yards_avg"] = qb_average
    row["qb_last3_passing_yards_avg"] = qb_average
    row["defense_season_passing_yards_allowed_avg"] = defense_average
    row["defense_last3_passing_yards_allowed_avg"] = defense_average
    row["qb_season_history_games"] = 0 if pd.isna(qb_average) else 2
    row["qb_last3_history_games"] = 0 if pd.isna(qb_average) else 2
    row["defense_season_history_games"] = 0 if pd.isna(defense_average) else 3
    row["defense_last3_history_games"] = 0 if pd.isna(defense_average) else 3
    row["qb_missing_history"] = pd.isna(qb_average)
    row["defense_missing_history"] = pd.isna(defense_average)
    return row


def _split() -> QBPassingYardsDatasetSplit:
    train = pd.DataFrame([
        _row(2023, 1, "2023_01_A", "qb_a", 100.0, np.nan, np.nan),
        _row(2023, 2, "2023_02_B", "qb_b", 200.0, 100.0, 120.0),
    ], columns=OUTPUT_COLUMNS)
    validation = pd.DataFrame([
        _row(2024, 1, "2024_01_C", "qb_c", 300.0, np.nan, 140.0),
        _row(2024, 2, "2024_02_D", "qb_d", 400.0, 200.0, np.nan),
    ], columns=OUTPUT_COLUMNS)
    test = pd.DataFrame([
        _row(2025, 1, "2025_01_E", "qb_e", 500.0, np.nan, np.nan),
        _row(2025, 2, "2025_02_F", "qb_f", 600.0, 250.0, 160.0),
    ], columns=OUTPUT_COLUMNS)
    return QBPassingYardsDatasetSplit(train=train, validation=validation, test=test)


def _summary(report, name):
    return next(summary for summary in report.partition_summaries if summary.name == name)


def test_reports_partition_counts_ranges_missingness_and_sample_sizes():
    report = audit_qb_passing_yards_dataset(_split())
    train = _summary(report, "train")

    assert isinstance(report, QBPassingYardsDatasetAudit)
    assert (train.row_count, train.unique_player_count, train.unique_game_count) == (2, 2, 2)
    assert train.minimum_period == (2023, 1)
    assert train.maximum_period == (2023, 2)
    assert train.missing_qb_season_history_count == 1
    assert train.missing_qb_season_history_rate == 0.5
    assert train.missing_defense_season_history_count == 1
    assert train.zero_qb_history_count == 1
    assert train.zero_defense_history_count == 1
    assert train.permitted_cold_start_count == 1
    sample_sizes = {item.feature_name: item for item in train.sample_size_summaries}
    assert sample_sizes["qb_season_history_games"].minimum == 0
    assert sample_sizes["qb_season_history_games"].median == 1.0
    assert sample_sizes["qb_season_history_games"].maximum == 2
    assert sample_sizes["qb_season_history_games"].zero_history_count == 1


def test_reports_training_target_features_and_validation_baseline_diagnostics():
    report = audit_qb_passing_yards_dataset(_split())

    target = report.training_target_summary
    assert target.row_count == 2
    assert target.mean == 150.0
    assert target.standard_deviation == pytest.approx(np.sqrt(5000.0))
    assert (target.minimum, target.percentile_25, target.median, target.percentile_75, target.maximum) == (
        100.0, 125.0, 150.0, 175.0, 200.0,
    )
    feature = next(item for item in report.training_feature_summaries if item.feature_name == "qb_season_passing_yards_avg")
    assert (feature.non_missing_count, feature.missing_count, feature.missing_rate) == (1, 1, 0.5)
    assert (feature.finite_numeric_count, feature.minimum, feature.median, feature.maximum) == (1, 100.0, 100.0, 100.0)

    baseline = report.validation_baseline
    assert baseline.training_fallback_value == 150.0
    assert baseline.overall_metrics.row_count == 2
    assert baseline.overall_metrics.mae == 175.0
    assert baseline.fallback_row_count == baseline.non_fallback_row_count == 1
    assert baseline.fallback_row_rate == baseline.non_fallback_row_rate == 0.5
    assert baseline.fallback_metrics is not None
    assert baseline.fallback_metrics.row_count == 1
    assert baseline.non_fallback_metrics is not None
    assert baseline.non_fallback_metrics.row_count == 1


def test_test_targets_are_never_read_or_scored_and_reports_retain_no_dataframes(monkeypatch):
    split = _split()
    expected = audit_qb_passing_yards_dataset(split)
    changed_test = split.test.copy(deep=True)
    changed_test[TARGET_COLUMN] = [999999.0, -999999.0]
    changed = audit_qb_passing_yards_dataset(
        QBPassingYardsDatasetSplit(split.train, split.validation, changed_test)
    )

    assert changed == expected
    for report_type in type(expected), *(type(item) for item in expected.partition_summaries):
        assert all("DataFrame" not in str(field.type) for field in fields(report_type))

    calls = []
    original_predict = audit_module.predict_qb_passing_yards_baseline
    def record_predict(baseline, data):
        calls.append(data)
        return original_predict(baseline, data)
    monkeypatch.setattr(audit_module, "predict_qb_passing_yards_baseline", record_predict)
    audit_qb_passing_yards_dataset(split)
    assert len(calls) == 1
    assert calls[0] is split.validation


def test_validation_targets_do_not_change_fallback_or_predictions():
    split = _split()
    report = audit_qb_passing_yards_dataset(split)
    changed_validation = split.validation.copy(deep=True)
    changed_validation[TARGET_COLUMN] = [9999.0, -9999.0]
    changed = audit_qb_passing_yards_dataset(
        QBPassingYardsDatasetSplit(split.train, changed_validation, split.test)
    )

    assert changed.validation_baseline.training_fallback_value == report.validation_baseline.training_fallback_value
    assert changed.validation_baseline.fallback_row_count == report.validation_baseline.fallback_row_count
    assert changed.validation_baseline.non_fallback_row_count == report.validation_baseline.non_fallback_row_count
    assert changed.validation_baseline.overall_metrics != report.validation_baseline.overall_metrics


def test_empty_subgroups_are_explicit_and_input_order_and_inputs_are_unchanged(capsys):
    split = _split()
    all_history_validation = split.validation.copy(deep=True)
    all_history_validation["qb_season_passing_yards_avg"] = [210.0, 220.0]
    all_history_validation["qb_last3_passing_yards_avg"] = [210.0, 220.0]
    all_history_validation["qb_season_history_games"] = [2, 2]
    all_history_validation["qb_last3_history_games"] = [2, 2]
    all_history_validation["qb_missing_history"] = [False, False]
    all_history_split = QBPassingYardsDatasetSplit(split.train, all_history_validation, split.test)
    report = audit_qb_passing_yards_dataset(all_history_split)
    assert report.validation_baseline.fallback_metrics is None
    assert report.validation_baseline.fallback_row_count == 0

    originals = tuple(data.copy(deep=True) for data in (split.train, split.validation, split.test))
    shuffled = QBPassingYardsDatasetSplit(
        split.train.sample(frac=1, random_state=1),
        split.validation.sample(frac=1, random_state=2),
        split.test.sample(frac=1, random_state=3),
    )
    assert audit_qb_passing_yards_dataset(shuffled) == audit_qb_passing_yards_dataset(split)
    for actual, original in zip((split.train, split.validation, split.test), originals, strict=True):
        pd.testing.assert_frame_equal(actual, original)
    assert capsys.readouterr().out == ""


def test_rejects_duplicate_and_cross_partition_keys_and_chronological_errors():
    split = _split()
    duplicate_train = pd.concat([split.train, split.train.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate target keys"):
        audit_qb_passing_yards_dataset(QBPassingYardsDatasetSplit(duplicate_train, split.validation, split.test))
    with pytest.raises(ValueError, match="overlapping target keys"):
        audit_qb_passing_yards_dataset(QBPassingYardsDatasetSplit(split.train, split.train.copy(), split.test))
    with pytest.raises(ValueError, match="chronologically precede validation"):
        audit_qb_passing_yards_dataset(QBPassingYardsDatasetSplit(split.validation, split.train, split.test))


def test_rejects_schema_target_sample_size_flag_and_empty_errors():
    split = _split()
    with pytest.raises(ValueError, match="missing required columns"):
        audit_qb_passing_yards_dataset(QBPassingYardsDatasetSplit(split.train.drop(columns="team"), split.validation, split.test))
    with pytest.raises(ValueError, match="inconsistent schemas"):
        audit_qb_passing_yards_dataset(QBPassingYardsDatasetSplit(split.train.assign(extra=1), split.validation, split.test))
    invalid_size = split.train.copy()
    invalid_size.loc[0, "qb_season_history_games"] = -1
    with pytest.raises(ValueError, match="invalid historical sample sizes"):
        audit_qb_passing_yards_dataset(QBPassingYardsDatasetSplit(invalid_size, split.validation, split.test))
    invalid_flag = split.train.copy()
    invalid_flag.loc[0, "qb_missing_history"] = False
    with pytest.raises(ValueError, match="inconsistent missing-history flags"):
        audit_qb_passing_yards_dataset(QBPassingYardsDatasetSplit(invalid_flag, split.validation, split.test))
    with pytest.raises(ValueError, match="must not be empty"):
        audit_qb_passing_yards_dataset(QBPassingYardsDatasetSplit(split.train.iloc[0:0], split.validation, split.test))


@pytest.mark.parametrize("partition_name", ["train", "validation"])
def test_rejects_nonfinite_train_or_validation_targets(partition_name):
    split = _split()
    corrupted = getattr(split, partition_name).copy()
    corrupted[TARGET_COLUMN] = np.nan
    kwargs = {"train": split.train, "validation": split.validation, "test": split.test}
    kwargs[partition_name] = corrupted
    with pytest.raises(ValueError, match="non-finite target_passing_yards"):
        audit_qb_passing_yards_dataset(QBPassingYardsDatasetSplit(**kwargs))
