from dataclasses import FrozenInstanceError, fields
from importlib import import_module

import numpy as np
import pandas as pd
import pytest

from football.training.build_qb_passing_yards_dataset import (
    FEATURE_COLUMNS,
    OUTPUT_COLUMNS,
    TARGET_COLUMN,
    TARGET_KEY,
)
from football.training.qb_passing_yards_model_comparison import (
    EXACT_PREDICTION_TOLERANCE,
    MODEL_NAMES,
    ErrorDistributionDiagnostics,
    QBPassingYardsModelComparison,
    _error_distribution,
    _matchup_rank_buckets,
    _row_level_wins,
    compare_qb_passing_yards_models,
)
from football.training.split_qb_passing_yards_dataset import QBPassingYardsDatasetSplit


comparison_module = import_module("football.training.qb_passing_yards_model_comparison")


def _row(season, week, game_id, player_id, target, base, *, cold=False, defense_cold=False, rank=17):
    row = {
        "season": season, "week": week, "game_id": game_id, "player_id": player_id,
        "player_name": f"{player_id} name", "team": "AAA", "opponent": "BBB",
        "home_away": "home", TARGET_COLUMN: target,
    }
    row.update({feature: float(base + index) for index, feature in enumerate(FEATURE_COLUMNS)})
    row["qb_season_history_games"] = 0 if cold else 2
    row["qb_last3_history_games"] = 0 if cold else 2
    row["qb_missing_history"] = cold
    row["defense_season_history_games"] = 0 if defense_cold else 3
    row["defense_last3_history_games"] = 0 if defense_cold else 3
    row["defense_missing_history"] = defense_cold
    row["defense_matchup_rank"] = np.nan if rank is None else rank
    if cold:
        for feature in FEATURE_COLUMNS[:4]:
            row[feature] = np.nan
    if defense_cold:
        for feature in (
            "defense_season_passing_yards_allowed_avg",
            "defense_last3_passing_yards_allowed_avg",
            "defense_season_passing_attempts_allowed_avg",
        ):
            row[feature] = np.nan
    return row


def _split() -> QBPassingYardsDatasetSplit:
    train = pd.DataFrame([
        _row(2023, 1, "2023_01_A", "qb_a", 100.0, 10),
        _row(2023, 2, "2023_02_B", "qb_b", 140.0, 20, cold=True),
        _row(2023, 3, "2023_03_C", "qb_c", 180.0, 30),
        _row(2023, 4, "2023_04_D", "qb_d", 220.0, 40),
        _row(2023, 5, "2023_05_E", "qb_e", 260.0, 50),
        _row(2023, 6, "2023_06_F", "qb_f", 300.0, 60),
    ], columns=OUTPUT_COLUMNS)
    validation = pd.DataFrame([
        _row(2024, 1, "2024_01_A", "qb_0", 155.0, 16, cold=True, defense_cold=True, rank=None),
        _row(2024, 2, "2024_02_B", "qb_1", 175.0, 23, rank=1),
        _row(2024, 3, "2024_03_C", "qb_2", 195.0, 30, rank=10),
        _row(2024, 4, "2024_04_D", "qb_3", 215.0, 37, rank=11),
        _row(2024, 5, "2024_05_E", "qb_4", 235.0, 44, rank=22),
        _row(2024, 6, "2024_06_F", "qb_5", 255.0, 51, rank=23),
        _row(2024, 7, "2024_07_G", "qb_6", 275.0, 58, rank=32),
        _row(2024, 8, "2024_08_H", "qb_7", 295.0, 65, rank=17),
    ], columns=OUTPUT_COLUMNS)
    qb_season = [0, 1, 3, 4, 8, 9, 12, 2]
    qb_last3 = [0, 1, 2, 3, 3, 3, 3, 2]
    defense_season = [0, 1, 3, 4, 8, 9, 12, 2]
    defense_last3 = [0, 1, 2, 3, 3, 3, 3, 2]
    for index in range(len(validation)):
        validation.loc[index, "qb_season_history_games"] = qb_season[index]
        validation.loc[index, "qb_last3_history_games"] = qb_last3[index]
        validation.loc[index, "qb_missing_history"] = qb_season[index] == 0
        validation.loc[index, "defense_season_history_games"] = defense_season[index]
        validation.loc[index, "defense_last3_history_games"] = defense_last3[index]
        validation.loc[index, "defense_missing_history"] = defense_season[index] == 0
        if qb_season[index] == 0:
            validation.loc[index, FEATURE_COLUMNS[:4]] = np.nan
        if defense_season[index] == 0:
            validation.loc[index, [
                "defense_season_passing_yards_allowed_avg",
                "defense_last3_passing_yards_allowed_avg",
                "defense_season_passing_attempts_allowed_avg",
            ]] = np.nan
    test = pd.DataFrame([
        _row(2025, 1, "2025_01_A", "qb_test_a", 999.0, 70, cold=True),
        _row(2025, 2, "2025_02_B", "qb_test_b", -999.0, 80),
    ], columns=OUTPUT_COLUMNS)
    return QBPassingYardsDatasetSplit(train=train, validation=validation, test=test)


def _slice_map(result):
    return {(item.family_name, item.bucket_name): item for item in result.slices}


def test_all_models_share_validation_keys_and_report_required_metrics_without_mutation():
    split = _split()
    originals = [frame.copy(deep=True) for frame in (split.train, split.validation, split.test)]
    result = compare_qb_passing_yards_models(split)

    assert isinstance(result, QBPassingYardsModelComparison)
    assert result.validation_row_count == len(split.validation)
    assert [item.model_name for item in result.model_diagnostics] == list(MODEL_NAMES)
    assert all(item.metrics.row_count == len(split.validation) for item in result.model_diagnostics)
    assert all(np.isfinite(item.mean_signed_error) for item in result.model_diagnostics)
    assert all(item.cold_start_metrics.row_count == 1 for item in result.model_diagnostics)
    assert all(item.non_cold_start_metrics.row_count == 7 for item in result.model_diagnostics)
    assert all(isinstance(item.error_distribution, ErrorDistributionDiagnostics) for item in result.model_diagnostics)
    for actual, original in zip((split.train, split.validation, split.test), originals, strict=True):
        pd.testing.assert_frame_equal(actual, original)
    with pytest.raises(FrozenInstanceError):
        result.validation_row_count = 0


def test_all_required_slice_boundaries_and_empty_slice_representation():
    result = compare_qb_passing_yards_models(_split())
    slices = _slice_map(result)

    assert slices[("cold_start_status", "cold_start")].row_count == 1
    assert slices[("cold_start_status", "non_cold_start")].row_count == 7
    assert [slices[("qb_season_history_depth", name)].row_count for name in ("0_games", "1_to_3_games", "4_to_8_games", "9_plus_games")] == [1, 3, 2, 2]
    assert [slices[("qb_recent_history_depth", name)].row_count for name in ("0_games", "1_game", "2_games", "3_games")] == [1, 1, 2, 4]
    assert [slices[("defense_season_history_depth", name)].row_count for name in ("0_games", "1_to_3_games", "4_to_8_games", "9_plus_games")] == [1, 3, 2, 2]
    assert [slices[("defense_matchup_rank_tier", name)].row_count for name in ("1_to_10", "11_to_22", "23_to_32", "missing")] == [2, 3, 2, 1]

    empty = comparison_module._slice_model_metrics(
        "Historical average", _split().validation, np.ones(8), np.zeros(8, dtype=bool)
    )
    assert empty.row_count == 0
    assert empty.metrics is None and empty.mean_signed_error is None


def test_signed_error_distribution_percentiles_and_tie_wins_are_explicit():
    diagnostics = _error_distribution(np.array([-2.0, 0.0, 3.0, EXACT_PREDICTION_TOLERANCE / 2]))
    assert diagnostics.median_absolute_error == pytest.approx(1.0)
    assert diagnostics.percentile_75_absolute_error == pytest.approx(2.25)
    assert diagnostics.percentile_90_absolute_error == pytest.approx(2.7)
    assert diagnostics.maximum_absolute_error == pytest.approx(3.0)
    assert diagnostics.underprediction_count == 1
    assert diagnostics.overprediction_count == 1
    assert diagnostics.approximately_exact_prediction_count == 2
    wins = _row_level_wins(
        {"one": np.array([10.0, 22.0]), "two": np.array([10.0 + 5e-13, 18.0])},
        np.array([10.0, 20.0]),
    )
    assert [(item.model_name, item.win_count) for item in wins] == [("one", 2), ("two", 2)]


def test_pairwise_signs_disagreement_and_row_wins_are_deterministic():
    result = compare_qb_passing_yards_models(_split())
    diagnostics = {item.model_name: item for item in result.model_diagnostics}
    for item in result.pairwise_mae_differences:
        assert item.first_minus_second_mae == pytest.approx(
            diagnostics[item.first_model_name].metrics.mae - diagnostics[item.second_model_name].metrics.mae
        )
    assert len(result.prediction_disagreements) == 10
    assert all(item.mean_absolute_prediction_difference >= 0 for item in result.prediction_disagreements)
    assert all(item.win_count >= 0 for item in result.row_level_win_counts)


def test_test_partition_is_unreachable_and_all_prediction_calls_use_validation(monkeypatch):
    split = _split()
    prediction_calls = []
    fitting_calls = []
    for attribute in (
        "fit_qb_passing_yards_baseline", "fit_qb_passing_yards_linear_regression",
        "fit_qb_passing_yards_random_forest", "fit_qb_passing_yards_gradient_boosting",
        "fit_qb_passing_yards_xgboost",
    ):
        original = getattr(comparison_module, attribute)
        monkeypatch.setattr(
            comparison_module, attribute,
            lambda frame, original=original: (fitting_calls.append(frame), original(frame))[1],
        )
    originals = {}
    for attribute in (
        "predict_qb_passing_yards_baseline", "predict_qb_passing_yards_linear_regression",
        "predict_qb_passing_yards_random_forest", "predict_qb_passing_yards_gradient_boosting",
        "predict_qb_passing_yards_xgboost",
    ):
        original = getattr(comparison_module, attribute)
        originals[attribute] = original
        monkeypatch.setattr(
            comparison_module, attribute,
            lambda model, frame, original=original: (
                prediction_calls.append(frame), original(model, frame)
            )[1],
        )
    expected = compare_qb_passing_yards_models(split)
    assert fitting_calls == [split.train] * 5
    assert len(prediction_calls) == 5
    assert all(frame is not split.test for frame in prediction_calls)
    assert all(frame[TARGET_KEY].values.tolist() == split.validation.sort_values(TARGET_KEY, kind="mergesort")[TARGET_KEY].values.tolist() for frame in prediction_calls)

    changed_test = split.test.astype({feature: object for feature in FEATURE_COLUMNS}).copy()
    changed_test[TARGET_COLUMN] = [123456.0, -123456.0]
    changed_test.loc[:, list(FEATURE_COLUMNS)] = np.inf
    assert compare_qb_passing_yards_models(QBPassingYardsDatasetSplit(split.train, split.validation, changed_test)) == expected


def test_shuffled_rows_do_not_change_aggregates_and_reports_retain_no_dataframes():
    split = _split()
    expected = compare_qb_passing_yards_models(split)
    shuffled = QBPassingYardsDatasetSplit(
        split.train.sample(frac=1, random_state=1),
        split.validation.sample(frac=1, random_state=2),
        split.test.sample(frac=1, random_state=3),
    )
    assert compare_qb_passing_yards_models(shuffled) == expected
    for report_type in (type(expected), type(expected.model_diagnostics[0]), type(expected.slices[0])):
        assert all("DataFrame" not in str(field.type) for field in fields(report_type))


@pytest.mark.parametrize("change, message", [
    (lambda data: data.drop(columns="game_id"), "missing required columns"),
    (lambda data: pd.concat([data, data.iloc[[0]]], ignore_index=True), "duplicate target keys"),
    (lambda data: data.assign(target_passing_yards=np.inf), "target_passing_yards values must be finite"),
])
def test_invalid_validation_structure_fails_before_model_prediction(change, message):
    split = _split()
    with pytest.raises(ValueError, match=message):
        compare_qb_passing_yards_models(QBPassingYardsDatasetSplit(split.train, change(split.validation), split.test))


def test_invalid_rank_and_nonfinite_model_prediction_fail_clearly(monkeypatch):
    split = _split()
    invalid = split.validation.copy(deep=True)
    invalid.loc[1, "defense_matchup_rank"] = 33
    with pytest.raises(ValueError, match="Defense matchup rank"):
        compare_qb_passing_yards_models(QBPassingYardsDatasetSplit(split.train, invalid, split.test))

    original = comparison_module.predict_qb_passing_yards_baseline
    def nonfinite(model, frame):
        output = original(model, frame)
        output.loc[0, "baseline_prediction"] = np.inf
        return output
    monkeypatch.setattr(comparison_module, "predict_qb_passing_yards_baseline", nonfinite)
    with pytest.raises(ValueError, match="Historical average prediction values must be finite"):
        compare_qb_passing_yards_models(split)


def test_prediction_key_mismatch_fails_clearly(monkeypatch):
    split = _split()
    original = comparison_module.predict_qb_passing_yards_baseline

    def missing_key(model, frame):
        return original(model, frame).iloc[1:].reset_index(drop=True)

    monkeypatch.setattr(comparison_module, "predict_qb_passing_yards_baseline", missing_key)
    with pytest.raises(ValueError, match="prediction keys do not match validation keys"):
        compare_qb_passing_yards_models(split)


def test_public_computation_is_silent(capsys):
    result = compare_qb_passing_yards_models(_split())
    assert result.validation_row_count == 8
    assert capsys.readouterr().out == ""


def test_matchup_rank_missing_and_boundaries_are_exhaustive():
    buckets = _matchup_rank_buckets(pd.Series([1, 10, 11, 22, 23, 32, np.nan]))
    assert [(name, int(mask.sum())) for name, mask in buckets] == [
        ("1_to_10", 2), ("11_to_22", 2), ("23_to_32", 2), ("missing", 1)
    ]
