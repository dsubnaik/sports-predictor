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
from football.training.qb_passing_yards_baseline import RegressionMetrics
from football.training.qb_passing_yards_random_forest import (
    BINARY_FEATURE_COLUMNS,
    COLD_START_COLUMN,
    NUMERIC_FEATURE_COLUMNS,
    PREDICTION_COLUMN,
    QBPassingYardsRandomForestEvaluation,
    RandomForestParameters,
    evaluate_qb_passing_yards_random_forest,
    fit_qb_passing_yards_random_forest,
    predict_qb_passing_yards_random_forest,
    train_and_validate_qb_passing_yards_random_forest,
)
from football.training.split_qb_passing_yards_dataset import QBPassingYardsDatasetSplit


random_forest_module = import_module("football.training.qb_passing_yards_random_forest")


def _row(season, week, game_id, player_id, target, base, *, cold_start=False):
    row = {
        "season": season,
        "week": week,
        "game_id": game_id,
        "player_id": player_id,
        "player_name": f"{player_id} name",
        "team": "AAA",
        "opponent": "BBB",
        "home_away": "home",
        TARGET_COLUMN: target,
    }
    row.update({feature: float(base + index) for index, feature in enumerate(FEATURE_COLUMNS)})
    row["qb_season_history_games"] = 0 if cold_start else 2
    row["qb_last3_history_games"] = 0 if cold_start else 2
    row["qb_missing_history"] = cold_start
    row["defense_season_history_games"] = 3
    row["defense_last3_history_games"] = 3
    row["defense_missing_history"] = False
    if cold_start:
        for feature in (
            "qb_season_passing_yards_avg",
            "qb_last3_passing_yards_avg",
            "qb_season_passing_attempts_avg",
            "qb_last3_passing_attempts_avg",
        ):
            row[feature] = np.nan
    return row


def _split() -> QBPassingYardsDatasetSplit:
    train = pd.DataFrame(
        [
            _row(2023, 1, "2023_01_A", "qb_a", 100.0, 10),
            _row(2023, 2, "2023_02_B", "qb_b", 140.0, 20, cold_start=True),
            _row(2023, 3, "2023_03_C", "qb_c", 180.0, 30),
            _row(2023, 4, "2023_04_D", "qb_d", 220.0, 40),
            _row(2023, 5, "2023_05_E", "qb_e", 260.0, 50),
            _row(2023, 6, "2023_06_F", "qb_f", 300.0, 60),
        ],
        columns=OUTPUT_COLUMNS,
    )
    validation = pd.DataFrame(
        [
            _row(2024, 1, "2024_01_G", "qb_g", 160.0, 25, cold_start=True),
            _row(2024, 2, "2024_02_H", "qb_h", 240.0, 45),
            _row(2024, 3, "2024_03_I", "qb_i", 280.0, 55),
        ],
        columns=OUTPUT_COLUMNS,
    )
    test = pd.DataFrame(
        [
            _row(2025, 1, "2025_01_J", "qb_j", 999.0, 70, cold_start=True),
            _row(2025, 2, "2025_02_K", "qb_k", -999.0, 80),
        ],
        columns=OUTPUT_COLUMNS,
    )
    return QBPassingYardsDatasetSplit(train=train, validation=validation, test=test)


def _prediction_by_key(predictions):
    return {
        tuple(row[column] for column in TARGET_KEY): row[PREDICTION_COLUMN]
        for row in predictions.to_dict(orient="records")
    }


def test_uses_public_contract_and_fixed_predetermined_parameters():
    model = fit_qb_passing_yards_random_forest(_split().train)

    assert model.feature_columns == tuple(FEATURE_COLUMNS)
    assert set(model.numeric_feature_columns) | set(model.binary_feature_columns) == set(
        FEATURE_COLUMNS
    )
    assert set(model.numeric_feature_columns) & set(model.binary_feature_columns) == set()
    assert model.binary_feature_columns == BINARY_FEATURE_COLUMNS
    assert model.parameters == RandomForestParameters(
        n_estimators=500,
        random_state=42,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=1.0,
        bootstrap=True,
        n_jobs=1,
    )
    names = [item.feature_name for item in model.feature_importances]
    assert set(names) == set(FEATURE_COLUMNS)
    assert not set(TARGET_KEY + [TARGET_COLUMN, "player_name", "team", "opponent", "home_away"]).intersection(names)


def test_training_only_imputation_flags_and_frozen_wrapper_do_not_mutate_input():
    split = _split()
    original = split.train.copy(deep=True)
    model = fit_qb_passing_yards_random_forest(split.train)
    medians = {item.feature_name: item.training_median for item in model.numeric_imputation_summaries}

    assert medians["qb_season_passing_yards_avg"] == pytest.approx(40.0)
    assert len(medians) == len(NUMERIC_FEATURE_COLUMNS)
    assert model.training_row_count == len(split.train)
    with pytest.raises(FrozenInstanceError):
        model.training_row_count = 0
    pd.testing.assert_frame_equal(split.train, original)


def test_validation_and_test_values_cannot_affect_fitted_training_metadata():
    split = _split()
    expected = fit_qb_passing_yards_random_forest(split.train)
    changed_validation = split.validation.copy(deep=True)
    changed_validation["defense_matchup_rank"] = 999999.0
    changed_test = split.test.copy(deep=True)
    changed_test.loc[:, list(NUMERIC_FEATURE_COLUMNS)] = -999999.0

    result = train_and_validate_qb_passing_yards_random_forest(
        QBPassingYardsDatasetSplit(split.train, changed_validation, changed_test)
    )
    assert result.model.numeric_imputation_summaries == expected.numeric_imputation_summaries
    assert result.model.feature_importances == expected.feature_importances


def test_prediction_has_one_finite_aligned_value_and_does_not_read_targets_or_mutate():
    split = _split()
    model = fit_qb_passing_yards_random_forest(split.train)
    changed = split.validation.copy(deep=True)
    changed[TARGET_COLUMN] = [9999.0, -9999.0, 12345.0]
    original = changed.copy(deep=True)

    expected = predict_qb_passing_yards_random_forest(model, split.validation)
    actual = predict_qb_passing_yards_random_forest(model, changed)

    assert len(expected) == len(split.validation)
    assert expected[TARGET_KEY].values.tolist() == split.validation[TARGET_KEY].values.tolist()
    assert np.isfinite(expected[PREDICTION_COLUMN]).all()
    assert expected[COLD_START_COLUMN].tolist() == [True, False, False]
    assert _prediction_by_key(actual) == _prediction_by_key(expected)
    pd.testing.assert_frame_equal(changed, original)


def test_evaluation_uses_shared_metrics_and_reports_cold_start_groups(monkeypatch):
    split = _split()
    model = fit_qb_passing_yards_random_forest(split.train)
    calls = []
    original = random_forest_module.evaluate_qb_passing_yards_predictions

    def record_evaluate(actual, predictions):
        calls.append((actual, predictions))
        return original(actual, predictions)

    monkeypatch.setattr(random_forest_module, "evaluate_qb_passing_yards_predictions", record_evaluate)
    evaluation = evaluate_qb_passing_yards_random_forest(model, split.validation)

    assert isinstance(evaluation, QBPassingYardsRandomForestEvaluation)
    assert isinstance(evaluation.overall_metrics, RegressionMetrics)
    assert evaluation.overall_metrics.row_count == 3
    assert evaluation.cold_start_row_count == 1
    assert evaluation.cold_start_row_rate == pytest.approx(1 / 3)
    assert evaluation.non_cold_start_row_count == 2
    assert evaluation.non_cold_start_row_rate == pytest.approx(2 / 3)
    assert evaluation.cold_start_metrics is not None
    assert evaluation.non_cold_start_metrics is not None
    assert evaluation.cold_start_metrics.row_count == 1
    assert evaluation.non_cold_start_metrics.row_count == 2
    assert evaluation.overall_metrics.mae == pytest.approx(14.08)
    assert evaluation.overall_metrics.rmse == pytest.approx(15.663256366413714)
    assert evaluation.overall_metrics.r_squared == pytest.approx(0.9014268571428572)
    assert evaluation.cold_start_metrics.mae == pytest.approx(5.84)
    assert evaluation.non_cold_start_metrics.mae == pytest.approx(18.2)
    assert len(calls) == 3


@pytest.mark.parametrize("all_cold, empty_field", [(False, "cold_start_metrics"), (True, "non_cold_start_metrics")])
def test_empty_subgroups_are_none(all_cold, empty_field):
    split = _split()
    validation = split.validation.copy(deep=True)
    validation["qb_missing_history"] = all_cold
    if all_cold:
        validation[["qb_season_history_games", "qb_last3_history_games"]] = 0
        for feature in NUMERIC_FEATURE_COLUMNS[:4]:
            validation[feature] = np.nan
    else:
        validation[["qb_season_history_games", "qb_last3_history_games"]] = 2
        for index, feature in enumerate(NUMERIC_FEATURE_COLUMNS[:4]):
            validation[feature] = 100.0 + index
    evaluation = evaluate_qb_passing_yards_random_forest(
        fit_qb_passing_yards_random_forest(split.train), validation
    )

    assert getattr(evaluation, empty_field) is None


def test_train_validation_only_reuses_baseline_and_linear_regression_and_never_predicts_test(monkeypatch):
    split = _split()
    rf_prediction_calls = []
    baseline_calls = []
    linear_fit_calls = []
    original_rf_predict = random_forest_module.predict_qb_passing_yards_random_forest
    original_baseline_fit = random_forest_module.fit_qb_passing_yards_baseline
    original_linear_fit = random_forest_module.fit_qb_passing_yards_linear_regression

    def record_rf_predict(model, frame):
        rf_prediction_calls.append(frame)
        return original_rf_predict(model, frame)

    def record_baseline_fit(frame):
        baseline_calls.append(frame)
        return original_baseline_fit(frame)

    def record_linear_fit(frame):
        linear_fit_calls.append(frame)
        return original_linear_fit(frame)

    monkeypatch.setattr(random_forest_module, "predict_qb_passing_yards_random_forest", record_rf_predict)
    monkeypatch.setattr(random_forest_module, "fit_qb_passing_yards_baseline", record_baseline_fit)
    monkeypatch.setattr(random_forest_module, "fit_qb_passing_yards_linear_regression", record_linear_fit)
    expected = train_and_validate_qb_passing_yards_random_forest(split)

    assert len(rf_prediction_calls) == 1
    assert rf_prediction_calls[0] is not split.test
    assert baseline_calls == [split.train]
    assert linear_fit_calls == [split.train]
    assert expected.historical_average_comparison.comparison_model_name == "Historical average"
    assert expected.linear_regression_comparison.comparison_model_name == "Linear Regression"

    changed_test = split.test.copy(deep=True)
    changed_test[TARGET_COLUMN] = [123456.0, -123456.0]
    changed_test.loc[:, list(NUMERIC_FEATURE_COLUMNS)] = -1e12
    changed = train_and_validate_qb_passing_yards_random_forest(
        QBPassingYardsDatasetSplit(split.train, split.validation, changed_test)
    )
    assert changed == expected


def test_comparison_directions_and_importances_are_deterministic_and_valid():
    split = _split()
    expected = train_and_validate_qb_passing_yards_random_forest(split)
    actual = train_and_validate_qb_passing_yards_random_forest(split)
    metrics = expected.validation.overall_metrics

    assert actual == expected
    for comparison in (
        expected.historical_average_comparison,
        expected.linear_regression_comparison,
    ):
        assert comparison.mae_improvement == pytest.approx(comparison.comparison_metrics.mae - metrics.mae)
        assert comparison.rmse_improvement == pytest.approx(comparison.comparison_metrics.rmse - metrics.rmse)
        assert comparison.r_squared_improvement == pytest.approx(metrics.r_squared - comparison.comparison_metrics.r_squared)
        assert comparison.random_forest_wins_on_mae is (metrics.mae < comparison.comparison_metrics.mae)

    importances = expected.model.feature_importances
    assert all(np.isfinite(item.importance) and item.importance >= 0 for item in importances)
    assert sum(item.importance for item in importances) == pytest.approx(1.0)
    assert list(importances) == sorted(importances, key=lambda item: (-item.importance, item.feature_name))


def test_input_order_does_not_change_fits_aggregates_or_key_to_prediction_mapping():
    split = _split()
    expected = train_and_validate_qb_passing_yards_random_forest(split)
    shuffled = QBPassingYardsDatasetSplit(
        split.train.sample(frac=1, random_state=11),
        split.validation.sample(frac=1, random_state=12),
        split.test.sample(frac=1, random_state=13),
    )
    actual = train_and_validate_qb_passing_yards_random_forest(shuffled)
    assert actual == expected
    assert _prediction_by_key(
        predict_qb_passing_yards_random_forest(expected.model, shuffled.validation)
    ) == _prediction_by_key(
        predict_qb_passing_yards_random_forest(expected.model, split.validation)
    )


@pytest.mark.parametrize(
    "operation, column",
    [
        ("fit", FEATURE_COLUMNS[0]),
        ("fit", TARGET_COLUMN),
        ("predict", FEATURE_COLUMNS[1]),
        ("predict", "game_id"),
    ],
)
def test_missing_required_columns_fail_clearly(operation, column):
    split = _split()
    frame = split.train.drop(columns=column)
    with pytest.raises(ValueError, match="missing required columns"):
        if operation == "fit":
            fit_qb_passing_yards_random_forest(frame)
        else:
            predict_qb_passing_yards_random_forest(
                fit_qb_passing_yards_random_forest(split.train), frame
            )


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda data: pd.concat([data, data.iloc[[0]]], ignore_index=True), "duplicate target keys"),
        (lambda data: data.assign(target_passing_yards=np.inf), "non-finite target_passing_yards"),
        (lambda data: data.assign(qb_last3_passing_yards_avg=np.inf), "non-finite predictive"),
        (lambda data: data.assign(qb_missing_history=1), "invalid missing-history flags"),
    ],
)
def test_structural_corruption_fails_clearly(change, message):
    with pytest.raises(ValueError, match=message):
        fit_qb_passing_yards_random_forest(change(_split().train))


def test_empty_frames_and_inconsistent_flags_fail_clearly():
    split = _split()
    with pytest.raises(ValueError, match="must not be empty"):
        fit_qb_passing_yards_random_forest(split.train.iloc[0:0])
    invalid = split.validation.copy(deep=True)
    invalid.loc[0, "qb_missing_history"] = False
    with pytest.raises(ValueError, match="inconsistent missing-history flags"):
        predict_qb_passing_yards_random_forest(
            fit_qb_passing_yards_random_forest(split.train), invalid
        )


def test_public_computation_is_silent_and_reports_retain_no_dataframes(capsys):
    result = train_and_validate_qb_passing_yards_random_forest(_split())

    assert capsys.readouterr().out == ""
    for report_type in (
        type(result), type(result.model), type(result.validation),
        type(result.historical_average_comparison),
    ):
        assert all("DataFrame" not in str(item.type) for item in fields(report_type))
