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
from football.training.qb_passing_yards_linear_regression import (
    BINARY_FEATURE_COLUMNS,
    COLD_START_COLUMN,
    NUMERIC_FEATURE_COLUMNS,
    PREDICTION_COLUMN,
    QBPassingYardsLinearRegressionEvaluation,
    evaluate_qb_passing_yards_linear_regression,
    fit_qb_passing_yards_linear_regression,
    predict_qb_passing_yards_linear_regression,
    train_and_validate_qb_passing_yards_linear_regression,
)
from football.training.split_qb_passing_yards_dataset import QBPassingYardsDatasetSplit


linear_regression_module = import_module(
    "football.training.qb_passing_yards_linear_regression"
)


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
        ],
        columns=OUTPUT_COLUMNS,
    )
    validation = pd.DataFrame(
        [
            _row(2024, 1, "2024_01_F", "qb_f", 160.0, 25, cold_start=True),
            _row(2024, 2, "2024_02_G", "qb_g", 240.0, 45),
            _row(2024, 3, "2024_03_H", "qb_h", 280.0, 55),
        ],
        columns=OUTPUT_COLUMNS,
    )
    test = pd.DataFrame(
        [
            _row(2025, 1, "2025_01_I", "qb_i", 999.0, 70, cold_start=True),
            _row(2025, 2, "2025_02_J", "qb_j", -999.0, 80),
        ],
        columns=OUTPUT_COLUMNS,
    )
    return QBPassingYardsDatasetSplit(train=train, validation=validation, test=test)


def _prediction_by_key(predictions):
    return {
        tuple(row[key] for key in TARGET_KEY): row[PREDICTION_COLUMN]
        for row in predictions.to_dict(orient="records")
    }


def test_uses_exact_public_feature_contract_without_identifiers_or_target():
    model = fit_qb_passing_yards_linear_regression(_split().train)

    assert model.feature_columns == tuple(FEATURE_COLUMNS)
    assert set(model.numeric_feature_columns) | set(model.binary_feature_columns) == set(
        FEATURE_COLUMNS
    )
    assert set(model.numeric_feature_columns) & set(model.binary_feature_columns) == set()
    assert set(BINARY_FEATURE_COLUMNS) == {
        "qb_missing_history",
        "defense_missing_history",
    }
    coefficient_names = [item.transformed_feature_name for item in model.coefficients]
    assert coefficient_names == list(model.numeric_feature_columns) + list(
        model.binary_feature_columns
    )
    assert set(coefficient_names) == set(FEATURE_COLUMNS)
    assert not set(TARGET_KEY + [TARGET_COLUMN, "player_name", "team", "opponent"]).intersection(
        coefficient_names
    )


def test_training_only_median_imputation_and_frozen_model_do_not_mutate_input():
    split = _split()
    original = split.train.copy(deep=True)
    model = fit_qb_passing_yards_linear_regression(split.train)
    medians = {item.feature_name: item.training_median for item in model.numeric_imputation_summaries}

    assert medians["qb_season_passing_yards_avg"] == pytest.approx(35.0)
    assert len(medians) == len(NUMERIC_FEATURE_COLUMNS)
    assert model.training_row_count == len(split.train)
    with pytest.raises(FrozenInstanceError):
        model.intercept = 0.0
    pd.testing.assert_frame_equal(split.train, original)


def test_validation_and_test_values_cannot_change_training_fitted_imputation():
    split = _split()
    expected = fit_qb_passing_yards_linear_regression(split.train)
    changed_validation = split.validation.copy(deep=True)
    changed_validation["defense_matchup_rank"] = 999999.0
    changed_test = split.test.copy(deep=True)
    changed_test.loc[:, list(NUMERIC_FEATURE_COLUMNS)] = -999999.0

    actual = fit_qb_passing_yards_linear_regression(split.train)
    assert actual.numeric_imputation_summaries == expected.numeric_imputation_summaries
    result = train_and_validate_qb_passing_yards_linear_regression(
        QBPassingYardsDatasetSplit(split.train, changed_validation, changed_test)
    )
    assert result.model.numeric_imputation_summaries == expected.numeric_imputation_summaries


def test_prediction_imputes_permitted_history_missingness_and_preserves_flags_and_keys():
    split = _split()
    model = fit_qb_passing_yards_linear_regression(split.train)
    predictions = predict_qb_passing_yards_linear_regression(model, split.validation)

    assert len(predictions) == len(split.validation)
    assert predictions[TARGET_KEY].values.tolist() == split.validation[TARGET_KEY].values.tolist()
    assert predictions[PREDICTION_COLUMN].notna().all()
    assert np.isfinite(predictions[PREDICTION_COLUMN]).all()
    assert predictions[COLD_START_COLUMN].tolist() == [True, False, False]
    assert predictions[COLD_START_COLUMN].dtype == bool


def test_predictions_do_not_depend_on_actual_target_values_or_mutate_input():
    split = _split()
    model = fit_qb_passing_yards_linear_regression(split.train)
    changed = split.validation.copy(deep=True)
    changed[TARGET_COLUMN] = [9999.0, -9999.0, 12345.0]
    original = changed.copy(deep=True)

    expected = predict_qb_passing_yards_linear_regression(model, split.validation)
    actual = predict_qb_passing_yards_linear_regression(model, changed)

    assert _prediction_by_key(actual) == _prediction_by_key(expected)
    pd.testing.assert_frame_equal(changed, original)


def test_evaluation_reports_metrics_and_cold_start_subgroups_using_shared_evaluator(monkeypatch):
    split = _split()
    model = fit_qb_passing_yards_linear_regression(split.train)
    calls = []
    original = linear_regression_module.evaluate_qb_passing_yards_predictions

    def record_evaluate(actual, predictions):
        calls.append((actual, predictions))
        return original(actual, predictions)

    monkeypatch.setattr(
        linear_regression_module, "evaluate_qb_passing_yards_predictions", record_evaluate
    )
    evaluation = evaluate_qb_passing_yards_linear_regression(model, split.validation)

    assert isinstance(evaluation, QBPassingYardsLinearRegressionEvaluation)
    assert isinstance(evaluation.overall_metrics, RegressionMetrics)
    assert evaluation.overall_metrics.row_count == 3
    assert evaluation.cold_start_row_count == 1
    assert evaluation.cold_start_row_rate == pytest.approx(1 / 3)
    assert evaluation.non_cold_start_row_count == 2
    assert evaluation.non_cold_start_metrics is not None
    assert evaluation.cold_start_metrics is not None
    assert evaluation.cold_start_metrics.row_count == 1
    assert evaluation.overall_metrics.mae == pytest.approx(0.0653594771241804)
    assert evaluation.overall_metrics.rmse == pytest.approx(0.11320593513521623)
    assert evaluation.overall_metrics.r_squared == pytest.approx(0.9999948508815291)
    assert evaluation.cold_start_metrics.mae == pytest.approx(0.19607843137254122)
    assert evaluation.non_cold_start_metrics is not None
    assert evaluation.non_cold_start_metrics.row_count == 2
    assert evaluation.non_cold_start_metrics.mae == pytest.approx(0.0, abs=1e-12)
    assert evaluation.non_cold_start_metrics.rmse == pytest.approx(0.0, abs=1e-12)
    assert evaluation.non_cold_start_metrics.r_squared == pytest.approx(1.0)
    assert len(calls) == 3


@pytest.mark.parametrize("all_cold, expected_field", [(False, "cold_start_metrics"), (True, "non_cold_start_metrics")])
def test_empty_subgroups_are_none(all_cold, expected_field):
    split = _split()
    validation = split.validation.copy(deep=True)
    validation["qb_missing_history"] = all_cold
    if all_cold:
        validation["qb_season_history_games"] = 0
        validation["qb_last3_history_games"] = 0
        for feature in (
            "qb_season_passing_yards_avg",
            "qb_last3_passing_yards_avg",
            "qb_season_passing_attempts_avg",
            "qb_last3_passing_attempts_avg",
        ):
            validation[feature] = np.nan
    else:
        validation["qb_season_history_games"] = 2
        validation["qb_last3_history_games"] = 2
        for index, feature in enumerate(NUMERIC_FEATURE_COLUMNS[:4]):
            validation[feature] = 100.0 + index
    evaluation = evaluate_qb_passing_yards_linear_regression(
        fit_qb_passing_yards_linear_regression(split.train), validation
    )

    assert getattr(evaluation, expected_field) is None


def test_train_validation_orchestration_never_predicts_test_and_test_changes_are_inert(monkeypatch):
    split = _split()
    calls = []
    original_predict = linear_regression_module.predict_qb_passing_yards_linear_regression

    def record_predict(model, frame):
        calls.append(frame)
        return original_predict(model, frame)

    monkeypatch.setattr(
        linear_regression_module,
        "predict_qb_passing_yards_linear_regression",
        record_predict,
    )
    expected = train_and_validate_qb_passing_yards_linear_regression(split)
    assert len(calls) == 1
    assert calls[0] is not split.test
    assert set(map(tuple, calls[0][TARGET_KEY].to_numpy())) == set(
        map(tuple, split.validation[TARGET_KEY].to_numpy())
    )

    changed_test = split.test.copy(deep=True)
    changed_test[TARGET_COLUMN] = [123456.0, -123456.0]
    changed_test.loc[:, list(NUMERIC_FEATURE_COLUMNS)] = -1e12
    changed = train_and_validate_qb_passing_yards_linear_regression(
        QBPassingYardsDatasetSplit(split.train, split.validation, changed_test)
    )
    assert changed == expected


def test_baseline_comparison_uses_training_only_baseline_and_correct_improvement_directions():
    result = train_and_validate_qb_passing_yards_linear_regression(_split())
    comparison = result.historical_average_comparison
    metrics = result.validation.overall_metrics

    assert comparison.baseline_metrics.row_count == metrics.row_count == 3
    assert comparison.mae_improvement == pytest.approx(
        comparison.baseline_metrics.mae - metrics.mae
    )
    assert comparison.rmse_improvement == pytest.approx(
        comparison.baseline_metrics.rmse - metrics.rmse
    )
    assert comparison.r_squared_improvement == pytest.approx(
        metrics.r_squared - comparison.baseline_metrics.r_squared
    )
    assert comparison.beats_baseline_on_mae is (metrics.mae < comparison.baseline_metrics.mae)


def test_input_order_does_not_change_fitted_or_aggregate_results_and_predictions_align_keys():
    split = _split()
    expected = train_and_validate_qb_passing_yards_linear_regression(split)
    shuffled = QBPassingYardsDatasetSplit(
        split.train.sample(frac=1, random_state=11),
        split.validation.sample(frac=1, random_state=12),
        split.test.sample(frac=1, random_state=13),
    )
    actual = train_and_validate_qb_passing_yards_linear_regression(shuffled)
    assert actual == expected

    shuffled_predictions = predict_qb_passing_yards_linear_regression(
        expected.model, shuffled.validation
    )
    expected_predictions = predict_qb_passing_yards_linear_regression(
        expected.model, split.validation
    )
    assert _prediction_by_key(shuffled_predictions) == _prediction_by_key(expected_predictions)


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
            fit_qb_passing_yards_linear_regression(frame)
        else:
            predict_qb_passing_yards_linear_regression(
                fit_qb_passing_yards_linear_regression(split.train), frame
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
        fit_qb_passing_yards_linear_regression(change(_split().train))


def test_empty_frames_and_inconsistent_qb_history_flag_fail_clearly():
    split = _split()
    with pytest.raises(ValueError, match="must not be empty"):
        fit_qb_passing_yards_linear_regression(split.train.iloc[0:0])
    corrupted = split.validation.copy(deep=True)
    corrupted.loc[0, "qb_missing_history"] = False
    with pytest.raises(ValueError, match="inconsistent missing-history flags"):
        predict_qb_passing_yards_linear_regression(
            fit_qb_passing_yards_linear_regression(split.train), corrupted
        )


def test_public_computation_is_silent_and_reports_retain_no_dataframes(capsys):
    result = train_and_validate_qb_passing_yards_linear_regression(_split())

    assert capsys.readouterr().out == ""
    for report_type in (
        type(result), type(result.model), type(result.validation),
        type(result.historical_average_comparison),
    ):
        assert all("DataFrame" not in str(item.type) for item in fields(report_type))
