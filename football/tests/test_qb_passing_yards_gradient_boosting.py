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
from football.training.qb_passing_yards_gradient_boosting import (
    BINARY_FEATURE_COLUMNS,
    COLD_START_COLUMN,
    NUMERIC_FEATURE_COLUMNS,
    PREDICTION_COLUMN,
    GradientBoostingParameters,
    QBPassingYardsGradientBoostingEvaluation,
    evaluate_qb_passing_yards_gradient_boosting,
    fit_qb_passing_yards_gradient_boosting,
    predict_qb_passing_yards_gradient_boosting,
    train_and_validate_qb_passing_yards_gradient_boosting,
)
from football.training.split_qb_passing_yards_dataset import QBPassingYardsDatasetSplit


gradient_boosting_module = import_module(
    "football.training.qb_passing_yards_gradient_boosting"
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
        for feature in NUMERIC_FEATURE_COLUMNS[:4]:
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
        ], columns=OUTPUT_COLUMNS,
    )
    validation = pd.DataFrame(
        [
            _row(2024, 1, "2024_01_G", "qb_g", 160.0, 25, cold_start=True),
            _row(2024, 2, "2024_02_H", "qb_h", 240.0, 45),
            _row(2024, 3, "2024_03_I", "qb_i", 280.0, 55),
        ], columns=OUTPUT_COLUMNS,
    )
    test = pd.DataFrame(
        [
            _row(2025, 1, "2025_01_J", "qb_j", 999.0, 70, cold_start=True),
            _row(2025, 2, "2025_02_K", "qb_k", -999.0, 80),
        ], columns=OUTPUT_COLUMNS,
    )
    return QBPassingYardsDatasetSplit(train=train, validation=validation, test=test)


def _prediction_by_key(predictions):
    return {
        tuple(row[column] for column in TARGET_KEY): row[PREDICTION_COLUMN]
        for row in predictions.to_dict(orient="records")
    }


def test_uses_exact_public_contract_and_fixed_parameters():
    model = fit_qb_passing_yards_gradient_boosting(_split().train)

    assert model.feature_columns == tuple(FEATURE_COLUMNS)
    assert set(model.numeric_feature_columns) | set(model.binary_feature_columns) == set(FEATURE_COLUMNS)
    assert not set(model.numeric_feature_columns) & set(model.binary_feature_columns)
    assert model.binary_feature_columns == BINARY_FEATURE_COLUMNS
    assert model.parameters == GradientBoostingParameters(
        loss="squared_error", learning_rate=0.05, n_estimators=200, subsample=1.0,
        criterion="friedman_mse", min_samples_split=2, min_samples_leaf=1,
        max_depth=3, max_features=None, random_state=42,
    )
    names = [item.feature_name for item in model.feature_importances]
    assert set(names) == set(FEATURE_COLUMNS)
    assert not set(TARGET_KEY + [TARGET_COLUMN, "player_name", "team", "opponent", "home_away"]).intersection(names)


def test_training_only_imputation_flags_frozen_model_and_input_ownership():
    split = _split()
    original = split.train.copy(deep=True)
    model = fit_qb_passing_yards_gradient_boosting(split.train)
    medians = {item.feature_name: item.training_median for item in model.numeric_imputation_summaries}

    assert medians["qb_season_passing_yards_avg"] == pytest.approx(40.0)
    assert len(medians) == len(NUMERIC_FEATURE_COLUMNS)
    assert model.training_row_count == len(split.train)
    with pytest.raises(FrozenInstanceError):
        model.training_row_count = 0
    pd.testing.assert_frame_equal(split.train, original)


def test_prediction_is_aligned_target_independent_finite_and_non_mutating():
    split = _split()
    model = fit_qb_passing_yards_gradient_boosting(split.train)
    changed = split.validation.copy(deep=True)
    changed[TARGET_COLUMN] = [9999.0, -9999.0, 12345.0]
    original = changed.copy(deep=True)
    expected = predict_qb_passing_yards_gradient_boosting(model, split.validation)
    actual = predict_qb_passing_yards_gradient_boosting(model, changed)

    assert len(expected) == len(split.validation)
    assert expected[TARGET_KEY].values.tolist() == split.validation[TARGET_KEY].values.tolist()
    assert np.isfinite(expected[PREDICTION_COLUMN]).all()
    assert expected[COLD_START_COLUMN].tolist() == [True, False, False]
    assert _prediction_by_key(actual) == _prediction_by_key(expected)
    pd.testing.assert_frame_equal(changed, original)


def test_evaluation_reuses_shared_metrics_and_reports_subgroups(monkeypatch):
    split = _split()
    model = fit_qb_passing_yards_gradient_boosting(split.train)
    calls = []
    original = gradient_boosting_module.evaluate_qb_passing_yards_predictions

    def record_evaluate(actual, predictions):
        calls.append((actual, predictions))
        return original(actual, predictions)

    monkeypatch.setattr(gradient_boosting_module, "evaluate_qb_passing_yards_predictions", record_evaluate)
    evaluation = evaluate_qb_passing_yards_gradient_boosting(model, split.validation)

    assert isinstance(evaluation, QBPassingYardsGradientBoostingEvaluation)
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
    assert len(calls) == 3


@pytest.mark.parametrize("all_cold, empty_field", [(False, "cold_start_metrics"), (True, "non_cold_start_metrics")])
def test_empty_subgroups_are_explicitly_none(all_cold, empty_field):
    split = _split()
    validation = split.validation.copy(deep=True)
    validation["qb_missing_history"] = all_cold
    if all_cold:
        validation[["qb_season_history_games", "qb_last3_history_games"]] = 0
        validation.loc[:, list(NUMERIC_FEATURE_COLUMNS[:4])] = np.nan
    else:
        validation[["qb_season_history_games", "qb_last3_history_games"]] = 2
        for index, feature in enumerate(NUMERIC_FEATURE_COLUMNS[:4]):
            validation[feature] = 100.0 + index
    evaluation = evaluate_qb_passing_yards_gradient_boosting(
        fit_qb_passing_yards_gradient_boosting(split.train), validation
    )
    assert getattr(evaluation, empty_field) is None


def test_orchestration_uses_train_validation_only_and_reuses_three_comparisons(monkeypatch):
    split = _split()
    prediction_calls, baseline_fit_calls, linear_fit_calls, forest_fit_calls = [], [], [], []
    original_predict = gradient_boosting_module.predict_qb_passing_yards_gradient_boosting
    original_baseline_fit = gradient_boosting_module.fit_qb_passing_yards_baseline
    original_linear_fit = gradient_boosting_module.fit_qb_passing_yards_linear_regression
    original_forest_fit = gradient_boosting_module.fit_qb_passing_yards_random_forest

    def record_predict(model, frame):
        prediction_calls.append(frame)
        return original_predict(model, frame)

    monkeypatch.setattr(gradient_boosting_module, "predict_qb_passing_yards_gradient_boosting", record_predict)
    monkeypatch.setattr(gradient_boosting_module, "fit_qb_passing_yards_baseline", lambda frame: (baseline_fit_calls.append(frame), original_baseline_fit(frame))[1])
    monkeypatch.setattr(gradient_boosting_module, "fit_qb_passing_yards_linear_regression", lambda frame: (linear_fit_calls.append(frame), original_linear_fit(frame))[1])
    monkeypatch.setattr(gradient_boosting_module, "fit_qb_passing_yards_random_forest", lambda frame: (forest_fit_calls.append(frame), original_forest_fit(frame))[1])
    expected = train_and_validate_qb_passing_yards_gradient_boosting(split)

    assert len(prediction_calls) == 1
    assert prediction_calls[0] is not split.test
    assert prediction_calls[0][TARGET_KEY].values.tolist() == split.validation.sort_values(
        TARGET_KEY, kind="mergesort"
    )[TARGET_KEY].values.tolist()
    assert baseline_fit_calls == [split.train]
    assert linear_fit_calls == [split.train]
    assert forest_fit_calls == [split.train]
    assert [item.comparison_model_name for item in (expected.historical_average_comparison, expected.linear_regression_comparison, expected.random_forest_comparison)] == ["Historical average", "Linear Regression", "Random Forest"]

    changed_test = split.test.copy(deep=True)
    changed_test[TARGET_COLUMN] = [123456.0, -123456.0]
    changed_test = changed_test.astype({feature: object for feature in FEATURE_COLUMNS})
    changed_test.loc[:, list(FEATURE_COLUMNS)] = np.inf
    actual = train_and_validate_qb_passing_yards_gradient_boosting(
        QBPassingYardsDatasetSplit(split.train, split.validation, changed_test)
    )
    assert actual == expected


def test_comparisons_importances_and_repeat_fit_are_deterministic():
    split = _split()
    expected = train_and_validate_qb_passing_yards_gradient_boosting(split)
    actual = train_and_validate_qb_passing_yards_gradient_boosting(split)
    assert actual == expected
    metrics = expected.validation.overall_metrics
    for comparison in (expected.historical_average_comparison, expected.linear_regression_comparison, expected.random_forest_comparison):
        assert comparison.mae_improvement == pytest.approx(comparison.comparison_metrics.mae - metrics.mae)
        assert comparison.rmse_improvement == pytest.approx(comparison.comparison_metrics.rmse - metrics.rmse)
        assert comparison.r_squared_improvement == pytest.approx(metrics.r_squared - comparison.comparison_metrics.r_squared)
        assert comparison.gradient_boosting_wins_on_mae is (metrics.mae < comparison.comparison_metrics.mae)
    importances = expected.model.feature_importances
    assert all(np.isfinite(item.importance) and item.importance >= 0 for item in importances)
    assert sum(item.importance for item in importances) == pytest.approx(1.0)
    assert list(importances) == sorted(importances, key=lambda item: (-item.importance, item.feature_name))


def test_input_order_does_not_change_results_or_key_alignment():
    split = _split()
    expected = train_and_validate_qb_passing_yards_gradient_boosting(split)
    shuffled = QBPassingYardsDatasetSplit(split.train.sample(frac=1, random_state=11), split.validation.sample(frac=1, random_state=12), split.test.sample(frac=1, random_state=13))
    actual = train_and_validate_qb_passing_yards_gradient_boosting(shuffled)
    assert actual == expected
    assert _prediction_by_key(predict_qb_passing_yards_gradient_boosting(expected.model, shuffled.validation)) == _prediction_by_key(predict_qb_passing_yards_gradient_boosting(expected.model, split.validation))


@pytest.mark.parametrize("change, message", [
    (lambda data: data.drop(columns=FEATURE_COLUMNS[0]), "missing required columns"),
    (lambda data: pd.concat([data, data.iloc[[0]]], ignore_index=True), "duplicate target keys"),
    (lambda data: data.assign(target_passing_yards=np.inf), "non-finite target_passing_yards"),
    (lambda data: data.assign(qb_last3_passing_yards_avg=np.inf), "non-finite predictive"),
    (lambda data: data.assign(qb_missing_history=1), "invalid missing-history flags"),
    (lambda data: data.assign(qb_season_history_games=-1), "invalid history sample sizes"),
    (lambda data: data.assign(defense_last3_history_games=1.5), "invalid history sample sizes"),
    (lambda data: data.assign(qb_last3_history_games=np.nan), "invalid history sample sizes"),
])
def test_structural_failures_are_clear(change, message):
    with pytest.raises(ValueError, match=message):
        fit_qb_passing_yards_gradient_boosting(change(_split().train))


def test_empty_and_inconsistent_flags_fail_and_public_computation_is_silent(capsys):
    split = _split()
    with pytest.raises(ValueError, match="must not be empty"):
        fit_qb_passing_yards_gradient_boosting(split.train.iloc[0:0])
    invalid = split.validation.copy(deep=True)
    invalid.loc[0, "qb_missing_history"] = False
    with pytest.raises(ValueError, match="inconsistent missing-history flags"):
        predict_qb_passing_yards_gradient_boosting(
            fit_qb_passing_yards_gradient_boosting(split.train), invalid
        )
    result = train_and_validate_qb_passing_yards_gradient_boosting(split)
    assert capsys.readouterr().out == ""
    for report_type in (type(result), type(result.model), type(result.validation), type(result.historical_average_comparison)):
        assert all("DataFrame" not in str(item.type) for item in fields(report_type))
