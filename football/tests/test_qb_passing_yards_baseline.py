from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from football.training.build_qb_passing_yards_dataset import (
    FEATURE_COLUMNS,
    OUTPUT_COLUMNS,
    TARGET_COLUMN,
    TARGET_KEY,
)
from football.training.qb_passing_yards_baseline import (
    FALLBACK_USED_COLUMN,
    HISTORICAL_AVERAGE_COLUMN,
    PREDICTION_COLUMN,
    QBPassingYardsBaseline,
    RegressionMetrics,
    evaluate_qb_passing_yards_predictions,
    fit_qb_passing_yards_baseline,
    predict_qb_passing_yards_baseline,
)


def _dataset() -> pd.DataFrame:
    rows = []
    for season, week, game_id, player_id, target, history in [
        (2024, 1, "2024_01_A", "qb_a", 100.0, 90.0),
        (2024, 2, "2024_02_B", "qb_b", 200.0, np.nan),
        (2025, 1, "2025_01_C", "qb_c", 300.0, 0.0),
        (2025, 2, "2025_02_D", "qb_d", 400.0, np.nan),
    ]:
        row = {
            "season": season, "week": week, "game_id": game_id,
            "player_id": player_id, "player_name": f"{player_id} name",
            "team": "AAA", "opponent": "BBB", "home_away": "home",
            TARGET_COLUMN: target,
        }
        row.update({column: 1.0 for column in FEATURE_COLUMNS})
        row[HISTORICAL_AVERAGE_COLUMN] = history
        rows.append(row)
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def _fitted() -> QBPassingYardsBaseline:
    return fit_qb_passing_yards_baseline(_dataset().iloc[:2])


def test_fit_uses_exact_training_target_mean_is_immutable_and_does_not_mutate():
    train = _dataset().iloc[:2].copy()
    original = train.copy(deep=True)

    fitted = fit_qb_passing_yards_baseline(train)

    assert fitted.training_target_mean == 150.0
    with pytest.raises(FrozenInstanceError):
        fitted.training_target_mean = 1.0
    pd.testing.assert_frame_equal(train, original)


def test_validation_and_test_targets_cannot_influence_training_fallback():
    train = _dataset().iloc[:2]
    validation_and_test = _dataset().iloc[2:].copy()
    validation_and_test[TARGET_COLUMN] = [9999.0, -9999.0]

    assert fit_qb_passing_yards_baseline(train).training_target_mean == 150.0
    assert fit_qb_passing_yards_baseline(train).training_target_mean != validation_and_test[TARGET_COLUMN].mean()


def test_prediction_uses_history_or_training_fallback_with_auditable_flags():
    result = predict_qb_passing_yards_baseline(_fitted(), _dataset())

    assert result.columns.tolist() == [
        *TARGET_KEY, TARGET_COLUMN, PREDICTION_COLUMN, FALLBACK_USED_COLUMN
    ]
    assert result[PREDICTION_COLUMN].tolist() == [90.0, 150.0, 0.0, 150.0]
    assert result[FALLBACK_USED_COLUMN].tolist() == [False, True, False, True]
    assert result[FALLBACK_USED_COLUMN].sum() == 2
    assert (~result[FALLBACK_USED_COLUMN]).sum() == 2


def test_predictions_are_key_sorted_independent_and_target_independent():
    data = _dataset()
    original = data.copy(deep=True)
    fitted = _fitted()

    expected = predict_qb_passing_yards_baseline(fitted, data)
    shuffled = predict_qb_passing_yards_baseline(fitted, data.sample(frac=1, random_state=3))
    changed_targets = data.copy(deep=True)
    changed_targets[TARGET_COLUMN] = [1.0, 2.0, 3.0, 4.0]
    changed = predict_qb_passing_yards_baseline(fitted, changed_targets)

    pd.testing.assert_frame_equal(shuffled, expected)
    pd.testing.assert_frame_equal(
        changed.drop(columns=TARGET_COLUMN), expected.drop(columns=TARGET_COLUMN)
    )
    expected.loc[0, PREDICTION_COLUMN] = -1.0
    assert data.loc[0, HISTORICAL_AVERAGE_COLUMN] == 90.0
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("column", [TARGET_COLUMN, "game_id"])
def test_fit_rejects_missing_required_columns(column):
    with pytest.raises(ValueError, match="missing required columns"):
        fit_qb_passing_yards_baseline(_dataset().drop(columns=column))


@pytest.mark.parametrize("column", [HISTORICAL_AVERAGE_COLUMN, "player_id"])
def test_prediction_rejects_missing_required_columns(column):
    with pytest.raises(ValueError, match="missing required columns"):
        predict_qb_passing_yards_baseline(_fitted(), _dataset().drop(columns=column))


@pytest.mark.parametrize("value", [None, np.nan, np.inf])
def test_fit_rejects_missing_or_nonfinite_training_targets(value):
    train = _dataset().iloc[:2].copy()
    train[TARGET_COLUMN] = train[TARGET_COLUMN].astype(object)
    train.loc[0, TARGET_COLUMN] = value

    with pytest.raises(ValueError, match="target_passing_yards"):
        fit_qb_passing_yards_baseline(train)


def test_fit_rejects_a_training_partition_without_usable_targets():
    with pytest.raises(ValueError, match="must not be empty"):
        fit_qb_passing_yards_baseline(_dataset().iloc[0:0])


@pytest.mark.parametrize("value", [np.inf, -np.inf, "not-a-number"])
def test_prediction_rejects_unexpected_nonfinite_history(value):
    data = _dataset().copy()
    data[HISTORICAL_AVERAGE_COLUMN] = data[HISTORICAL_AVERAGE_COLUMN].astype(object)
    data.loc[0, HISTORICAL_AVERAGE_COLUMN] = value

    with pytest.raises(ValueError, match=HISTORICAL_AVERAGE_COLUMN):
        predict_qb_passing_yards_baseline(_fitted(), data)


@pytest.mark.parametrize("for_prediction", [False, True])
def test_duplicate_target_keys_fail_clearly(for_prediction):
    data = pd.concat([_dataset(), _dataset().iloc[[0]]], ignore_index=True)
    function = predict_qb_passing_yards_baseline if for_prediction else fit_qb_passing_yards_baseline
    with pytest.raises(ValueError, match="duplicate target keys"):
        if for_prediction:
            function(_fitted(), data)
        else:
            function(data)


def test_evaluation_metrics_match_hand_calculation_and_do_not_mutate_inputs():
    actual = _dataset().iloc[:2].copy()
    predictions = actual.loc[:, TARGET_KEY].copy()
    predictions[PREDICTION_COLUMN] = [90.0, 220.0]
    actual_before = actual.copy(deep=True)
    predictions_before = predictions.copy(deep=True)

    metrics = evaluate_qb_passing_yards_predictions(actual, predictions)

    assert isinstance(metrics, RegressionMetrics)
    assert metrics.row_count == 2
    assert metrics.mae == pytest.approx(15.0)
    assert metrics.rmse == pytest.approx(np.sqrt(250.0))
    assert metrics.r_squared == pytest.approx(0.9)
    pd.testing.assert_frame_equal(actual, actual_before)
    pd.testing.assert_frame_equal(predictions, predictions_before)


@pytest.mark.parametrize(
    ("predictions", "expected_r_squared"),
    [([100.0, 100.0], 1.0), ([99.0, 101.0], 0.0)],
)
def test_constant_target_r_squared_uses_sklearn_finite_convention(predictions, expected_r_squared):
    actual = _dataset().iloc[:2].copy()
    actual[TARGET_COLUMN] = 100.0
    prediction_data = actual.loc[:, TARGET_KEY].copy()
    prediction_data[PREDICTION_COLUMN] = predictions

    assert evaluate_qb_passing_yards_predictions(
        actual, prediction_data
    ).r_squared == expected_r_squared


def test_evaluation_rejects_empty_invalid_or_misaligned_data():
    actual = _dataset().iloc[:2].copy()
    predictions = actual.loc[:, TARGET_KEY].copy()
    predictions[PREDICTION_COLUMN] = [100.0, 200.0]

    with pytest.raises(ValueError, match="must not be empty"):
        evaluate_qb_passing_yards_predictions(actual.iloc[0:0], predictions.iloc[0:0])
    with pytest.raises(ValueError, match="must be finite"):
        evaluate_qb_passing_yards_predictions(
            actual.assign(target_passing_yards=np.nan), predictions
        )
    with pytest.raises(ValueError, match="must be finite"):
        evaluate_qb_passing_yards_predictions(
            actual, predictions.assign(baseline_prediction=np.inf)
        )
    with pytest.raises(ValueError, match="must match exactly"):
        evaluate_qb_passing_yards_predictions(actual, predictions.iloc[[0]])
