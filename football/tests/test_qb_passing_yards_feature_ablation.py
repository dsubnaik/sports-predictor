from dataclasses import FrozenInstanceError, fields

import numpy as np
import pandas as pd
import pytest

from football.tests.test_qb_passing_yards_model_comparison import _split
from football.training.build_qb_passing_yards_dataset import FEATURE_COLUMNS, TARGET_COLUMN
from football.training.qb_passing_yards_feature_ablation import (
    DEFENSE_FEATURE_COLUMNS,
    DEFENSE_ONLY,
    FEATURE_GROUP_COLUMNS,
    FEATURE_GROUP_NAMES,
    QB_AND_DEFENSE,
    QB_HISTORY_FEATURE_COLUMNS,
    QB_HISTORY_ONLY,
    QBPassingYardsFeatureAblation,
    _estimator,
    _fit_predict,
    _paired_bootstrap,
    analyze_qb_passing_yards_feature_ablation,
)
from football.training.qb_passing_yards_gradient_boosting import (
    evaluate_qb_passing_yards_gradient_boosting,
    fit_qb_passing_yards_gradient_boosting,
)
from football.training.qb_passing_yards_linear_regression import (
    evaluate_qb_passing_yards_linear_regression,
    fit_qb_passing_yards_linear_regression,
    predict_qb_passing_yards_linear_regression,
)
from football.training.qb_passing_yards_random_forest import (
    evaluate_qb_passing_yards_random_forest,
    fit_qb_passing_yards_random_forest,
    predict_qb_passing_yards_random_forest,
)
from football.training.qb_passing_yards_xgboost import (
    evaluate_qb_passing_yards_xgboost,
    fit_qb_passing_yards_xgboost,
    predict_qb_passing_yards_xgboost,
)
from football.training.qb_passing_yards_gradient_boosting import (
    predict_qb_passing_yards_gradient_boosting,
)
from football.training.split_qb_passing_yards_dataset import QBPassingYardsDatasetSplit


def _result(split=None):
    return analyze_qb_passing_yards_feature_ablation(split or _split(), bootstrap_replicates=19)


def _variants(result):
    return {
        (item.estimator_name, variant.feature_group_name): variant
        for item in result.estimator_results
        for variant in item.variants
    }


def test_fixed_feature_groups_are_exact_public_contract_and_exclude_metadata():
    assert FEATURE_GROUP_NAMES == (QB_HISTORY_ONLY, DEFENSE_ONLY, QB_AND_DEFENSE)
    assert FEATURE_GROUP_COLUMNS[QB_HISTORY_ONLY] == QB_HISTORY_FEATURE_COLUMNS
    assert FEATURE_GROUP_COLUMNS[DEFENSE_ONLY] == DEFENSE_FEATURE_COLUMNS
    assert FEATURE_GROUP_COLUMNS[QB_AND_DEFENSE] == tuple(FEATURE_COLUMNS)
    assert tuple(FEATURE_COLUMNS) == (*QB_HISTORY_FEATURE_COLUMNS, *DEFENSE_FEATURE_COLUMNS)
    forbidden = {"game_id", "player_id", "player_name", "team", "opponent", "home_away", TARGET_COLUMN}
    assert not forbidden.intersection((*QB_HISTORY_FEATURE_COLUMNS, *DEFENSE_FEATURE_COLUMNS))


def test_all_variants_are_train_only_validation_only_immutable_and_nonmutating():
    split = _split()
    originals = [part.copy(deep=True) for part in (split.train, split.validation, split.test)]
    result = _result(split)
    assert isinstance(result, QBPassingYardsFeatureAblation)
    assert result.validation_row_count == len(split.validation)
    assert [item.estimator_name for item in result.estimator_results] == [
        "Linear Regression", "Random Forest", "sklearn Gradient Boosting", "XGBoost"
    ]
    for estimator in result.estimator_results:
        assert [item.feature_group_name for item in estimator.variants] == list(FEATURE_GROUP_NAMES)
        assert all(item.validation.row_count == len(split.validation) for item in estimator.variants)
        assert all(item.cold_start.row_count == 1 for item in estimator.variants)
        assert all(item.non_cold_start.row_count == 7 for item in estimator.variants)
    for actual, expected in zip((split.train, split.validation, split.test), originals, strict=True):
        pd.testing.assert_frame_equal(actual, expected)
    with pytest.raises(FrozenInstanceError):
        result.validation_row_count = 0
    assert all("DataFrame" not in str(field.type) for field in fields(type(result)))


def test_full_feature_variants_reproduce_completed_model_validation_metrics():
    split = _split()
    result = _result(split)
    expected = {
        "Linear Regression": evaluate_qb_passing_yards_linear_regression(
            fit_qb_passing_yards_linear_regression(split.train), split.validation
        ).overall_metrics,
        "Random Forest": evaluate_qb_passing_yards_random_forest(
            fit_qb_passing_yards_random_forest(split.train), split.validation
        ).overall_metrics,
        "sklearn Gradient Boosting": evaluate_qb_passing_yards_gradient_boosting(
            fit_qb_passing_yards_gradient_boosting(split.train), split.validation
        ).overall_metrics,
        "XGBoost": evaluate_qb_passing_yards_xgboost(
            fit_qb_passing_yards_xgboost(split.train), split.validation
        ).overall_metrics,
    }
    variants = _variants(result)
    for estimator_name, metrics in expected.items():
        observed = variants[(estimator_name, QB_AND_DEFENSE)].validation.metrics
        assert observed is not None
        assert observed.row_count == metrics.row_count
        assert observed.mae == pytest.approx(metrics.mae, abs=1e-10)
        assert observed.rmse == pytest.approx(metrics.rmse, abs=1e-10)
        assert observed.r_squared == pytest.approx(metrics.r_squared, abs=1e-10)


def test_full_feature_variant_predictions_and_fixed_parameters_match_completed_models():
    split = _split()
    canonical_train = split.train.sort_values(["season", "week", "game_id", "player_id"], kind="mergesort")
    canonical_validation = split.validation.sort_values(["season", "week", "game_id", "player_id"], kind="mergesort")
    completed = {
        "Linear Regression": predict_qb_passing_yards_linear_regression(
            fit_qb_passing_yards_linear_regression(split.train), canonical_validation
        ),
        "Random Forest": predict_qb_passing_yards_random_forest(
            fit_qb_passing_yards_random_forest(split.train), canonical_validation
        ),
        "sklearn Gradient Boosting": predict_qb_passing_yards_gradient_boosting(
            fit_qb_passing_yards_gradient_boosting(split.train), canonical_validation
        ),
        "XGBoost": predict_qb_passing_yards_xgboost(
            fit_qb_passing_yards_xgboost(split.train), canonical_validation
        ),
    }
    for name, output in completed.items():
        prediction_column = next(column for column in output if column.endswith("_prediction"))
        actual = _fit_predict(name, tuple(FEATURE_COLUMNS), canonical_train, canonical_validation)
        assert actual == pytest.approx(output[prediction_column].to_numpy(dtype=float), abs=1e-10)

    assert _estimator("Random Forest").get_params()["n_estimators"] == 500
    assert _estimator("Random Forest").get_params()["random_state"] == 42
    assert _estimator("sklearn Gradient Boosting").get_params()["n_estimators"] == 200
    assert _estimator("sklearn Gradient Boosting").get_params()["learning_rate"] == pytest.approx(0.05)
    assert _estimator("XGBoost").get_params()["n_estimators"] == 300
    assert _estimator("XGBoost").get_params()["tree_method"] == "hist"
    assert _estimator("XGBoost").get_params()["random_state"] == 42


def test_comparison_signs_cold_start_membership_and_bootstrap_are_explicit():
    result = _result()
    variants = _variants(result)
    for estimator in result.estimator_results:
        history = variants[(estimator.estimator_name, QB_HISTORY_ONLY)].validation.metrics
        combined = variants[(estimator.estimator_name, QB_AND_DEFENSE)].validation.metrics
        assert history is not None and combined is not None
        comparison = estimator.qb_history_to_combined
        assert comparison.mae_improvement_for_second_group == pytest.approx(history.mae - combined.mae)
        assert comparison.rmse_improvement_for_second_group == pytest.approx(history.rmse - combined.rmse)
        assert comparison.r_squared_improvement_for_second_group == pytest.approx(combined.r_squared - history.r_squared)
        assert estimator.adding_defense_bootstrap.point_estimate == pytest.approx(comparison.mae_improvement_for_second_group)
        assert estimator.adding_defense_bootstrap.replicate_count == 19
        assert estimator.adding_defense_bootstrap.seed == 42
        assert estimator.adding_defense_bootstrap.cluster_count == 8

    helps = _paired_bootstrap("synthetic", np.array(["a", "a", "b", "b"]), np.zeros(4), np.ones(4), np.zeros(4), 31)
    hurts = _paired_bootstrap("synthetic", np.array(["a", "a", "b", "b"]), np.zeros(4), np.zeros(4), np.ones(4), 31)
    assert helps.point_estimate == pytest.approx(1.0)
    assert helps.lower_bound == pytest.approx(1.0)
    assert helps.upper_bound == pytest.approx(1.0)
    assert hurts.point_estimate == pytest.approx(-1.0)


def test_test_partition_is_unreachable_and_shuffling_is_deterministic():
    class ExplodingTest:
        def __getattribute__(self, name):
            raise AssertionError(f"test partition accessed via {name}")

    split = _split()
    expected = _result(split)
    sentinel = QBPassingYardsDatasetSplit(split.train, split.validation, ExplodingTest())
    assert _result(sentinel) == expected
    invalid_test = split.test.astype(object).copy(deep=True)
    invalid_test.loc[:, :] = np.inf
    assert _result(QBPassingYardsDatasetSplit(split.train, split.validation, invalid_test)) == expected
    shuffled = QBPassingYardsDatasetSplit(
        split.train.sample(frac=1, random_state=1),
        split.validation.sample(frac=1, random_state=2),
        split.test,
    )
    assert _result(shuffled) == expected


@pytest.mark.parametrize("change, message", [
    (lambda data: data.drop(columns="qb_season_passing_yards_avg"), "missing required columns"),
    (lambda data: pd.concat([data, data.iloc[[0]]], ignore_index=True), "duplicate target keys"),
    (lambda data: data.assign(target_passing_yards=np.inf), "target_passing_yards values must be finite"),
    (lambda data: data.assign(qb_missing_history=1), "invalid missing-history flags"),
    (lambda data: data.assign(qb_season_history_games=1.5), "invalid history sample sizes"),
    (lambda data: data.assign(defense_last3_history_games=np.nan), "invalid history sample sizes"),
])
def test_invalid_selected_contract_data_fails_clearly(change, message):
    split = _split()
    with pytest.raises(ValueError, match=message):
        _result(QBPassingYardsDatasetSplit(change(split.train), split.validation, split.test))


def test_empty_cold_start_population_is_explicit_none_metrics():
    split = _split()
    train = split.train.copy(deep=True)
    validation = split.validation.loc[~split.validation["qb_missing_history"]].copy(deep=True)
    result = _result(QBPassingYardsDatasetSplit(train, validation, split.test))
    for estimator in result.estimator_results:
        for variant in estimator.variants:
            assert variant.cold_start.row_count == 0
            assert variant.cold_start.metrics is None
            assert variant.cold_start.mean_signed_error is None
