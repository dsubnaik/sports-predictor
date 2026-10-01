from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from football.tests.test_qb_passing_yards_model_comparison import _split
from football.training.build_qb_passing_yards_dataset import FEATURE_COLUMNS
from football.training.qb_passing_yards_defensive_representation import (
    CONTINUOUS_DEFENSE,
    CONTINUOUS_PLUS_RANK,
    CONTINUOUS_PLUS_TIER,
    LINEAR_REFERENCE_TIER,
    QB_ONLY_REFERENCE,
    RANK_ONLY,
    REPRESENTATION_NAMES,
    TIER_COLUMNS,
    TIER_ONLY,
    _columns,
    _with_tiers,
    analyze_qb_passing_yards_defensive_representation,
)
from football.training.qb_passing_yards_gradient_boosting import evaluate_qb_passing_yards_gradient_boosting, fit_qb_passing_yards_gradient_boosting
from football.training.qb_passing_yards_linear_regression import evaluate_qb_passing_yards_linear_regression, fit_qb_passing_yards_linear_regression
from football.training.qb_passing_yards_random_forest import evaluate_qb_passing_yards_random_forest, fit_qb_passing_yards_random_forest
from football.training.qb_passing_yards_xgboost import evaluate_qb_passing_yards_xgboost, fit_qb_passing_yards_xgboost
from football.training.split_qb_passing_yards_dataset import QBPassingYardsDatasetSplit


def _result(split=None):
    return analyze_qb_passing_yards_defensive_representation(split or _split(), bootstrap_replicates=17)


def _variants(result):
    return {(item.estimator_name, variant.representation_name): variant for item in result.estimator_results for variant in item.variants}


def test_tier_boundaries_missing_behavior_and_linear_reference_are_exact():
    frame = pd.DataFrame({"defense_matchup_rank": [1, 10, 11, 22, 23, 32, np.nan]})
    result = _with_tiers(frame)
    assert result.loc[:, TIER_COLUMNS].sum(axis=1).tolist() == [1] * 7
    assert result[TIER_COLUMNS[0]].tolist() == [True, True, False, False, False, False, False]
    assert result[TIER_COLUMNS[1]].tolist() == [False, False, True, True, False, False, False]
    assert result[TIER_COLUMNS[2]].tolist() == [False, False, False, False, True, True, False]
    assert result[TIER_COLUMNS[3]].tolist() == [False, False, False, False, False, False, True]
    assert LINEAR_REFERENCE_TIER not in _columns(TIER_ONLY, "Linear Regression")
    assert LINEAR_REFERENCE_TIER in _columns(TIER_ONLY, "Random Forest")


@pytest.mark.parametrize("rank", [0, 33, -1, 1.5, np.inf])
def test_invalid_rank_rejected(rank):
    split = _split()
    train = split.train.astype({"defense_matchup_rank": object}).copy(deep=True)
    train.loc[0, "defense_matchup_rank"] = rank
    with pytest.raises(ValueError, match="Defense matchup rank"):
        _result(QBPassingYardsDatasetSplit(train, split.validation, split.test))


def test_all_variants_are_deterministic_train_validation_only_and_immutable():
    class ExplodingTest:
        def __getattribute__(self, name):
            raise AssertionError(f"test partition accessed: {name}")

    split = _split()
    originals = [part.copy(deep=True) for part in (split.train, split.validation, split.test)]
    result = _result(split)
    assert [item.representation_name for item in result.estimator_results[0].variants] == list(REPRESENTATION_NAMES)
    assert all(item.validation.row_count == len(split.validation) for estimator in result.estimator_results for item in estimator.variants)
    assert all(item.cold_start.row_count == 1 for estimator in result.estimator_results for item in estimator.variants)
    assert result.rank_direction == "rank 1 is the highest point-in-time mean passing yards allowed"
    with pytest.raises(FrozenInstanceError):
        result.validation_row_count = 0
    for actual, expected in zip((split.train, split.validation, split.test), originals, strict=True):
        pd.testing.assert_frame_equal(actual, expected)
    assert _result(QBPassingYardsDatasetSplit(split.train, split.validation, ExplodingTest())) == result
    shuffled = QBPassingYardsDatasetSplit(split.train.sample(frac=1, random_state=1), split.validation.sample(frac=1, random_state=2), split.test)
    assert _result(shuffled) == result


def test_continuous_plus_rank_reproduces_completed_models_and_comparison_signs():
    split = _split()
    result = _result(split)
    expected = {
        "Linear Regression": evaluate_qb_passing_yards_linear_regression(fit_qb_passing_yards_linear_regression(split.train), split.validation).overall_metrics,
        "Random Forest": evaluate_qb_passing_yards_random_forest(fit_qb_passing_yards_random_forest(split.train), split.validation).overall_metrics,
        "sklearn Gradient Boosting": evaluate_qb_passing_yards_gradient_boosting(fit_qb_passing_yards_gradient_boosting(split.train), split.validation).overall_metrics,
        "XGBoost": evaluate_qb_passing_yards_xgboost(fit_qb_passing_yards_xgboost(split.train), split.validation).overall_metrics,
    }
    variants = _variants(result)
    for estimator, metrics in expected.items():
        observed = variants[(estimator, CONTINUOUS_PLUS_RANK)].validation.metrics
        assert observed is not None
        assert observed.mae == pytest.approx(metrics.mae, abs=1e-10)
        assert observed.rmse == pytest.approx(metrics.rmse, abs=1e-10)
        assert observed.r_squared == pytest.approx(metrics.r_squared, abs=1e-10)
    for estimator in result.estimator_results:
        base = variants[(estimator.estimator_name, CONTINUOUS_DEFENSE)].validation.metrics
        rank = variants[(estimator.estimator_name, CONTINUOUS_PLUS_RANK)].validation.metrics
        tier = variants[(estimator.estimator_name, CONTINUOUS_PLUS_TIER)].validation.metrics
        assert base and rank and tier
        assert estimator.rank_beyond_continuous.mae_improvement == pytest.approx(base.mae - rank.mae)
        assert estimator.tiers_beyond_continuous.mae_improvement == pytest.approx(base.mae - tier.mae)
        assert estimator.rank_versus_tiers.mae_improvement == pytest.approx(tier.mae - rank.mae)
        assert estimator.rank_beyond_continuous_bootstrap.cluster_count == 8
        assert estimator.rank_beyond_continuous_bootstrap.seed == 42


@pytest.mark.parametrize("change, message", [
    (lambda data: data.drop(columns="game_id"), "missing required columns"),
    (lambda data: pd.concat([data, data.iloc[[0]]]), "unique, nonmissing target keys"),
    (lambda data: data.assign(target_passing_yards=np.inf), "target_passing_yards values must be finite"),
])
def test_structural_failures_are_clear(change, message):
    split = _split()
    with pytest.raises(ValueError, match=message):
        _result(QBPassingYardsDatasetSplit(change(split.train), split.validation, split.test))
