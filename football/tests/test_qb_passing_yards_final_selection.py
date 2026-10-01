from dataclasses import FrozenInstanceError
import pytest
from football.tests.test_qb_passing_yards_model_comparison import _split
from football.training.build_qb_passing_yards_dataset import FEATURE_COLUMNS
from football.training.qb_passing_yards_final_selection import select_qb_passing_yards_final_model
from football.training.qb_passing_yards_gradient_boosting import (
    fit_qb_passing_yards_gradient_boosting,
    evaluate_qb_passing_yards_gradient_boosting,
)
from football.training.split_qb_passing_yards_dataset import QBPassingYardsDatasetSplit

def test_final_selection_is_fixed_and_test_blind():
    split = _split()
    class Exploding:
        def __getattribute__(self, name): raise AssertionError("test accessed")
    result = select_qb_passing_yards_final_model(QBPassingYardsDatasetSplit(split.train, split.validation, Exploding()))
    assert result.report.selected_model_name == "sklearn GradientBoostingRegressor"
    assert result.report.feature_columns == tuple(FEATURE_COLUMNS)
    assert result.model.parameters.n_estimators == 200
    assert result.model.parameters.random_state == 42
    assert "opponent-adjusted form" in result.report.excluded_experiments
    with pytest.raises(FrozenInstanceError): result.report.selected_model_name = "x"


def test_exact_parameters_feature_contract_and_completed_metric_parity():
    split = _split()
    result = select_qb_passing_yards_final_model(split)
    parameters = result.report.parameters
    assert (parameters.loss, parameters.learning_rate, parameters.n_estimators,
            parameters.subsample, parameters.criterion, parameters.min_samples_split,
            parameters.min_samples_leaf, parameters.max_depth, parameters.max_features,
            parameters.random_state) == ("squared_error", 0.05, 200, 1.0, "friedman_mse", 2, 1, 3, None, 42)
    assert result.report.feature_columns == tuple(FEATURE_COLUMNS)
    forbidden = {"game_id", "player_id", "player_name", "team", "opponent", "home_away", "target_passing_yards", "sportsbook", "depth_chart", "tier", "style", "opponent_adjusted"}
    assert not any(any(term in feature for term in forbidden) for feature in result.report.feature_columns)
    expected = evaluate_qb_passing_yards_gradient_boosting(
        fit_qb_passing_yards_gradient_boosting(split.train), split.validation
    )
    assert result.report.validation_metrics.mae == pytest.approx(expected.overall_metrics.mae, abs=1e-10)
    assert result.report.validation_metrics.rmse == pytest.approx(expected.overall_metrics.rmse, abs=1e-10)
    assert result.report.validation_metrics.r_squared == pytest.approx(expected.overall_metrics.r_squared, abs=1e-10)
    for actual, expected_metrics in (
        (result.report.validation_evaluation.cold_start_metrics, expected.cold_start_metrics),
        (result.report.validation_evaluation.non_cold_start_metrics, expected.non_cold_start_metrics),
    ):
        assert actual is not None
        assert expected_metrics is not None
        assert actual.row_count == expected_metrics.row_count
        assert actual.mae == pytest.approx(expected_metrics.mae, abs=1e-10)
        assert actual.rmse == pytest.approx(expected_metrics.rmse, abs=1e-10)
        assert actual.r_squared == pytest.approx(expected_metrics.r_squared, abs=1e-10)


def test_rationale_preprocessing_and_safety_contract_are_explicit(monkeypatch):
    import football.training.qb_passing_yards_final_selection as module
    split = _split()
    calls, original = [], module.evaluate_qb_passing_yards_gradient_boosting
    def record(model, frame):
        calls.append(tuple(frame[["season", "week", "game_id", "player_id"]].itertuples(index=False, name=None)))
        return original(model, frame)
    monkeypatch.setattr(module, "evaluate_qb_passing_yards_gradient_boosting", record)
    original_frames = [part.copy(deep=True) for part in (split.train, split.validation, split.test)]
    result = module.select_qb_passing_yards_final_model(split)
    expected_keys = tuple(split.validation.sort_values(["season", "week", "game_id", "player_id"], kind="mergesort")[["season", "week", "game_id", "player_id"]].itertuples(index=False, name=None))
    assert calls == [expected_keys]
    assert result.report.numeric_preprocessing == "training-only median imputation"
    assert result.report.binary_preprocessing == "validated boolean pass-through"
    assert result.report.scaling == "no scaling"
    assert "2025" in result.report.validation_period and "2026" in result.report.holdout_status
    assert {"opponent-adjusted form", "fixed rank tiers", "sportsbook lines", "unverified style features"}.issubset(result.report.excluded_experiments)
    assert any("likely-primary selection is separate" in item for item in result.report.limitations)
    for actual, expected in zip((split.train, split.validation, split.test), original_frames, strict=True):
        __import__("pandas").testing.assert_frame_equal(actual, expected)


def test_shuffling_and_invalid_test_replacement_do_not_change_selection():
    split = _split()
    expected = select_qb_passing_yards_final_model(split).report
    invalid = split.test.astype(object).copy(deep=True)
    invalid.loc[:, :] = float("inf")
    changed = QBPassingYardsDatasetSplit(split.train, split.validation, invalid)
    assert select_qb_passing_yards_final_model(changed).report == expected
    shuffled = QBPassingYardsDatasetSplit(split.train.sample(frac=1, random_state=1), split.validation.sample(frac=1, random_state=2), split.test)
    assert select_qb_passing_yards_final_model(shuffled).report == expected
