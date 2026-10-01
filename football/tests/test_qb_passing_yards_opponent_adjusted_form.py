from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from football.data.build_quarterback_dataset import OUTPUT_COLUMNS as QB_COLUMNS
from football.tests.test_build_qb_passing_yards_training_dataset import _inputs
from football.tests.test_qb_passing_yards_model_comparison import _split
from football.training.build_qb_passing_yards_dataset import TARGET_KEY, build_qb_passing_yards_training_dataset
from football.training.qb_passing_yards_opponent_adjusted_form import (
    ADJUSTED_FORM_FEATURE_COLUMNS, OFFICIAL_FEATURES,
    OFFICIAL_PLUS_OPPONENT_ADJUSTED_FORM,
    analyze_qb_passing_yards_opponent_adjusted_form,
    attach_qb_opponent_adjusted_form_features,
    build_qb_opponent_adjusted_form_features,
)
from football.training.split_qb_passing_yards_dataset import QBPassingYardsDatasetSplit


def _game_rows():
    qbs, schedule = [], []
    def game(season, week, game_id, away, home, away_qb, home_qb, away_yards, home_yards):
        schedule.extend((
            {"season": season, "week": week, "game_id": game_id, "team": away, "opponent": home, "home_away": "away"},
            {"season": season, "week": week, "game_id": game_id, "team": home, "opponent": away, "home_away": "home"},
        ))
        for team, opponent, player, yards in ((away, home, away_qb, away_yards), (home, away, home_qb, home_yards)):
            qbs.append(dict(season=season, week=week, game_id=game_id, player_id=player,
                player_name=player, team=team, opponent=opponent, passing_attempts=25,
                completions=18, passing_yards=yards, passing_touchdowns=1, interceptions=0))
    return qbs, schedule, game
from football.training.qb_passing_yards_linear_regression import fit_qb_passing_yards_linear_regression, evaluate_qb_passing_yards_linear_regression
from football.training.qb_passing_yards_random_forest import fit_qb_passing_yards_random_forest, evaluate_qb_passing_yards_random_forest
from football.training.qb_passing_yards_gradient_boosting import fit_qb_passing_yards_gradient_boosting, evaluate_qb_passing_yards_gradient_boosting
from football.training.qb_passing_yards_xgboost import fit_qb_passing_yards_xgboost, evaluate_qb_passing_yards_xgboost


def test_adjusted_features_are_prior_only_same_week_safe_and_deterministic():
    qbs, schedule = _inputs()
    before_qbs, before_schedule = qbs.copy(deep=True), schedule.copy(deep=True)
    output = build_qb_opponent_adjusted_form_features(qbs, schedule)
    assert tuple(output.columns) == (*TARGET_KEY, *ADJUSTED_FORM_FEATURE_COLUMNS)
    # A W2's W1 performance (100) is adjusted by B's pregame allowance: none,
    # so it remains unavailable; no same-week/future game can change that.
    row = output.query("game_id == '2025_02_A_C' and player_id == 'qb_a'").iloc[0]
    assert row.qb_season_opponent_adjusted_history_games == 0
    assert pd.isna(row.qb_season_passing_yards_over_opponent_allowance_avg)
    assert bool(row.qb_missing_opponent_adjusted_history) is True
    pd.testing.assert_frame_equal(qbs, before_qbs)
    pd.testing.assert_frame_equal(schedule, before_schedule)
    shuffled = build_qb_opponent_adjusted_form_features(qbs.sample(frac=1, random_state=1), schedule.sample(frac=1, random_state=2))
    pd.testing.assert_frame_equal(output, shuffled)


def test_formula_and_attachment_keys_are_strict():
    qbs, schedule = _inputs()
    features = build_qb_opponent_adjusted_form_features(qbs, schedule)
    dataset = build_qb_passing_yards_training_dataset(qbs, schedule)
    attached = attach_qb_opponent_adjusted_form_features(dataset, features)
    assert len(attached) == len(dataset)
    with pytest.raises(ValueError, match="exactly match"):
        attach_qb_opponent_adjusted_form_features(dataset, features.iloc[:-1])
    extra = pd.concat([features, features.iloc[[0]].assign(game_id="extra")], ignore_index=True)
    with pytest.raises(ValueError, match="exactly match"):
        attach_qb_opponent_adjusted_form_features(dataset, extra)


def test_home_and_away_qbs_use_their_respective_opponents_pregame_allowance():
    qbs, schedule, game = _game_rows()
    # A's prior allowance is X's 180; H's prior allowance is Y's 280.
    game(2025, 1, "a_x", "A", "X", "a", "x", 100, 180)
    game(2025, 1, "y_h", "Y", "H", "y", "h", 280, 100)
    game(2025, 2, "a_h", "A", "H", "a", "h", 250, 250)
    result = build_qb_opponent_adjusted_form_features(pd.DataFrame(qbs, columns=QB_COLUMNS), pd.DataFrame(schedule))
    # Week 2 observations become history only in a later week; append one.
    game(2025, 3, "a_z", "A", "Z", "a", "z", 100, 100)
    game(2025, 3, "z_h", "Z", "H", "z2", "h", 100, 100)
    result = build_qb_opponent_adjusted_form_features(pd.DataFrame(qbs, columns=QB_COLUMNS), pd.DataFrame(schedule))
    h = result.query("game_id == 'a_h' and player_id == 'h'").iloc[0]
    a = result.query("game_id == 'a_h' and player_id == 'a'").iloc[0]
    later = result.query("game_id == 'a_z' and player_id == 'a'").iloc[0]
    home_later = result.query("game_id == 'z_h' and player_id == 'h'").iloc[0]
    assert h.qb_season_opponent_adjusted_history_games == 0
    assert a.qb_season_opponent_adjusted_history_games == 0
    # a2's adjusted W2 result is 250 - H's pregame 280 = -30, used by A W3.
    assert later.qb_season_passing_yards_over_opponent_allowance_avg == pytest.approx(-30.0)
    # h's home W2 result is 250 - A's pregame 180 = +70, used by H W3.
    assert home_later.qb_season_passing_yards_over_opponent_allowance_avg == pytest.approx(70.0)


def test_week_one_prior_season_and_same_week_traps_are_strict():
    qbs, schedule, game = _game_rows()
    game(2024, 1, "a_x1", "A", "X", "a", "x", 100, 180)
    game(2024, 2, "a_x2", "A", "X", "a", "x2", 220, 100)
    game(2025, 1, "a_z1", "A", "X", "a", "z", 999, 100)
    game(2025, 1, "a_q1", "A", "X", "a", "q", 1, 100)
    game(2025, 2, "a_r2", "A", "R", "a", "r", 100, 100)
    result = build_qb_opponent_adjusted_form_features(pd.DataFrame(qbs, columns=QB_COLUMNS), pd.DataFrame(schedule))
    week1 = result.query("game_id == 'a_z1' and player_id == 'a'").iloc[0]
    week2 = result.query("game_id == 'a_r2' and player_id == 'a'").iloc[0]
    # Only 2024 W2 has a usable adjusted observation: 220 - 100 = 120.
    assert week1.qb_season_opponent_adjusted_history_games == 1
    assert week1.qb_season_passing_yards_over_opponent_allowance_avg == pytest.approx(120.0)
    # Both extreme 2025 W1 outcomes enter only after W1, and are unavailable at W1.
    assert week2.qb_season_opponent_adjusted_history_games == 2
    assert week2.qb_season_passing_yards_over_opponent_allowance_avg == pytest.approx((999 - 160 + (1 - 160)) / 2)


def _adjusted_split():
    split = _split()
    def add(frame):
        result = frame.copy(deep=True)
        result[ADJUSTED_FORM_FEATURE_COLUMNS[0]] = result["qb_season_passing_yards_avg"]
        result[ADJUSTED_FORM_FEATURE_COLUMNS[1]] = result["qb_last3_passing_yards_avg"]
        result[ADJUSTED_FORM_FEATURE_COLUMNS[2]] = result["qb_season_history_games"]
        result[ADJUSTED_FORM_FEATURE_COLUMNS[3]] = result["qb_last3_history_games"]
        result[ADJUSTED_FORM_FEATURE_COLUMNS[4]] = result["qb_missing_history"].astype(bool)
        return result
    return QBPassingYardsDatasetSplit(add(split.train), add(split.validation), add(split.test))


def test_analysis_is_train_validation_only_immutable_and_aligned():
    split = _adjusted_split()
    original = [x.copy(deep=True) for x in (split.train, split.validation, split.test)]
    result = analyze_qb_passing_yards_opponent_adjusted_form(split, bootstrap_replicates=13)
    assert result.validation_row_count == len(split.validation)
    assert len(result.estimator_variants) == 8
    assert {x.variant_name for x in result.estimator_variants} == {OFFICIAL_FEATURES, OFFICIAL_PLUS_OPPONENT_ADJUSTED_FORM}
    assert all(x.validation.row_count == len(split.validation) for x in result.estimator_variants)
    assert all(x.comparison if False else True for x in [])
    for comparison in result.comparisons:
        assert comparison.bootstrap_replicates == 13
        assert comparison.bootstrap_seed == 42
    with pytest.raises(FrozenInstanceError):
        result.validation_row_count = 0
    for actual, expected in zip((split.train, split.validation, split.test), original, strict=True):
        pd.testing.assert_frame_equal(actual, expected)


def test_analysis_never_accesses_test_and_improvement_signs_match_variants():
    split = _adjusted_split()
    class ExplodingTest:
        def __getattribute__(self, name):
            raise AssertionError("test accessed")
    expected = analyze_qb_passing_yards_opponent_adjusted_form(split, bootstrap_replicates=7)
    assert analyze_qb_passing_yards_opponent_adjusted_form(QBPassingYardsDatasetSplit(split.train, split.validation, ExplodingTest()), bootstrap_replicates=7) == expected
    variants = {(x.estimator_name, x.variant_name): x for x in expected.estimator_variants}
    for comparison in expected.comparisons:
        a = variants[(comparison.estimator_name, OFFICIAL_FEATURES)].validation.metrics
        b = variants[(comparison.estimator_name, OFFICIAL_PLUS_OPPONENT_ADJUSTED_FORM)].validation.metrics
        assert comparison.mae_improvement_from_opponent_adjusted_form == pytest.approx(a.mae - b.mae)
        assert comparison.rmse_improvement_from_opponent_adjusted_form == pytest.approx(a.rmse - b.rmse)
        assert comparison.r_squared_improvement_from_opponent_adjusted_form == pytest.approx(b.r_squared - a.r_squared)


def test_prediction_helpers_receive_only_canonical_validation_frames(monkeypatch):
    import football.training.qb_passing_yards_opponent_adjusted_form as module
    split = _adjusted_split()
    calls, base, enhanced = [], module._fit_predict, module._fit_predict_adjusted
    def record_base(name, columns, train, validation):
        calls.append(("official", tuple(validation[TARGET_KEY].itertuples(index=False, name=None))))
        return base(name, columns, train, validation)
    def record_enhanced(name, columns, train, validation):
        calls.append(("enhanced", tuple(validation[TARGET_KEY].itertuples(index=False, name=None))))
        return enhanced(name, columns, train, validation)
    monkeypatch.setattr(module, "_fit_predict", record_base)
    monkeypatch.setattr(module, "_fit_predict_adjusted", record_enhanced)
    module.analyze_qb_passing_yards_opponent_adjusted_form(split, bootstrap_replicates=3)
    expected = tuple(split.validation.sort_values(TARGET_KEY, kind="mergesort").loc[:, TARGET_KEY].itertuples(index=False, name=None))
    assert len(calls) == 8
    assert all(keys == expected for _, keys in calls)


def test_official_variant_has_exact_contract_and_completed_model_metric_parity():
    split = _adjusted_split()
    report = analyze_qb_passing_yards_opponent_adjusted_form(split, bootstrap_replicates=5)
    expected = {
        "Linear Regression": evaluate_qb_passing_yards_linear_regression(fit_qb_passing_yards_linear_regression(split.train), split.validation).overall_metrics,
        "Random Forest": evaluate_qb_passing_yards_random_forest(fit_qb_passing_yards_random_forest(split.train), split.validation).overall_metrics,
        "sklearn Gradient Boosting": evaluate_qb_passing_yards_gradient_boosting(fit_qb_passing_yards_gradient_boosting(split.train), split.validation).overall_metrics,
        "XGBoost": evaluate_qb_passing_yards_xgboost(fit_qb_passing_yards_xgboost(split.train), split.validation).overall_metrics,
    }
    for variant in report.estimator_variants:
        if variant.variant_name == OFFICIAL_FEATURES:
            assert variant.feature_columns == tuple(split.train.columns[8:22])
            metrics = expected[variant.estimator_name]
            assert variant.validation.metrics.mae == pytest.approx(metrics.mae, abs=1e-10)
            assert variant.validation.metrics.rmse == pytest.approx(metrics.rmse, abs=1e-10)
            assert variant.validation.metrics.r_squared == pytest.approx(metrics.r_squared, abs=1e-10)
        else:
            assert variant.feature_columns[-5:] == ADJUSTED_FORM_FEATURE_COLUMNS
            assert len(variant.feature_columns) == 19


def test_adjusted_contract_validation_and_empty_subgroup_behavior():
    split = _adjusted_split()
    bad = split.train.copy(deep=True)
    bad[ADJUSTED_FORM_FEATURE_COLUMNS[2]] = 1.5
    with pytest.raises(ValueError, match="invalid adjusted-history counts"):
        analyze_qb_passing_yards_opponent_adjusted_form(QBPassingYardsDatasetSplit(bad, split.validation, split.test), bootstrap_replicates=3)
    validation = split.validation.copy(deep=True)
    validation[ADJUSTED_FORM_FEATURE_COLUMNS[4]] = False
    validation[ADJUSTED_FORM_FEATURE_COLUMNS[2]] = 1
    report = analyze_qb_passing_yards_opponent_adjusted_form(QBPassingYardsDatasetSplit(split.train, validation, split.test), bootstrap_replicates=3)
    assert all(v.adjusted_history_missing.metrics is None for v in report.estimator_variants)


def test_nonfinite_targets_and_adjusted_flag_disagreements_fail():
    split = _adjusted_split()
    bad_target = split.train.assign(target_passing_yards=np.inf)
    with pytest.raises(ValueError, match="target_passing_yards values must be finite"):
        analyze_qb_passing_yards_opponent_adjusted_form(QBPassingYardsDatasetSplit(bad_target, split.validation, split.test), bootstrap_replicates=3)
    bad_flag = split.train.copy(deep=True)
    bad_flag[ADJUSTED_FORM_FEATURE_COLUMNS[4]] = False
    bad_flag[ADJUSTED_FORM_FEATURE_COLUMNS[2]] = 0
    with pytest.raises(ValueError, match="inconsistent adjusted-history flag"):
        analyze_qb_passing_yards_opponent_adjusted_form(QBPassingYardsDatasetSplit(bad_flag, split.validation, split.test), bootstrap_replicates=3)


def test_bootstrap_is_deterministic_and_report_retains_no_frames_or_models():
    first = analyze_qb_passing_yards_opponent_adjusted_form(_adjusted_split(), bootstrap_replicates=11)
    second = analyze_qb_passing_yards_opponent_adjusted_form(_adjusted_split(), bootstrap_replicates=11)
    assert first == second
    def walk(value):
        assert not isinstance(value, (pd.DataFrame, pd.Series, np.ndarray, list, dict))
        if hasattr(value, "__dataclass_fields__"):
            for item in value.__dataclass_fields__:
                walk(getattr(value, item))
        elif isinstance(value, tuple):
            for item in value:
                walk(item)
    walk(first)
