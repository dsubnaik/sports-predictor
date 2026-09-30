from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from football.tests.test_qb_passing_yards_model_comparison import _split
from football.training.qb_likely_primary_population_analysis import (
    LIKELY_PRIMARY,
    NOT_LIKELY_PRIMARY,
    UNKNOWN,
    _classify_validation,
    analyze_qb_likely_primary_population,
    audit_qb_primary_qb_feasibility,
)
from football.training.split_qb_passing_yards_dataset import QBPassingYardsDatasetSplit


def _schedule(split):
    rows = []
    for index, row in split.validation.reset_index(drop=True).iterrows():
        rows.append({"season": row.season, "week": row.week, "game_id": row.game_id, "team": row.team, "game_date": f"2024-01-{10 + index:02d}"})
    return pd.DataFrame(rows)


def _snapshots(split):
    rows = []
    for index, row in split.validation.reset_index(drop=True).iterrows():
        rows.extend([
            {"team": row.team, "player_id": row.player_id, "position": "QB", "snapshot_timestamp": f"2024-01-{9 + index:02d}T00:00:00Z", "depth_rank": 1},
            {"team": row.team, "player_id": f"backup_{index}", "position": "QB", "snapshot_timestamp": f"2024-01-{9 + index:02d}T00:00:00Z", "depth_rank": 2},
        ])
    return pd.DataFrame(rows)


def test_feasibility_audit_is_immutable_and_distinguishes_postgame_signals():
    report = audit_qb_primary_qb_feasibility()
    assert report.selector_supported_for_dated_snapshots is True
    signals = {item.signal: item for item in report.signals}
    assert signals["weekly player statistics"].known_before_kickoff is False
    assert signals["dated depth chart"].identifies_nonparticipants is True
    assert signals["legacy depth chart"].known_before_kickoff is False
    with pytest.raises(FrozenInstanceError):
        report.conclusion = "changed"


def test_strictly_prior_depth_snapshots_select_candidates_without_target_participation():
    split = _split()
    validation = split.validation.sort_values(["season", "week", "game_id", "player_id"], kind="mergesort").reset_index(drop=True)
    categories, summary = _classify_validation(validation, _schedule(split), _snapshots(split))
    assert categories.tolist() == [LIKELY_PRIMARY] * len(validation)
    assert summary.one_selected_team_games == len(validation)
    assert summary.zero_selected_team_games == 0
    forbidden_changed = validation.assign(passing_attempts=9999, passing_yards=-9999, snaps=0, starts=False, participated=False)
    changed_categories, _ = _classify_validation(forbidden_changed, _schedule(split), _snapshots(split))
    assert changed_categories.tolist() == categories.tolist()

    same_day = _snapshots(split).copy()
    same_day["snapshot_timestamp"] = "2024-01-31T23:59:59Z"
    categories, summary = _classify_validation(validation, _schedule(split), same_day)
    assert set(categories) == {UNKNOWN}
    assert summary.zero_selected_team_games == len(validation)


def test_tie_breaking_unknown_and_invalid_schedule_are_deterministic():
    split = _split()
    validation = split.validation.sort_values(["season", "week", "game_id", "player_id"], kind="mergesort").reset_index(drop=True)
    snapshots = _snapshots(split)
    snapshots.loc[snapshots["depth_rank"].eq(2), "depth_rank"] = 1
    snapshots.loc[snapshots["depth_rank"].eq(2), "player_id"] = "aaa_tie_winner"
    categories, _ = _classify_validation(validation, _schedule(split), snapshots)
    assert set(categories) == {NOT_LIKELY_PRIMARY}
    with pytest.raises(ValueError, match="duplicate team-game"):
        _classify_validation(validation, pd.concat([_schedule(split), _schedule(split).iloc[[0]]]), _snapshots(split))


def test_validation_only_analysis_reports_full_populations_wins_and_no_mutation(monkeypatch):
    split = _split()
    original_train, original_validation, original_test = (split.train.copy(deep=True), split.validation.copy(deep=True), split.test.copy(deep=True))
    calls = []
    import football.training.qb_likely_primary_population_analysis as module
    original = module.predict_qb_passing_yards_baseline
    monkeypatch.setattr(module, "predict_qb_passing_yards_baseline", lambda model, frame: (calls.append(frame), original(model, frame))[1])
    result = analyze_qb_likely_primary_population(split, _schedule(split), _snapshots(split))
    assert len(calls) == 1 and calls[0] is not split.test
    assert result.selection_summary.coverage_rate == pytest.approx(1.0)
    assert result.selection_summary.observed_unknown_rate == pytest.approx(0.0)
    assert all(item.full_population.row_count == len(split.validation) for item in result.model_diagnostics)
    assert all(item.populations[0].row_count == len(split.validation) for item in result.model_diagnostics)
    assert all(item.populations[1].metrics is None and item.populations[2].metrics is None for item in result.model_diagnostics)
    assert len(result.row_level_win_counts) == 15
    pd.testing.assert_frame_equal(split.train, original_train)
    pd.testing.assert_frame_equal(split.validation, original_validation)
    pd.testing.assert_frame_equal(split.test, original_test)

    invalid_test = split.test.astype(object).copy(deep=True)
    invalid_test.loc[:, :] = "invalid test sentinel"
    same = analyze_qb_likely_primary_population(QBPassingYardsDatasetSplit(split.train, split.validation, invalid_test), _schedule(split), _snapshots(split))
    assert same == result


def test_shuffled_rows_preserve_results_and_empty_unknown_population_is_explicit():
    split = _split()
    expected = analyze_qb_likely_primary_population(split, _schedule(split), _snapshots(split))
    shuffled = QBPassingYardsDatasetSplit(split.train.sample(frac=1, random_state=3), split.validation.sample(frac=1, random_state=4), split.test)
    assert analyze_qb_likely_primary_population(shuffled, _schedule(split), _snapshots(split)) == expected
