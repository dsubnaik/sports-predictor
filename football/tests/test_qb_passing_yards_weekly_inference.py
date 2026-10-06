"""Focused pregame-only tests for QB weekly inference."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pandas as pd
import pytest

import football.pipeline.qb_passing_yards_weekly_inference as weekly_inference
from football.pipeline.qb_passing_yards_weekly_inference import (
    build_qb_passing_yards_weekly_feature_rows,
    generate_qb_passing_yards_weekly_projections,
)
from football.training.build_qb_passing_yards_dataset import FEATURE_COLUMNS
from football.training.qb_passing_yards_production_model import (
    QBPassingYardsLoadedArtifact,
    train_qb_passing_yards_production_model,
)
from football.training.split_qb_passing_yards_dataset import QBPassingYardsDatasetSplit


def _schedule() -> pd.DataFrame:
    rows = []
    for week, game, kickoff, home, away in [
        (1, "g1", "2026-09-10T17:00:00Z", "AAA", "BBB"),
        (2, "g2", "2026-09-17T17:00:00Z", "AAA", "CCC"),
        (2, "g3", "2026-09-18T17:00:00Z", "DDD", "BBB"),
        (3, "g4", "2026-09-24T17:00:00Z", "AAA", "BBB"),
    ]:
        rows.extend([
            {"season": 2026, "week": week, "game_id": game, "team": home, "opponent": away, "home_away": "home", "scheduled_kickoff": kickoff},
            {"season": 2026, "week": week, "game_id": game, "team": away, "opponent": home, "home_away": "away", "scheduled_kickoff": kickoff},
        ])
    return pd.DataFrame(rows)


def _qbs() -> pd.DataFrame:
    rows = []
    for week, game, player, team, opponent, yards, attempts in [
        (1, "g1", "qa", "AAA", "BBB", 200, 20),
        (1, "g1", "qb", "BBB", "AAA", 180, 18),
        (2, "g2", "qa", "AAA", "CCC", 300, 30),
        (2, "g2", "qc", "CCC", "AAA", 250, 25),
        (2, "g3", "qd", "DDD", "BBB", 150, 15),
        (2, "g3", "qb", "BBB", "DDD", 220, 22),
    ]:
        rows.append({"season": 2026, "week": week, "game_id": game, "player_id": player, "player_name": player, "team": team, "opponent": opponent, "passing_attempts": attempts, "completions": attempts - 5, "passing_yards": yards, "passing_touchdowns": 1, "interceptions": 0})
    return pd.DataFrame(rows)


def _snapshots() -> pd.DataFrame:
    return pd.DataFrame([
        {"team": "AAA", "player_id": "qa", "player_name": "QA", "position": "QB", "snapshot_timestamp": "2026-09-23T17:00:00Z", "depth_rank": 1},
        {"team": "AAA", "player_id": "qx", "position": "QB", "snapshot_timestamp": "2026-09-23T17:00:00Z", "depth_rank": 2},
        {"team": "BBB", "player_id": "qb", "player_name": "QB", "position": "QB", "snapshot_timestamp": "2026-09-23T17:00:00Z", "depth_rank": 1},
    ])


def _result(qbs: pd.DataFrame | None = None):
    return build_qb_passing_yards_weekly_feature_rows(qbs if qbs is not None else _qbs(), _schedule(), _snapshots(), target_season=2026, target_week=3, as_of_utc="2026-09-24T12:00:00Z")


def test_constructs_selected_qb_rows_from_pregame_snapshots_with_strict_history():
    result = _result()
    assert [row.player_id for row in result.rows] == ["qa", "qb"]
    qa = result.rows[0]
    assert qa.feature_values[:7] == pytest.approx((250.0, 250.0, 25.0, 25.0, 2.0, 2.0, False))
    # BBB has allowed 200 and 150; rank 1 is the highest permitted mean allowed.
    assert qa.feature_values[7:13] == pytest.approx((175.0, 175.0, 17.5, 2.0, 2.0, False))
    assert qa.feature_values[13] == pytest.approx(4.0)
    assert result.cold_start_projections == 0


def test_target_week_and_future_outcomes_cannot_change_rows():
    original = _result()
    altered = _qbs()
    altered.loc[len(altered)] = {"season": 2026, "week": 3, "game_id": "g4", "player_id": "qa", "player_name": "qa", "team": "AAA", "opponent": "BBB", "passing_attempts": 99, "completions": 99, "passing_yards": 999, "passing_touchdowns": 9, "interceptions": 0}
    altered.loc[len(altered)] = {"season": 2026, "week": 4, "game_id": "future", "player_id": "qa", "player_name": "qa", "team": "AAA", "opponent": "BBB", "passing_attempts": 99, "completions": 99, "passing_yards": 999, "passing_touchdowns": 9, "interceptions": 0}
    schedule = _schedule()
    schedule = pd.concat([schedule, pd.DataFrame([
        {"season": 2026, "week": 4, "game_id": "future", "team": "AAA", "opponent": "BBB", "home_away": "home", "scheduled_kickoff": "2026-10-01T17:00:00Z"},
        {"season": 2026, "week": 4, "game_id": "future", "team": "BBB", "opponent": "AAA", "home_away": "away", "scheduled_kickoff": "2026-10-01T17:00:00Z"},
    ])], ignore_index=True)
    changed = build_qb_passing_yards_weekly_feature_rows(altered, schedule, _snapshots(), target_season=2026, target_week=3, as_of_utc="2026-09-24T12:00:00Z")
    assert changed.rows == original.rows


def test_as_of_and_depth_chart_cutoffs_leave_unknown_diagnostics():
    snapshots = _snapshots().astype({"depth_rank": float})
    snapshots.loc[snapshots["team"].eq("BBB"), "snapshot_timestamp"] = "2026-09-24T17:00:00Z"
    result = build_qb_passing_yards_weekly_feature_rows(_qbs(), _schedule(), snapshots, target_season=2026, target_week=3, as_of_utc="2026-09-24T12:00:00Z")
    assert [row.player_id for row in result.rows] == ["qa"]
    assert result.unknown_team_games == 1
    assert any(skip.reason == "no_usable_pregame_depth_chart_candidate" for skip in result.skips)


def test_same_time_kickoff_is_not_pregame_eligible_and_inputs_are_unchanged():
    schedule = _schedule()
    qbs = _qbs()
    before_schedule = schedule.copy(deep=True)
    before_qbs = qbs.copy(deep=True)
    result = build_qb_passing_yards_weekly_feature_rows(qbs, schedule, _snapshots(), target_season=2026, target_week=3, as_of_utc="2026-09-24T17:00:00Z")
    assert not result.rows
    assert result.eligible_games == 0
    pd.testing.assert_frame_equal(schedule, before_schedule)
    pd.testing.assert_frame_equal(qbs, before_qbs)


def test_rejects_naive_as_of_and_invalid_depth_rank():
    with pytest.raises(ValueError, match="timezone-aware"):
        build_qb_passing_yards_weekly_feature_rows(_qbs(), _schedule(), _snapshots(), target_season=2026, target_week=3, as_of_utc="2026-09-24 12:00:00")
    snapshots = _snapshots().astype({"depth_rank": float})
    snapshots.loc[0, "depth_rank"] = 1.5
    with pytest.raises(ValueError, match="invalid ranks"):
        _result.__wrapped__ if False else build_qb_passing_yards_weekly_feature_rows(_qbs(), _schedule(), snapshots, target_season=2026, target_week=3, as_of_utc="2026-09-24T12:00:00Z")


def _loaded_artifact() -> QBPassingYardsLoadedArtifact:
    rows = []
    for index, (season, week) in enumerate([(2021, 1), (2021, 2), (2021, 3), (2021, 4), (2025, 1), (2025, 2)]):
        row = {"season": season, "week": week, "game_id": f"fit-{index}", "player_id": f"fit-qb-{index}", "player_name": "fixture", "team": "FIT", "opponent": "OPP", "home_away": "home", "target_passing_yards": float(200 + index)}
        row.update({name: (False if name.endswith("missing_history") else float(index + 1)) for name in FEATURE_COLUMNS})
        rows.append(row)
    data = pd.DataFrame(rows)
    model = train_qb_passing_yards_production_model(QBPassingYardsDatasetSplit(data.iloc[:4].copy(), data.iloc[4:].copy(), data.iloc[:1].copy()), created_at_utc="2026-01-01T00:00:00+00:00")
    return QBPassingYardsLoadedArtifact(model.model, model.metadata, "fixture.joblib")


def test_projection_reuses_production_contract_and_excludes_outcomes(monkeypatch):
    artifact = _loaded_artifact()
    report = generate_qb_passing_yards_weekly_projections(artifact, _qbs(), _schedule(), _snapshots(), target_season=2026, target_week=3, as_of_utc="2026-09-24T12:00:00Z")
    assert len(report.projections) == 2
    assert all(projection.predicted_passing_yards == pytest.approx(projection.predicted_passing_yards) for projection in report.projections)
    forbidden = {"target_passing_yards", "actual_passing_yards", "residual", "sportsbook_line"}
    assert forbidden.isdisjoint(report.projections[0].__dataclass_fields__)
    assert report.projections[0].model_schema_version == artifact.metadata.schema_version
    assert report.projections[0].artifact_identifier == "fixture.joblib"


def test_schedule_status_invalid_kickoff_postseason_and_boundary_diagnostics():
    schedule = _schedule()
    schedule["season_type"] = "REG"
    schedule.loc[schedule["game_id"].eq("g4"), "season_type"] = "POST"
    result = build_qb_passing_yards_weekly_feature_rows(_qbs(), schedule, _snapshots(), target_season=2026, target_week=3, as_of_utc="2026-09-24T12:00:00Z")
    assert not result.rows
    # A postseason target is not counted as an eligible regular-season game.
    assert result.scheduled_games_considered == 0
    schedule.loc[schedule["game_id"].eq("g4"), "season_type"] = "REG"
    schedule.loc[schedule["team"].eq("AAA") & schedule["game_id"].eq("g4"), "scheduled_kickoff"] = None
    result = build_qb_passing_yards_weekly_feature_rows(_qbs(), schedule, _snapshots(), target_season=2026, target_week=3, as_of_utc="2026-09-24T12:00:00Z")
    assert {skip.reason for skip in result.skips} == {"invalid_or_missing_kickoff"}
    schedule.loc[schedule["game_id"].eq("g4"), "status"] = "completed"
    result = build_qb_passing_yards_weekly_feature_rows(_qbs(), schedule, _snapshots(), target_season=2026, target_week=3, as_of_utc="2026-09-24T12:00:00Z")
    assert {skip.reason for skip in result.skips} == {"terminal_game_status"}


def test_schedule_conflicts_and_duplicate_team_game_keys_fail():
    duplicate = pd.concat([_schedule(), _schedule().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        build_qb_passing_yards_weekly_feature_rows(_qbs(), duplicate, _snapshots(), target_season=2026, target_week=3, as_of_utc="2026-09-24T12:00:00Z")
    conflicting = _schedule()
    conflicting.loc[conflicting.index[1], "opponent"] = "CCC"
    with pytest.raises(ValueError, match="conflicting"):
        build_qb_passing_yards_weekly_feature_rows(_qbs(), conflicting, _snapshots(), target_season=2026, target_week=3, as_of_utc="2026-09-24T12:00:00Z")


def test_latest_snapshot_lowest_rank_and_player_tie_break_are_deterministic():
    snapshots = _snapshots().astype({"depth_rank": float})
    # An older rank-one candidate must not beat the latest snapshot.
    snapshots.loc[len(snapshots)] = {"team": "AAA", "player_id": "old", "position": "QB", "snapshot_timestamp": "2026-09-22T17:00:00Z", "depth_rank": 1.0}
    snapshots.loc[snapshots["team"].eq("AAA"), "snapshot_timestamp"] = "2026-09-23T17:00:00Z"
    snapshots.loc[snapshots["player_id"].eq("qa"), "depth_rank"] = 2.0
    snapshots.loc[len(snapshots)] = {"team": "AAA", "player_id": "aa", "position": "QB", "snapshot_timestamp": "2026-09-23T17:00:00Z", "depth_rank": 1.0}
    snapshots.loc[len(snapshots)] = {"team": "AAA", "player_id": "zz", "position": "QB", "snapshot_timestamp": "2026-09-23T17:00:00Z", "depth_rank": 1.0}
    result = build_qb_passing_yards_weekly_feature_rows(_qbs(), _schedule(), snapshots, target_season=2026, target_week=3, as_of_utc="2026-09-24T12:00:00Z")
    assert [row.player_id for row in result.rows if row.team == "AAA"] == ["aa"]
    assert len([row for row in result.rows if row.team == "AAA"]) == 1


@pytest.mark.parametrize("value", [0.0, -1.0, 1.5, np.inf])
def test_invalid_depth_ranks_fail(value):
    snapshots = _snapshots().astype({"depth_rank": float})
    snapshots.loc[0, "depth_rank"] = value
    with pytest.raises(ValueError, match="invalid ranks"):
        build_qb_passing_yards_weekly_feature_rows(_qbs(), _schedule(), snapshots, target_season=2026, target_week=3, as_of_utc="2026-09-24T12:00:00Z")


def test_cold_start_and_week_one_use_only_prior_season_history():
    schedule = _schedule()
    prior = _qbs()
    prior["season"] = 2025
    prior["week"] = 18
    prior["game_id"] = prior["game_id"].map(lambda value: f"prior-{value}")
    historic_schedule = schedule.copy()
    historic_schedule["season"] = 2025
    historic_schedule["week"] = 18
    historic_schedule["game_id"] = historic_schedule["game_id"].map(lambda value: f"prior-{value}")
    target = schedule.loc[schedule["week"].eq(1)].copy()
    target["week"] = 1
    target["game_id"] = "week-one"
    all_schedule = pd.concat([historic_schedule, target], ignore_index=True)
    snapshots = _snapshots()
    snapshots["snapshot_timestamp"] = "2026-09-09T12:00:00Z"
    result = build_qb_passing_yards_weekly_feature_rows(prior, all_schedule, snapshots, target_season=2026, target_week=1, as_of_utc="2026-09-10T12:00:00Z")
    qa = next(row for row in result.rows if row.player_id == "qa")
    assert qa.feature_values[0] == pytest.approx(250.0)
    assert qa.feature_values[4:7] == pytest.approx((2.0, 2.0, False))
    cold_snapshots = snapshots.loc[~snapshots["team"].eq("AAA")].copy()
    cold = build_qb_passing_yards_weekly_feature_rows(prior, all_schedule, cold_snapshots, target_season=2026, target_week=1, as_of_utc="2026-09-10T12:00:00Z")
    assert cold.unknown_team_games == 1


def test_projection_frame_is_exactly_canonical_and_report_is_frozen(monkeypatch):
    captured = {}
    original = weekly_inference.predict_qb_passing_yards_production_model

    def record(artifact, frame):
        captured["columns"] = tuple(frame.columns)
        assert "target_passing_yards" not in frame
        return original(artifact, frame)

    monkeypatch.setattr(weekly_inference, "predict_qb_passing_yards_production_model", record)
    report = generate_qb_passing_yards_weekly_projections(_loaded_artifact(), _qbs(), _schedule(), _snapshots(), target_season=2026, target_week=3, as_of_utc="2026-09-24T12:00:00Z")
    assert captured["columns"] == ("season", "week", "game_id", "player_id", *FEATURE_COLUMNS)
    with pytest.raises(FrozenInstanceError):
        report.feature_diagnostics.eligible_games = 1
    assert not any(isinstance(value, pd.DataFrame) for value in report.__dict__.values())


def test_incompatible_loaded_artifact_is_rejected_before_projection():
    artifact = _loaded_artifact()
    bad_metadata = replace(artifact.metadata, schema_version=999)
    with pytest.raises(ValueError, match="schema"):
        generate_qb_passing_yards_weekly_projections(QBPassingYardsLoadedArtifact(artifact.model, bad_metadata, artifact.artifact_name), _qbs(), _schedule(), _snapshots(), target_season=2026, target_week=3, as_of_utc="2026-09-24T12:00:00Z")
