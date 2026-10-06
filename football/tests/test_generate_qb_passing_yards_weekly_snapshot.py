"""Tests for the explicit weekly QB projection command orchestration."""
from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path

import pandas as pd
import pytest

command = importlib.import_module(
    "football.pipeline.generate_qb_passing_yards_weekly_snapshot"
)
from football.pipeline.qb_passing_yards_weekly_inference import (
    QBPassingYardsWeeklyFeatureRows,
    QBPassingYardsWeeklyProjectionReport,
)
from football.tests.test_qb_passing_yards_weekly_projection_snapshots import _report


@dataclass(frozen=True)
class _Metadata:
    training_row_count: int = 3279
    maximum_training_period: tuple[int, int] = (2025, 18)


@dataclass(frozen=True)
class _Artifact:
    metadata: _Metadata = _Metadata()
    artifact_name: str = "production.joblib"


def _qbs() -> pd.DataFrame:
    return pd.DataFrame({
        "season": [2025, 2026, 2026], "week": [18, 4, 5],
        "game_id": ["old", "prior", "target"], "player_id": ["a", "a", "a"],
    })


def _diagnostic_report() -> QBPassingYardsWeeklyProjectionReport:
    source = _report()
    diagnostics = source.feature_diagnostics
    projection = source.projections[0]
    return QBPassingYardsWeeklyProjectionReport(
        projections=(projection,),
        feature_diagnostics=QBPassingYardsWeeklyFeatureRows(
            rows=(), skips=(), target_season=2026, target_week=5,
            as_of_utc="2026-10-06T14:00:00+00:00",
            scheduled_games_considered=1, eligible_games=1,
            team_games_considered=2, likely_primary_selections=1,
            unknown_team_games=1, cold_start_projections=1,
            defense_missing_projections=0,
        ),
    )


def _patch_generation(monkeypatch, tmp_path):
    artifact = tmp_path / "production.joblib"
    artifact.write_bytes(b"trusted-model")
    seen: dict[str, object] = {}
    monkeypatch.setattr(command, "load_qb_passing_yards_production_model", lambda path: _Artifact())
    monkeypatch.setattr(command, "load_player_game_stats", lambda seasons, loader=None: _qbs())
    monkeypatch.setattr(command, "normalize_quarterback_game_stats", lambda value: value.copy())
    monkeypatch.setattr(command, "load_schedules", lambda seasons, loader=None: pd.DataFrame({"ignored": []}))
    monkeypatch.setattr(command, "_inference_schedule", lambda value: value)
    monkeypatch.setattr(command, "load_depth_charts", lambda seasons, loader=None: pd.DataFrame({"ignored": []}))
    monkeypatch.setattr(command, "_inference_depth", lambda value: value)

    def fake_generate(loaded, qbs, schedules, depth, **kwargs):
        seen["qbs"] = qbs.copy(deep=True)
        seen["kwargs"] = kwargs
        return _diagnostic_report()

    monkeypatch.setattr(command, "generate_qb_passing_yards_weekly_projections", fake_generate)
    return artifact, seen


def test_generation_filters_target_and_future_outcomes_and_dry_run_writes_nothing(monkeypatch, tmp_path):
    artifact, seen = _patch_generation(monkeypatch, tmp_path)
    snapshot = tmp_path / "current.json"
    result = command.generate_qb_passing_yards_weekly_snapshot(
        season=2026, week=5, as_of_utc="2026-10-06T14:00:00Z",
        artifact_path=artifact, snapshot_path=snapshot, dry_run=True,
    )
    assert result.dry_run is True
    assert result.snapshot_written is False
    assert not snapshot.exists()
    assert seen["qbs"][["season", "week"]].values.tolist() == [[2025, 18], [2026, 4]]
    assert seen["kwargs"] == {
        "target_season": 2026, "target_week": 5,
        "as_of_utc": pd.Timestamp("2026-10-06T14:00:00+00:00"),
    }
    assert result.projections == 1
    assert result.selected_candidates == 1
    assert result.unknown_team_games == 1


def test_generation_uses_snapshot_api_and_requires_explicit_overwrite(monkeypatch, tmp_path):
    artifact, _ = _patch_generation(monkeypatch, tmp_path)
    destination = tmp_path / "current.json"
    calls = []
    monkeypatch.setattr(
        command, "save_qb_passing_yards_weekly_projection_snapshot",
        lambda report, path, identity, *, overwrite: calls.append((report, path, identity, overwrite)),
    )
    result = command.generate_qb_passing_yards_weekly_snapshot(
        season=2026, week=5, as_of_utc="2026-10-06T14:00:00Z",
        artifact_path=artifact, snapshot_path=destination, overwrite_snapshot=True,
    )
    assert result.snapshot_written is True
    assert len(calls) == 1
    assert calls[0][1] == destination
    assert calls[0][3] is True
    assert calls[0][2].artifact_sha256 == command._sha256_file(artifact)


@pytest.mark.parametrize("value", ["2026-10-06T14:00:00", None])
def test_generation_requires_timezone_aware_as_of_before_loading(monkeypatch, tmp_path, value):
    calls = []
    monkeypatch.setattr(command, "load_qb_passing_yards_production_model", lambda _: calls.append(True))
    with pytest.raises(ValueError, match="timezone-aware"):
        command.generate_qb_passing_yards_weekly_snapshot(
            season=2026, week=5, as_of_utc=value,
            artifact_path=tmp_path / "missing.joblib",
        )
    assert not calls


def test_preparation_uses_only_2020_to_2025_sources_and_never_requires_test(monkeypatch, tmp_path):
    calls = []
    dataset = pd.DataFrame({
        "season": [2024, 2025, 2026], "week": [18, 1, 1],
        "game_id": ["a", "b", "c"], "player_id": ["x", "y", "z"],
    })
    monkeypatch.setattr(command, "load_player_game_stats", lambda seasons, loader=None: calls.append(("players", seasons)) or pd.DataFrame())
    monkeypatch.setattr(command, "load_schedules", lambda seasons, loader=None: calls.append(("schedule", seasons)) or pd.DataFrame())
    monkeypatch.setattr(command, "normalize_quarterback_game_stats", lambda frame: frame)
    monkeypatch.setattr(command, "normalize_schedule_dataset", lambda frame: frame)
    monkeypatch.setattr(command, "build_qb_passing_yards_training_dataset", lambda qbs, schedules: dataset.copy())
    observed = {}
    monkeypatch.setattr(command, "train_qb_passing_yards_production_model", lambda split: observed.setdefault("split", split) or object())
    monkeypatch.setattr(command, "save_qb_passing_yards_production_model", lambda model, path, *, overwrite: (model, Path(path), overwrite))
    _, path, overwrite = command.prepare_qb_passing_yards_production_artifact(
        artifact_path=tmp_path / "production.joblib", overwrite=True
    )
    assert calls == [
        ("players", list(command.SOURCE_TRAINING_SEASONS)),
        ("schedule", list(command.SOURCE_TRAINING_SEASONS)),
    ]
    split = observed["split"]
    assert split.train[["season", "week"]].values.tolist() == [[2024, 18]]
    assert split.validation[["season", "week"]].values.tolist() == [[2025, 1]]
    assert split.test.empty
    assert path.name == "production.joblib"
    assert overwrite is True


def test_cli_requires_explicit_preparation_or_complete_generation_arguments(monkeypatch):
    with pytest.raises(SystemExit, match="2"):
        command.main([])
    prepared = []
    monkeypatch.setattr(command, "prepare_qb_passing_yards_production_artifact", lambda **kwargs: prepared.append(kwargs))
    assert command.main(["--prepare-artifact"]) == 0
    assert prepared
    with pytest.raises(SystemExit, match="2"):
        command.main(["--season", "2026", "--week", "5"])
    with pytest.raises(SystemExit, match="2"):
        command.main(["--overwrite-artifact"])


def test_schedule_normalization_localizes_established_eastern_schedule_clocks():
    raw = pd.DataFrame({
        "season": [2026], "week": [5], "game_id": ["g"], "game_type": ["REG"],
        "gameday": ["2026-10-08"], "gametime": ["20:15"],
        "home_team": ["AAA"], "away_team": ["BBB"],
        "home_score": [None], "away_score": [None],
    })
    result = command._inference_schedule(raw)
    assert result["scheduled_kickoff"].iloc[0] == pd.Timestamp("2026-10-09T00:15:00Z")
    assert set(result["season_type"]) == {"REG"}
