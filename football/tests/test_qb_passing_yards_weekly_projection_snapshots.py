"""Focused JSON persistence tests for unscored weekly QB projections."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
import math

import pandas as pd
import pytest

import football.pipeline.qb_passing_yards_weekly_projection_snapshots as snapshots
from football.pipeline.qb_passing_yards_weekly_inference import (
    QBPassingYardsWeeklyFeatureRows,
    QBPassingYardsWeeklyProjection,
    QBPassingYardsWeeklyProjectionReport,
    QBPassingYardsWeeklySkip,
)
from football.pipeline.qb_passing_yards_weekly_projection_snapshots import (
    QBPassingYardsWeeklyProjectionSnapshotArtifact,
    load_qb_passing_yards_weekly_projection_snapshot,
    save_qb_passing_yards_weekly_projection_snapshot,
)
from football.tests.test_qb_passing_yards_model_comparison import _split
from football.training.qb_passing_yards_production_model import (
    train_qb_passing_yards_production_model,
)
from football.training.split_qb_passing_yards_dataset import QBPassingYardsDatasetSplit


_SHA = "a" * 64


class _ExplodingTest:
    def __getattribute__(self, name):
        raise AssertionError("test partition accessed")


def _artifact() -> QBPassingYardsWeeklyProjectionSnapshotArtifact:
    source = _split()
    production = train_qb_passing_yards_production_model(
        QBPassingYardsDatasetSplit(source.train, source.validation, _ExplodingTest()),
        created_at_utc="2026-01-01T00:00:00+00:00",
    )
    return QBPassingYardsWeeklyProjectionSnapshotArtifact(
        "qb-passing-yards.joblib", _SHA, production.metadata,
    )


def _report() -> QBPassingYardsWeeklyProjectionReport:
    diagnostics = QBPassingYardsWeeklyFeatureRows(
        rows=(),
        skips=(
            QBPassingYardsWeeklySkip("g2", "CCC", "kickoff_not_strictly_after_as_of"),
            QBPassingYardsWeeklySkip("g2", "DDD", "kickoff_not_strictly_after_as_of"),
        ),
        target_season=2026,
        target_week=5,
        as_of_utc="2026-10-06T14:00:00+00:00",
        scheduled_games_considered=2,
        eligible_games=1,
        team_games_considered=2,
        likely_primary_selections=2,
        unknown_team_games=0,
        cold_start_projections=1,
        defense_missing_projections=0,
    )
    # Deliberately reversed so save must canonicalize identity ordering.
    projections = (
        QBPassingYardsWeeklyProjection(2026, 5, "g1", "2026-10-09T00:15:00+00:00", "qb-b", "B", "BBB", "AAA", "away", 240.5, True, False, 1, "qb-passing-yards.joblib", "2026-10-06T14:00:00+00:00"),
        QBPassingYardsWeeklyProjection(2026, 5, "g1", "2026-10-09T00:15:00+00:00", "qb-a", "A", "AAA", "BBB", "home", 260.25, False, False, 1, "qb-passing-yards.joblib", "2026-10-06T14:00:00+00:00"),
    )
    return QBPassingYardsWeeklyProjectionReport(projections, diagnostics)


def _save(tmp_path, report=None, artifact=None, name="snapshot.json"):
    return save_qb_passing_yards_weekly_projection_snapshot(
        report or _report(), tmp_path / name, artifact or _artifact(),
        created_at_utc="2026-10-06T14:01:00+00:00",
    )


def test_deterministic_atomic_round_trip_and_immutable_output(tmp_path):
    report = _report()
    before = repr(report)
    saved = _save(tmp_path, report, name="one.json")
    _save(tmp_path, report, name="two.json")
    assert (tmp_path / "one.json").read_bytes() == (tmp_path / "two.json").read_bytes()
    loaded = load_qb_passing_yards_weekly_projection_snapshot(tmp_path / "one.json", expected_artifact_sha256=_SHA)
    assert loaded == saved
    assert [value.player_id for value in loaded.projections] == ["qb-a", "qb-b"]
    assert repr(report) == before
    with pytest.raises(FrozenInstanceError):
        loaded.metadata.week = 6
    assert not list(tmp_path.glob(".*.tmp"))


def test_saved_json_contains_safe_projection_fields_only(tmp_path):
    _save(tmp_path)
    payload = json.loads((tmp_path / "snapshot.json").read_text(encoding="utf-8"))
    text = json.dumps(payload, sort_keys=True)
    for forbidden in ("target_passing_yards", "actual_passing_yards", "residual", "sportsbook", "feature_values", "pipeline", "estimator"):
        assert forbidden not in text
    assert payload["metadata"]["feature_columns"]
    assert payload["projections"][0]["predicted_passing_yards"] == pytest.approx(260.25)


def test_rejects_duplicate_nonfinite_timestamp_and_count_inconsistent_reports(tmp_path):
    duplicate = _report()
    projection = duplicate.projections[0]
    with pytest.raises(ValueError, match="duplicate"):
        _save(tmp_path, replace(duplicate, projections=(projection, projection)))
    nonfinite = replace(_report(), projections=(replace(_report().projections[0], predicted_passing_yards=math.inf), *_report().projections[1:]))
    with pytest.raises(ValueError, match="finite"):
        _save(tmp_path, nonfinite)
    invalid_timestamp = replace(_report(), projections=(replace(_report().projections[0], scheduled_kickoff_utc="2026-10-09"), *_report().projections[1:]))
    with pytest.raises(ValueError, match="timezone-aware"):
        _save(tmp_path, invalid_timestamp)
    bad_diagnostics = replace(_report(), feature_diagnostics=replace(_report().feature_diagnostics, likely_primary_selections=1))
    with pytest.raises(ValueError, match="selected-candidate"):
        _save(tmp_path, bad_diagnostics)


def test_load_rejects_schema_contract_artifact_and_malformed_json(tmp_path):
    _save(tmp_path)
    path = tmp_path / "snapshot.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["metadata"]["snapshot_schema_version"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_qb_passing_yards_weekly_projection_snapshot(path)
    _save(tmp_path, name="contract.json")
    contract = tmp_path / "contract.json"
    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["metadata"]["feature_columns"] = list(reversed(payload["metadata"]["feature_columns"]))
    contract.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="feature contract"):
        load_qb_passing_yards_weekly_projection_snapshot(contract)
    _save(tmp_path, name="identity.json")
    with pytest.raises(ValueError, match="identity"):
        load_qb_passing_yards_weekly_projection_snapshot(tmp_path / "identity.json", expected_artifact_sha256="b" * 64)
    malformed = tmp_path / "malformed.json"
    malformed.write_text('{"metadata":', encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        load_qb_passing_yards_weekly_projection_snapshot(malformed)


def test_save_rejects_report_artifact_identity_mismatch(tmp_path):
    report = _report()
    mismatched = replace(report, projections=(
        replace(report.projections[0], artifact_identifier="different.joblib"),
        *report.projections[1:],
    ))
    with pytest.raises(ValueError, match="artifact identity"):
        _save(tmp_path, mismatched)


def test_failed_atomic_replacement_preserves_destination_and_cleans_temporary(tmp_path, monkeypatch):
    path = tmp_path / "snapshot.json"
    path.write_text("old", encoding="utf-8")
    original = snapshots.os.replace

    def fail_replace(source, destination):
        raise OSError("replacement failed")

    monkeypatch.setattr(snapshots.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement failed"):
        save_qb_passing_yards_weekly_projection_snapshot(
            _report(), path, _artifact(), created_at_utc="2026-10-06T14:01:00+00:00", overwrite=True,
        )
    assert path.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".*.tmp"))
    monkeypatch.setattr(snapshots.os, "replace", original)


def test_load_is_json_only_and_snapshot_retains_no_frames_or_models(tmp_path, monkeypatch):
    _save(tmp_path)
    monkeypatch.setattr(snapshots, "_snapshot_from_report", lambda *args: pytest.fail("load attempted inference"))
    loaded = load_qb_passing_yards_weekly_projection_snapshot(tmp_path / "snapshot.json")
    values = [loaded, loaded.metadata, *loaded.projections, *loaded.skips]
    assert not any(isinstance(value, (pd.DataFrame, pd.Series)) for value in values)
    assert not any(hasattr(value, "fit") or hasattr(value, "predict") for value in values)
