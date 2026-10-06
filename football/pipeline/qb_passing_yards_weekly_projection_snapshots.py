"""Atomic, validated JSON snapshots for unscored weekly QB projections.

The snapshot format is deliberately model-free.  It records safe display
values produced by the weekly inference pipeline plus enough immutable model
identity metadata for a consumer to reject an incompatible or stale snapshot.
It never deserializes a joblib artifact.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd

from football.pipeline.qb_passing_yards_weekly_inference import (
    QBPassingYardsWeeklyFeatureRows,
    QBPassingYardsWeeklyProjectionReport,
    QBPassingYardsWeeklySkip,
)
from football.training.build_qb_passing_yards_dataset import FEATURE_COLUMNS
from football.training.qb_passing_yards_gradient_boosting import GradientBoostingParameters
from football.training.qb_passing_yards_production_model import (
    ARTIFACT_SCHEMA_VERSION,
    HOLDOUT_BOUNDARY,
    MODEL_FAMILY,
    QBPassingYardsProductionMetadata,
)


SNAPSHOT_SCHEMA_VERSION = 1
_SHA256_LENGTH = 64
_IDENTITY_FIELDS = ("season", "week", "game_id", "player_id")


@dataclass(frozen=True)
class QBPassingYardsWeeklyProjectionSnapshotArtifact:
    """Trusted production-artifact identity supplied by the snapshot creator."""

    artifact_identifier: str
    artifact_sha256: str
    production_metadata: QBPassingYardsProductionMetadata


@dataclass(frozen=True)
class QBPassingYardsWeeklyProjectionSnapshotSkip:
    game_id: str
    team: str | None
    reason: str


@dataclass(frozen=True)
class QBPassingYardsWeeklyProjectionSnapshotProjection:
    season: int
    week: int
    game_id: str
    scheduled_kickoff_utc: str
    player_id: str
    player_name: str | None
    team: str
    opponent: str
    home_away: str
    predicted_passing_yards: float
    qb_cold_start: bool
    defense_missing_history: bool


@dataclass(frozen=True)
class QBPassingYardsWeeklyProjectionSnapshotMetadata:
    snapshot_schema_version: int
    model_schema_version: int
    model_format_version: int
    model_family: str
    artifact_identifier: str
    artifact_sha256: str
    feature_columns: tuple[str, ...]
    feature_contract_sha256: str
    training_row_count: int
    minimum_training_period: tuple[int, int]
    maximum_training_period: tuple[int, int]
    training_target_seasons: tuple[int, ...]
    exclusive_holdout_boundary: tuple[int, int]
    season: int
    week: int
    as_of_utc: str
    created_at_utc: str
    scheduled_games_considered: int
    eligible_games: int
    team_games_considered: int
    selected_candidates: int
    unknown_or_unselected_team_games: int
    projection_count: int
    cold_start_projections: int
    defense_missing_projections: int
    skip_count: int


@dataclass(frozen=True)
class QBPassingYardsWeeklyProjectionSnapshot:
    metadata: QBPassingYardsWeeklyProjectionSnapshotMetadata
    skips: tuple[QBPassingYardsWeeklyProjectionSnapshotSkip, ...]
    projections: tuple[QBPassingYardsWeeklyProjectionSnapshotProjection, ...]


def save_qb_passing_yards_weekly_projection_snapshot(
    report: QBPassingYardsWeeklyProjectionReport,
    path: str | Path,
    artifact: QBPassingYardsWeeklyProjectionSnapshotArtifact,
    *,
    created_at_utc: str | None = None,
    overwrite: bool = False,
) -> QBPassingYardsWeeklyProjectionSnapshot:
    """Validate and atomically write one explicit unscored projection snapshot."""
    snapshot = _snapshot_from_report(report, artifact, created_at_utc)
    destination = _validate_snapshot_path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError("Projection snapshot already exists; pass overwrite=True to replace")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(destination)
    try:
        _write_json(temporary, _snapshot_payload(snapshot))
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return snapshot


def load_qb_passing_yards_weekly_projection_snapshot(
    path: str | Path,
    *,
    expected_artifact_sha256: str | None = None,
) -> QBPassingYardsWeeklyProjectionSnapshot:
    """Load only validated JSON; no model is loaded, fitted, or executed.

    A checksum validates identity against a supplied manifest value, but does
    not make arbitrary joblib/pickle files safe to deserialize.  Snapshot
    loading never deserializes those files in the first place.
    """
    source = _validate_snapshot_path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Projection snapshot does not exist: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Projection snapshot JSON is malformed") from error
    snapshot = _snapshot_from_payload(payload)
    if expected_artifact_sha256 is not None:
        expected = _sha256_text(expected_artifact_sha256, "expected_artifact_sha256")
        if snapshot.metadata.artifact_sha256 != expected:
            raise ValueError("Projection snapshot artifact identity does not match the expected checksum")
    return snapshot


def _snapshot_from_report(
    report: QBPassingYardsWeeklyProjectionReport,
    artifact: QBPassingYardsWeeklyProjectionSnapshotArtifact,
    created_at_utc: str | None,
) -> QBPassingYardsWeeklyProjectionSnapshot:
    if not isinstance(report, QBPassingYardsWeeklyProjectionReport):
        raise TypeError("report must be a QBPassingYardsWeeklyProjectionReport")
    _validate_artifact(artifact)
    diagnostics = report.feature_diagnostics
    if not isinstance(diagnostics, QBPassingYardsWeeklyFeatureRows):
        raise TypeError("projection report has invalid feature diagnostics")
    projections = tuple(
        QBPassingYardsWeeklyProjectionSnapshotProjection(
            season=_integer(value.season, "projection season", minimum=1),
            week=_integer(value.week, "projection week", minimum=1),
            game_id=_text(value.game_id, "projection game_id"),
            scheduled_kickoff_utc=_utc(value.scheduled_kickoff_utc, "projection scheduled_kickoff_utc"),
            player_id=_text(value.player_id, "projection player_id"),
            player_name=_optional_text(value.player_name, "projection player_name"),
            team=_text(value.team, "projection team"),
            opponent=_text(value.opponent, "projection opponent"),
            home_away=_home_away(value.home_away),
            predicted_passing_yards=_finite_float(value.predicted_passing_yards, "predicted_passing_yards"),
            qb_cold_start=_boolean(value.qb_cold_start, "qb_cold_start"),
            defense_missing_history=_boolean(value.defense_missing_history, "defense_missing_history"),
        )
        for value in report.projections
    )
    if any(value.model_schema_version != artifact.production_metadata.schema_version for value in report.projections):
        raise ValueError("Projection report model schema does not match the artifact metadata")
    if any(value.artifact_identifier != artifact.artifact_identifier for value in report.projections):
        raise ValueError("Projection report artifact identity does not match the supplied artifact")
    skips = tuple(
        QBPassingYardsWeeklyProjectionSnapshotSkip(
            game_id=_text(value.game_id, "skip game_id"),
            team=_optional_text(value.team, "skip team"),
            reason=_text(value.reason, "skip reason"),
        )
        for value in diagnostics.skips
    )
    projections = tuple(sorted(projections, key=_projection_key))
    skips = tuple(sorted(skips, key=lambda value: (value.game_id, value.team or "", value.reason)))
    _validate_projection_keys(projections)
    _validate_skip_keys(skips)
    season_week = {(value.season, value.week) for value in projections}
    if len(season_week) > 1:
        raise ValueError("Projection report contains more than one season/week")
    season = _integer(diagnostics.target_season, "target_season", minimum=1)
    week = _integer(diagnostics.target_week, "target_week", minimum=1)
    if season_week and season_week != {(season, week)}:
        raise ValueError("Projection report rows do not match diagnostic season/week")
    as_of_values = {_utc(value.as_of_utc, "projection as_of_utc") for value in report.projections}
    if len(as_of_values) > 1:
        raise ValueError("Projection report has inconsistent as-of timestamps")
    diagnostic_as_of = _utc(diagnostics.as_of_utc, "diagnostic as_of_utc")
    as_of = next(iter(as_of_values), diagnostic_as_of)
    if as_of != diagnostic_as_of:
        raise ValueError("Projection report timestamps do not match diagnostics")
    metadata = QBPassingYardsWeeklyProjectionSnapshotMetadata(
        snapshot_schema_version=SNAPSHOT_SCHEMA_VERSION,
        model_schema_version=artifact.production_metadata.schema_version,
        model_format_version=artifact.production_metadata.model_format_version,
        model_family=artifact.production_metadata.model_family,
        artifact_identifier=artifact.artifact_identifier,
        artifact_sha256=artifact.artifact_sha256,
        feature_columns=tuple(FEATURE_COLUMNS),
        feature_contract_sha256=_feature_contract_sha256(),
        training_row_count=artifact.production_metadata.training_row_count,
        minimum_training_period=artifact.production_metadata.minimum_training_period,
        maximum_training_period=artifact.production_metadata.maximum_training_period,
        training_target_seasons=artifact.production_metadata.training_target_seasons,
        exclusive_holdout_boundary=artifact.production_metadata.exclusive_holdout_boundary,
        season=season,
        week=week,
        as_of_utc=as_of,
        created_at_utc=_utc(created_at_utc or datetime.now(timezone.utc).isoformat(), "created_at_utc"),
        scheduled_games_considered=_integer(diagnostics.scheduled_games_considered, "scheduled_games_considered", minimum=0),
        eligible_games=_integer(diagnostics.eligible_games, "eligible_games", minimum=0),
        team_games_considered=_integer(diagnostics.team_games_considered, "team_games_considered", minimum=0),
        selected_candidates=_integer(diagnostics.likely_primary_selections, "selected_candidates", minimum=0),
        unknown_or_unselected_team_games=_integer(diagnostics.unknown_team_games, "unknown_team_games", minimum=0),
        projection_count=len(projections),
        cold_start_projections=sum(value.qb_cold_start for value in projections),
        defense_missing_projections=sum(value.defense_missing_history for value in projections),
        skip_count=len(skips),
    )
    snapshot = QBPassingYardsWeeklyProjectionSnapshot(metadata, skips, projections)
    _validate_snapshot(snapshot)
    return snapshot


def _snapshot_payload(snapshot: QBPassingYardsWeeklyProjectionSnapshot) -> dict[str, object]:
    return {
        "metadata": _json_value(asdict(snapshot.metadata)),
        "projections": [_json_value(asdict(value)) for value in snapshot.projections],
        "skips": [_json_value(asdict(value)) for value in snapshot.skips],
    }


def _snapshot_from_payload(payload: object) -> QBPassingYardsWeeklyProjectionSnapshot:
    if not isinstance(payload, dict) or set(payload) != {"metadata", "projections", "skips"}:
        raise ValueError("Projection snapshot has an invalid top-level schema")
    metadata_raw, projections_raw, skips_raw = payload["metadata"], payload["projections"], payload["skips"]
    if not isinstance(metadata_raw, dict) or not isinstance(projections_raw, list) or not isinstance(skips_raw, list):
        raise ValueError("Projection snapshot has invalid JSON value types")
    try:
        metadata = QBPassingYardsWeeklyProjectionSnapshotMetadata(
            snapshot_schema_version=metadata_raw["snapshot_schema_version"], model_schema_version=metadata_raw["model_schema_version"], model_format_version=metadata_raw["model_format_version"], model_family=metadata_raw["model_family"], artifact_identifier=metadata_raw["artifact_identifier"], artifact_sha256=metadata_raw["artifact_sha256"], feature_columns=tuple(metadata_raw["feature_columns"]), feature_contract_sha256=metadata_raw["feature_contract_sha256"], training_row_count=metadata_raw["training_row_count"], minimum_training_period=tuple(metadata_raw["minimum_training_period"]), maximum_training_period=tuple(metadata_raw["maximum_training_period"]), training_target_seasons=tuple(metadata_raw["training_target_seasons"]), exclusive_holdout_boundary=tuple(metadata_raw["exclusive_holdout_boundary"]), season=metadata_raw["season"], week=metadata_raw["week"], as_of_utc=metadata_raw["as_of_utc"], created_at_utc=metadata_raw["created_at_utc"], scheduled_games_considered=metadata_raw["scheduled_games_considered"], eligible_games=metadata_raw["eligible_games"], team_games_considered=metadata_raw["team_games_considered"], selected_candidates=metadata_raw["selected_candidates"], unknown_or_unselected_team_games=metadata_raw["unknown_or_unselected_team_games"], projection_count=metadata_raw["projection_count"], cold_start_projections=metadata_raw["cold_start_projections"], defense_missing_projections=metadata_raw["defense_missing_projections"], skip_count=metadata_raw["skip_count"],
        )
        projections = tuple(QBPassingYardsWeeklyProjectionSnapshotProjection(**value) for value in projections_raw)
        skips = tuple(QBPassingYardsWeeklyProjectionSnapshotSkip(**value) for value in skips_raw)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Projection snapshot is incomplete or malformed") from error
    snapshot = QBPassingYardsWeeklyProjectionSnapshot(metadata, skips, projections)
    _validate_snapshot(snapshot)
    return snapshot


def _validate_snapshot(snapshot: QBPassingYardsWeeklyProjectionSnapshot) -> None:
    metadata = snapshot.metadata
    if metadata.snapshot_schema_version != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("Projection snapshot schema version is unsupported")
    _validate_snapshot_metadata(metadata)
    if snapshot.projections != tuple(sorted(snapshot.projections, key=_projection_key)):
        raise ValueError("Projection snapshot projections are not in canonical order")
    if snapshot.skips != tuple(sorted(snapshot.skips, key=lambda value: (value.game_id, value.team or "", value.reason))):
        raise ValueError("Projection snapshot skips are not in canonical order")
    _validate_projection_keys(snapshot.projections)
    _validate_skip_keys(snapshot.skips)
    if any((value.season, value.week) != (metadata.season, metadata.week) for value in snapshot.projections):
        raise ValueError("Projection snapshot rows do not match metadata season/week")
    if len(snapshot.projections) != metadata.projection_count:
        raise ValueError("Projection snapshot projection count does not reconcile")
    if len(snapshot.skips) != metadata.skip_count:
        raise ValueError("Projection snapshot skip count does not reconcile")
    if sum(value.qb_cold_start for value in snapshot.projections) != metadata.cold_start_projections:
        raise ValueError("Projection snapshot cold-start count does not reconcile")
    if sum(value.defense_missing_history for value in snapshot.projections) != metadata.defense_missing_projections:
        raise ValueError("Projection snapshot defense-missing count does not reconcile")
    if metadata.projection_count != metadata.selected_candidates:
        raise ValueError("Projection snapshot selected-candidate count does not reconcile")
    if metadata.team_games_considered != metadata.selected_candidates + metadata.unknown_or_unselected_team_games:
        raise ValueError("Projection snapshot team-game counts do not reconcile")
    if metadata.eligible_games > metadata.scheduled_games_considered:
        raise ValueError("Projection snapshot eligible-game count is invalid")
    if metadata.team_games_considered != 2 * metadata.eligible_games:
        raise ValueError("Projection snapshot eligible team-game count does not reconcile")
    if metadata.skip_count != 2 * (metadata.scheduled_games_considered - metadata.eligible_games):
        raise ValueError("Projection snapshot skip count is inconsistent with scheduled games")


def _validate_snapshot_metadata(metadata: QBPassingYardsWeeklyProjectionSnapshotMetadata) -> None:
    if metadata.model_schema_version != ARTIFACT_SCHEMA_VERSION or metadata.model_format_version != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Projection snapshot model schema is unsupported")
    if metadata.model_family != MODEL_FAMILY:
        raise ValueError("Projection snapshot model family is incompatible")
    if metadata.feature_columns != tuple(FEATURE_COLUMNS) or metadata.feature_contract_sha256 != _feature_contract_sha256():
        raise ValueError("Projection snapshot feature contract is incompatible")
    _sha256_text(metadata.artifact_sha256, "artifact_sha256")
    _text(metadata.artifact_identifier, "artifact_identifier")
    for label, value in (("season", metadata.season), ("week", metadata.week), ("training_row_count", metadata.training_row_count), ("scheduled_games_considered", metadata.scheduled_games_considered), ("eligible_games", metadata.eligible_games), ("team_games_considered", metadata.team_games_considered), ("selected_candidates", metadata.selected_candidates), ("unknown_or_unselected_team_games", metadata.unknown_or_unselected_team_games), ("projection_count", metadata.projection_count), ("cold_start_projections", metadata.cold_start_projections), ("defense_missing_projections", metadata.defense_missing_projections), ("skip_count", metadata.skip_count)):
        _integer(value, label, minimum=0 if label != "season" and label != "week" else 1)
    _period(metadata.minimum_training_period, "minimum_training_period")
    _period(metadata.maximum_training_period, "maximum_training_period")
    if metadata.minimum_training_period > metadata.maximum_training_period:
        raise ValueError("Projection snapshot training period bounds are invalid")
    if metadata.exclusive_holdout_boundary != HOLDOUT_BOUNDARY:
        raise ValueError("Projection snapshot holdout boundary is incompatible")
    if not metadata.training_target_seasons or any(_integer(value, "training_target_season", minimum=1) >= HOLDOUT_BOUNDARY[0] for value in metadata.training_target_seasons):
        raise ValueError("Projection snapshot training target seasons are invalid")
    _utc(metadata.as_of_utc, "as_of_utc")
    _utc(metadata.created_at_utc, "created_at_utc")


def _validate_artifact(artifact: QBPassingYardsWeeklyProjectionSnapshotArtifact) -> None:
    if not isinstance(artifact, QBPassingYardsWeeklyProjectionSnapshotArtifact):
        raise TypeError("artifact must be a QBPassingYardsWeeklyProjectionSnapshotArtifact")
    _text(artifact.artifact_identifier, "artifact_identifier")
    _sha256_text(artifact.artifact_sha256, "artifact_sha256")
    metadata = artifact.production_metadata
    if not isinstance(metadata, QBPassingYardsProductionMetadata):
        raise TypeError("artifact production_metadata is invalid")
    if metadata.schema_version != ARTIFACT_SCHEMA_VERSION or metadata.model_format_version != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Artifact metadata schema is unsupported")
    if metadata.model_family != MODEL_FAMILY or metadata.feature_columns != tuple(FEATURE_COLUMNS):
        raise ValueError("Artifact metadata feature contract is incompatible")
    if metadata.estimator_parameters != GradientBoostingParameters():
        raise ValueError("Artifact metadata parameters are incompatible")
    if metadata.exclusive_holdout_boundary != HOLDOUT_BOUNDARY:
        raise ValueError("Artifact metadata holdout boundary is incompatible")


def _validate_projection_keys(values: tuple[QBPassingYardsWeeklyProjectionSnapshotProjection, ...]) -> None:
    keys = tuple(_projection_key(value) for value in values)
    if len(set(keys)) != len(keys):
        raise ValueError("Projection snapshot has duplicate canonical projection keys")
    for value in values:
        _utc(value.scheduled_kickoff_utc, "projection scheduled_kickoff_utc")
        _finite_float(value.predicted_passing_yards, "predicted_passing_yards")
        _boolean(value.qb_cold_start, "qb_cold_start")
        _boolean(value.defense_missing_history, "defense_missing_history")


def _validate_skip_keys(values: tuple[QBPassingYardsWeeklyProjectionSnapshotSkip, ...]) -> None:
    for value in values:
        _text(value.game_id, "skip game_id")
        _optional_text(value.team, "skip team")
        _text(value.reason, "skip reason")


def _projection_key(value: QBPassingYardsWeeklyProjectionSnapshotProjection) -> tuple[int, int, str, str]:
    return value.season, value.week, value.game_id, value.player_id


def _feature_contract_sha256() -> str:
    return sha256(json.dumps(list(FEATURE_COLUMNS), ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _validate_snapshot_path(value: str | Path) -> Path:
    path = Path(value)
    if path.name in {"", "."} or path.suffix != ".json":
        raise ValueError("Projection snapshot path must be an explicit .json file path")
    return path


def _temporary_path(path: Path) -> Path:
    handle = tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False)
    handle.close()
    return Path(handle.name)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


def _utc(value: object, label: str) -> str:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a timezone-aware timestamp") from error
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"{label} must be a timezone-aware timestamp")
    return timestamp.tz_convert("UTC").isoformat()


def _integer(value: object, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer at least {minimum}")
    return value


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _text(value, label)


def _home_away(value: object) -> str:
    if value not in {"home", "away"}:
        raise ValueError("projection home_away must be home or away")
    return str(value)


def _period(value: object, label: str) -> tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{label} must be a season/week pair")
    return _integer(value[0], label, minimum=1), _integer(value[1], label, minimum=1)


def _sha256_text(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return value
