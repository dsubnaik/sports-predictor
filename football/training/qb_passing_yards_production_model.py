"""Train, persist, load, and predict with the frozen QB production candidate.

Joblib artifacts are trusted-local files only.  The manifest checksum detects
unexpected changes relative to the manifest; it does not make untrusted pickle
or joblib data safe to deserialize.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import tempfile

import joblib
import numpy as np
import pandas as pd
import sklearn

from football.training.build_qb_passing_yards_dataset import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TARGET_KEY,
)
from football.training.qb_passing_yards_gradient_boosting import (
    GradientBoostingParameters,
    QBPassingYardsGradientBoostingModel,
    fit_qb_passing_yards_gradient_boosting,
    predict_qb_passing_yards_gradient_boosting,
)
from football.training.split_qb_passing_yards_dataset import QBPassingYardsDatasetSplit


ARTIFACT_SCHEMA_VERSION = 1
MODEL_FAMILY = "sklearn.ensemble.GradientBoostingRegressor"
HOLDOUT_BOUNDARY = (2026, 1)
MODEL_PURPOSE = "NFL QB passing-yards production candidate"


@dataclass(frozen=True)
class QBPassingYardsProductionMetadata:
    schema_version: int
    model_format_version: int
    model_purpose: str
    model_family: str
    estimator_parameters: GradientBoostingParameters
    feature_columns: tuple[str, ...]
    training_row_count: int
    minimum_training_period: tuple[int, int]
    maximum_training_period: tuple[int, int]
    training_target_seasons: tuple[int, ...]
    selection_validation_season: int
    exclusive_holdout_boundary: tuple[int, int]
    cold_start_definition: str
    preprocessing: str
    target_column: str
    target_key_columns: tuple[str, ...]
    created_at_utc: str
    python_version: str
    numpy_version: str
    pandas_version: str
    scikit_learn_version: str
    joblib_version: str


@dataclass(frozen=True)
class QBPassingYardsProductionModel:
    model: QBPassingYardsGradientBoostingModel
    metadata: QBPassingYardsProductionMetadata


@dataclass(frozen=True)
class QBPassingYardsSavedArtifact:
    artifact_name: str
    metadata_name: str
    byte_size: int
    sha256: str


@dataclass(frozen=True)
class QBPassingYardsLoadedArtifact:
    model: QBPassingYardsGradientBoostingModel
    metadata: QBPassingYardsProductionMetadata
    artifact_name: str


@dataclass(frozen=True)
class QBPassingYardsProductionPredictions:
    keys: tuple[tuple[object, ...], ...]
    predictions: tuple[float, ...]


def train_qb_passing_yards_production_model(
    dataset_split: QBPassingYardsDatasetSplit,
    *,
    created_at_utc: str | None = None,
) -> QBPassingYardsProductionModel:
    """Refit the frozen specification on train plus validation, never test."""
    if not isinstance(dataset_split, QBPassingYardsDatasetSplit):
        raise TypeError("dataset_split must be a QBPassingYardsDatasetSplit")
    train = dataset_split.train
    validation = dataset_split.validation
    _validate_partition(train, "train")
    _validate_partition(validation, "validation")
    if train.empty or validation.empty:
        raise ValueError("Production training requires nonempty train and validation partitions")
    train_keys = set(map(tuple, train[TARGET_KEY].itertuples(index=False, name=None)))
    validation_keys = set(map(tuple, validation[TARGET_KEY].itertuples(index=False, name=None)))
    if train_keys.intersection(validation_keys):
        raise ValueError("Production train and validation partitions overlap on target keys")
    combined = pd.concat([train, validation], ignore_index=True).copy(deep=True)
    canonical = combined.sort_values(TARGET_KEY, kind="mergesort").reset_index(drop=True)
    model = fit_qb_passing_yards_gradient_boosting(canonical)
    periods = tuple(map(tuple, canonical[["season", "week"]].itertuples(index=False, name=None)))
    metadata = QBPassingYardsProductionMetadata(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        model_format_version=ARTIFACT_SCHEMA_VERSION,
        model_purpose=MODEL_PURPOSE,
        model_family=MODEL_FAMILY,
        estimator_parameters=model.parameters,
        feature_columns=tuple(FEATURE_COLUMNS),
        training_row_count=len(canonical),
        minimum_training_period=min(periods),
        maximum_training_period=max(periods),
        training_target_seasons=tuple(sorted({int(value) for value in canonical["season"]})),
        selection_validation_season=2025,
        exclusive_holdout_boundary=HOLDOUT_BOUNDARY,
        cold_start_definition="qb_season_passing_yards_avg is missing",
        preprocessing="training-only median imputation; validated boolean pass-through; no scaling; no row dropping",
        target_column=TARGET_COLUMN,
        target_key_columns=tuple(TARGET_KEY),
        created_at_utc=created_at_utc or datetime.now(timezone.utc).isoformat(),
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        pandas_version=pd.__version__,
        scikit_learn_version=sklearn.__version__,
        joblib_version=version("joblib"),
    )
    return QBPassingYardsProductionModel(model, metadata)


def save_qb_passing_yards_production_model(
    production_model: QBPassingYardsProductionModel,
    artifact_path: str | Path,
    *,
    overwrite: bool = False,
) -> QBPassingYardsSavedArtifact:
    """Atomically save an explicit trusted-local joblib artifact and manifest."""
    _validate_production_model(production_model)
    path = _validate_artifact_path(artifact_path)
    metadata_path = _metadata_path(path)
    if (path.exists() or metadata_path.exists()) and not overwrite:
        raise FileExistsError("Artifact or metadata already exists; pass overwrite=True to replace")
    path.parent.mkdir(parents=True, exist_ok=True)
    model_temp = _temporary_path(path)
    manifest_temp = _temporary_path(metadata_path)
    try:
        joblib.dump(production_model.model, model_temp)
        _flush_file(model_temp)
        digest, size = _checksum(model_temp)
        manifest = _manifest(production_model.metadata, path.name, digest, size)
        _write_json(manifest_temp, manifest)
        os.replace(model_temp, path)
        os.replace(manifest_temp, metadata_path)
    finally:
        for temporary in (model_temp, manifest_temp):
            if temporary.exists():
                temporary.unlink()
    return QBPassingYardsSavedArtifact(path.name, metadata_path.name, size, digest)


def load_qb_passing_yards_production_model(
    artifact_path: str | Path,
) -> QBPassingYardsLoadedArtifact:
    """Validate the manifest and trusted-local artifact before deserializing."""
    path = _validate_artifact_path(artifact_path)
    metadata_path = _metadata_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact does not exist: {path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata manifest does not exist: {metadata_path}")
    try:
        manifest = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Production metadata manifest is malformed") from error
    metadata = _metadata_from_manifest(manifest, path.name)
    digest, size = _checksum(path)
    if size != manifest["artifact_byte_size"]:
        raise ValueError("Production artifact byte size does not match manifest")
    if digest != manifest["artifact_sha256"]:
        raise ValueError("Production artifact checksum does not match manifest")
    try:
        model = joblib.load(path)
    except Exception as error:  # trusted file may still be truncated/corrupt
        raise ValueError("Production artifact could not be deserialized") from error
    loaded = QBPassingYardsLoadedArtifact(model, metadata, path.name)
    _validate_loaded(loaded)
    return loaded


def predict_qb_passing_yards_production_model(
    loaded_artifact: QBPassingYardsLoadedArtifact,
    feature_frame: pd.DataFrame,
) -> QBPassingYardsProductionPredictions:
    """Predict a feature-only frame without targets, files, or network access."""
    _validate_loaded(loaded_artifact)
    if TARGET_COLUMN in feature_frame.columns:
        raise ValueError("Production prediction frame must not include target_passing_yards")
    _validate_prediction_frame(feature_frame)
    canonical = feature_frame.sort_values(TARGET_KEY, kind="mergesort").reset_index(drop=True)
    output = predict_qb_passing_yards_gradient_boosting(loaded_artifact.model, canonical)
    values = tuple(float(value) for value in output["gradient_boosting_prediction"])
    return QBPassingYardsProductionPredictions(
        keys=tuple(map(tuple, output[TARGET_KEY].itertuples(index=False, name=None))),
        predictions=values,
    )


def _validate_partition(frame: pd.DataFrame, name: str) -> None:
    required = [*TARGET_KEY, *FEATURE_COLUMNS, TARGET_COLUMN]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} partition is missing required columns: {missing}")
    if frame[TARGET_KEY].isna().any().any() or frame.duplicated(TARGET_KEY).any():
        raise ValueError(f"{name} partition has missing or duplicate target keys")
    for season, week in frame[["season", "week"]].itertuples(index=False, name=None):
        if not _valid_period(season, week):
            raise ValueError(f"{name} partition has invalid or holdout-period rows")


def _validate_prediction_frame(frame: pd.DataFrame) -> None:
    required = [*TARGET_KEY, *FEATURE_COLUMNS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Prediction frame is missing required columns: {missing}")
    if frame[TARGET_KEY].isna().any().any() or frame.duplicated(TARGET_KEY).any():
        raise ValueError("Prediction frame has missing or duplicate target keys")
    # Delegate established feature/flag validation to completed prediction API.


def _valid_period(season: object, week: object) -> bool:
    try:
        return (
            not isinstance(season, bool)
            and not isinstance(week, bool)
            and int(season) == season
            and int(week) == week
            and (int(season), int(week)) < HOLDOUT_BOUNDARY
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _validate_production_model(value: QBPassingYardsProductionModel) -> None:
    if not isinstance(value, QBPassingYardsProductionModel):
        raise TypeError("production_model must be a QBPassingYardsProductionModel")
    if (
        value.metadata.schema_version != ARTIFACT_SCHEMA_VERSION
        or value.metadata.model_format_version != ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError("Production model schema version is unsupported")
    if value.metadata.feature_columns != tuple(FEATURE_COLUMNS) or value.model.feature_columns != tuple(FEATURE_COLUMNS):
        raise ValueError("Production model feature contract does not match the canonical contract")
    if value.metadata.estimator_parameters != GradientBoostingParameters() or value.model.parameters != GradientBoostingParameters():
        raise ValueError("Production model parameters do not match the frozen specification")


def _validate_loaded(value: QBPassingYardsLoadedArtifact) -> None:
    if not isinstance(value.model, QBPassingYardsGradientBoostingModel):
        raise ValueError("Production artifact does not contain a Gradient Boosting model")
    _validate_production_model(QBPassingYardsProductionModel(value.model, value.metadata))
    if value.metadata.model_family != MODEL_FAMILY or value.metadata.exclusive_holdout_boundary != HOLDOUT_BOUNDARY:
        raise ValueError("Production metadata is incompatible with the frozen specification")
    if value.metadata.target_column != TARGET_COLUMN or value.metadata.target_key_columns != tuple(TARGET_KEY):
        raise ValueError("Production metadata target contract is incompatible")


def _validate_artifact_path(value: str | Path) -> Path:
    path = Path(value)
    if path.name in {"", "."} or path.suffix != ".joblib":
        raise ValueError("Artifact path must be an explicit .joblib file path")
    return path


def _metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".metadata.json")


def _temporary_path(path: Path) -> Path:
    handle = tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False)
    handle.close()
    return Path(handle.name)


def _flush_file(path: Path) -> None:
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _checksum(path: Path) -> tuple[str, int]:
    digest = sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _manifest(metadata: QBPassingYardsProductionMetadata, artifact_name: str, digest: str, size: int) -> dict[str, object]:
    result = asdict(metadata)
    result["estimator_parameters"] = asdict(metadata.estimator_parameters)
    result.update({"artifact_filename": artifact_name, "artifact_byte_size": size, "artifact_sha256": digest})
    return result


def _write_json(path: Path, data: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _metadata_from_manifest(manifest: object, artifact_name: str) -> QBPassingYardsProductionMetadata:
    if not isinstance(manifest, dict) or manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Production metadata has an unsupported schema version")
    if manifest.get("model_format_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Production metadata has an unsupported model format version")
    if manifest.get("artifact_filename") != artifact_name or not isinstance(manifest.get("artifact_sha256"), str):
        raise ValueError("Production metadata does not match artifact path")
    try:
        parameters = GradientBoostingParameters(**manifest["estimator_parameters"])
        return QBPassingYardsProductionMetadata(
            schema_version=manifest["schema_version"], model_format_version=manifest["model_format_version"], model_purpose=manifest["model_purpose"], model_family=manifest["model_family"], estimator_parameters=parameters,
            feature_columns=tuple(manifest["feature_columns"]), training_row_count=manifest["training_row_count"], minimum_training_period=tuple(manifest["minimum_training_period"]), maximum_training_period=tuple(manifest["maximum_training_period"]), training_target_seasons=tuple(manifest["training_target_seasons"]), selection_validation_season=manifest["selection_validation_season"], exclusive_holdout_boundary=tuple(manifest["exclusive_holdout_boundary"]), cold_start_definition=manifest["cold_start_definition"], preprocessing=manifest["preprocessing"], target_column=manifest["target_column"], target_key_columns=tuple(manifest["target_key_columns"]), created_at_utc=manifest["created_at_utc"], python_version=manifest["python_version"], numpy_version=manifest["numpy_version"], pandas_version=manifest["pandas_version"], scikit_learn_version=manifest["scikit_learn_version"], joblib_version=manifest["joblib_version"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Production metadata is incomplete or invalid") from error
