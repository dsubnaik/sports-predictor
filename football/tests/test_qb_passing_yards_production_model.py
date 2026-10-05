from dataclasses import FrozenInstanceError
import json

import numpy as np
import pandas as pd
import pytest

from football.tests.test_qb_passing_yards_model_comparison import _split
from football.training.build_qb_passing_yards_dataset import FEATURE_COLUMNS, TARGET_COLUMN, TARGET_KEY
from football.training.qb_passing_yards_gradient_boosting import fit_qb_passing_yards_gradient_boosting
from football.training.qb_passing_yards_production_model import (
    load_qb_passing_yards_production_model,
    predict_qb_passing_yards_production_model,
    save_qb_passing_yards_production_model,
    train_qb_passing_yards_production_model,
)
from football.training.split_qb_passing_yards_dataset import QBPassingYardsDatasetSplit


class _ExplodingTest:
    def __getattribute__(self, name):
        raise AssertionError("test partition accessed")


def _production_split():
    source = _split()
    return QBPassingYardsDatasetSplit(source.train, source.validation, _ExplodingTest())


def _features(frame):
    return frame.loc[:, [*TARGET_KEY, *FEATURE_COLUMNS]].copy(deep=True)


def test_training_combines_only_train_validation_and_matches_completed_api():
    split = _production_split()
    train_before, validation_before = split.train.copy(deep=True), split.validation.copy(deep=True)
    result = train_qb_passing_yards_production_model(split, created_at_utc="2026-01-01T00:00:00+00:00")
    combined = pd.concat([split.train, split.validation], ignore_index=True).sort_values(TARGET_KEY, kind="mergesort")
    direct = fit_qb_passing_yards_gradient_boosting(combined)
    assert result.metadata.training_row_count == len(combined)
    assert result.metadata.feature_columns == tuple(FEATURE_COLUMNS)
    assert result.model.parameters == direct.parameters
    assert result.metadata.exclusive_holdout_boundary == (2026, 1)
    assert result.metadata.minimum_training_period < (2026, 1)
    assert result.metadata.maximum_training_period < (2026, 1)
    pd.testing.assert_frame_equal(split.train, train_before)
    pd.testing.assert_frame_equal(split.validation, validation_before)
    with pytest.raises(FrozenInstanceError):
        result.metadata.model_family = "other"


def test_training_rejects_overlap_empty_holdout_and_invalid_inputs():
    source = _split()
    overlap = QBPassingYardsDatasetSplit(source.train, source.train.iloc[:1], source.test)
    with pytest.raises(ValueError, match="overlap"):
        train_qb_passing_yards_production_model(overlap)
    empty = QBPassingYardsDatasetSplit(source.train.iloc[:0], source.validation, source.test)
    with pytest.raises(ValueError, match="nonempty"):
        train_qb_passing_yards_production_model(empty)
    invalid = source.validation.copy(deep=True)
    invalid.loc[invalid.index[0], "season"] = 2026
    with pytest.raises(ValueError, match="holdout"):
        train_qb_passing_yards_production_model(QBPassingYardsDatasetSplit(source.train, invalid, source.test))
    nonfinite = source.train.copy(deep=True)
    nonfinite.loc[nonfinite.index[0], TARGET_COLUMN] = np.inf
    with pytest.raises(ValueError):
        train_qb_passing_yards_production_model(QBPassingYardsDatasetSplit(nonfinite, source.validation, source.test))


def test_save_load_prediction_integrity_and_no_overwrite(tmp_path):
    production = train_qb_passing_yards_production_model(_production_split(), created_at_utc="2026-01-01T00:00:00+00:00")
    path = tmp_path / "nested" / "qb.joblib"
    saved = save_qb_passing_yards_production_model(production, path)
    assert saved.byte_size > 0 and len(saved.sha256) == 64
    manifest_path = path.with_suffix(".joblib.metadata.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifact_filename"] == "qb.joblib"
    assert manifest["feature_columns"] == list(FEATURE_COLUMNS)
    assert str(tmp_path) not in manifest_path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        save_qb_passing_yards_production_model(production, path)
    loaded = load_qb_passing_yards_production_model(path)
    frame = _features(_split().validation)
    expected = predict_qb_passing_yards_production_model(
        type("Loaded", (), {"model": production.model, "metadata": production.metadata, "artifact_name": "x"})(), frame
    )
    actual = predict_qb_passing_yards_production_model(loaded, frame.sample(frac=1, random_state=4))
    assert actual.keys == expected.keys
    assert actual.predictions == pytest.approx(expected.predictions, abs=1e-12)
    save_qb_passing_yards_production_model(production, path, overwrite=True)


def test_load_rejects_manifest_or_artifact_tampering_and_prediction_rejects_target(tmp_path):
    production = train_qb_passing_yards_production_model(_production_split())
    path = tmp_path / "qb.joblib"
    save_qb_passing_yards_production_model(production, path)
    manifest_path = path.with_suffix(".joblib.metadata.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_qb_passing_yards_production_model(path)
    save_qb_passing_yards_production_model(production, path, overwrite=True)
    loaded = load_qb_passing_yards_production_model(path)
    target_frame = _split().validation.copy(deep=True)
    with pytest.raises(ValueError, match="must not include"):
        predict_qb_passing_yards_production_model(loaded, target_frame)
    missing = _features(_split().validation).drop(columns=[FEATURE_COLUMNS[0]])
    with pytest.raises(ValueError, match="missing"):
        predict_qb_passing_yards_production_model(loaded, missing)
