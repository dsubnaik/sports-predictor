"""Pure presentation tests for saved QB projection snapshots."""
from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from football.pipeline.qb_passing_yards_weekly_projection_snapshots import (
    QBPassingYardsWeeklyProjectionSnapshot,
    QBPassingYardsWeeklyProjectionSnapshotMetadata,
    QBPassingYardsWeeklyProjectionSnapshotProjection,
)
from football.tests.test_qb_passing_yards_model_comparison import _split
from football.training.build_qb_passing_yards_dataset import FEATURE_COLUMNS
from football.training.qb_passing_yards_production_model import train_qb_passing_yards_production_model
from football.training.split_qb_passing_yards_dataset import QBPassingYardsDatasetSplit
from football.ui.qb_projection_view import (
    QBProjectionSnapshotLoadResult,
    build_qb_projection_presentation,
    load_qb_projection_snapshot_for_ui,
)


class _ExplodingTest:
    def __getattribute__(self, name):
        raise AssertionError("test partition accessed")


def _metadata() -> QBPassingYardsWeeklyProjectionSnapshotMetadata:
    source = _split()
    production = train_qb_passing_yards_production_model(
        QBPassingYardsDatasetSplit(source.train, source.validation, _ExplodingTest()),
        created_at_utc="2026-01-01T00:00:00+00:00",
    ).metadata
    import football.pipeline.qb_passing_yards_weekly_projection_snapshots as snapshots
    return QBPassingYardsWeeklyProjectionSnapshotMetadata(
        snapshot_schema_version=1, model_schema_version=production.schema_version,
        model_format_version=production.model_format_version, model_family=production.model_family,
        artifact_identifier="fixture.joblib", artifact_sha256="a" * 64,
        feature_columns=tuple(FEATURE_COLUMNS), feature_contract_sha256=snapshots._feature_contract_sha256(),
        training_row_count=production.training_row_count, minimum_training_period=production.minimum_training_period,
        maximum_training_period=production.maximum_training_period, training_target_seasons=production.training_target_seasons,
        exclusive_holdout_boundary=production.exclusive_holdout_boundary, season=2026, week=5,
        as_of_utc="2026-10-06T14:00:00+00:00", created_at_utc="2026-10-06T14:01:00+00:00",
        scheduled_games_considered=1, eligible_games=1, team_games_considered=2,
        selected_candidates=2, unknown_or_unselected_team_games=0, projection_count=2,
        cold_start_projections=0, defense_missing_projections=0, skip_count=0,
    )


def _snapshot(*, player_name="John Doe Jr.", player_id="qb-1", prediction=258.9) -> QBPassingYardsWeeklyProjectionSnapshot:
    first = QBPassingYardsWeeklyProjectionSnapshotProjection(
        2026, 5, "game-1", "2026-10-09T00:15:00+00:00", player_id,
        player_name, "AAA", "BBB", "home", prediction, False, False,
    )
    second = QBPassingYardsWeeklyProjectionSnapshotProjection(
        2026, 5, "game-1", "2026-10-09T00:15:00+00:00", "qb-2",
        "Other QB", "BBB", "AAA", "away", 240.0, False, False,
    )
    return QBPassingYardsWeeklyProjectionSnapshot(_metadata(), (), (first, second))


def _matchup(*, player_id="qb-1", player_name="John Doe Jr.", game_id="game-1", season=2026, week=5):
    return pd.Series({"season": season, "report_week": week, "game_id": game_id, "team": "AAA", "opponent": "BBB", "expected_player_id": player_id, "expected_player_name": player_name})


def _props(point=250.5):
    return pd.DataFrame([
        {"bookmaker_key": "book", "point": point},
        {"bookmaker_key": "book", "point": point},
    ])


def _loaded(snapshot=None):
    return QBProjectionSnapshotLoadResult("snapshot_available", snapshot or _snapshot())


@pytest.mark.parametrize(("line", "difference", "text"), [
    (250.5, 8.4, "above"), (270.0, -11.1, "below"), (258.9, 0.0, "equals"),
])
def test_id_matched_projection_displays_signed_line_difference(line, difference, text):
    result = build_qb_projection_presentation(_loaded(), _matchup(), _props(line))
    assert result.status == "matched"
    assert result.match_method == "player_id"
    assert result.displayed_projection == "258.9"
    assert result.line_comparisons[0].projection_minus_line == pytest.approx(difference)
    assert text in result.line_comparisons[0].direction_text
    assert result.as_of_display == "2026-10-06 14:00 UTC"


def test_projection_without_line_and_line_without_projection_are_nonfatal():
    no_line = build_qb_projection_presentation(_loaded(), _matchup(), None)
    assert no_line.status == "matched" and not no_line.line_comparisons
    no_projection = build_qb_projection_presentation(_loaded(), _matchup(player_id="missing"), _props())
    assert no_projection.status == "projection_unavailable"


def test_unavailable_invalid_stale_and_game_mismatch_are_classified(tmp_path):
    unavailable = load_qb_projection_snapshot_for_ui(tmp_path / "absent.json")
    assert unavailable.status == "snapshot_unavailable"
    broken = tmp_path / "broken.json"; broken.write_text("{", encoding="utf-8")
    assert load_qb_projection_snapshot_for_ui(broken).status == "snapshot_invalid"
    stale = build_qb_projection_presentation(_loaded(), _matchup(week=6), _props())
    assert stale.status == "snapshot_stale"
    wrong_game = build_qb_projection_presentation(_loaded(), _matchup(game_id="other"), _props())
    assert wrong_game.status == "projection_unavailable"


def test_name_fallback_reuses_suffix_matching_and_rejects_ambiguity():
    suffix = build_qb_projection_presentation(_loaded(), _matchup(player_id=None, player_name="John Doe"), _props())
    assert suffix.status == "matched"
    assert suffix.match_method == "suffix_normalized_name_and_position"
    snapshot = _snapshot()
    duplicate = QBPassingYardsWeeklyProjectionSnapshotProjection(
        2026, 5, "game-1", "2026-10-09T00:15:00+00:00", "qb-3",
        "John Doe III", "AAA", "BBB", "home", 250.0, False, False,
    )
    ambiguous = replace(snapshot, projections=(snapshot.projections[0], duplicate, snapshot.projections[1]))
    result = build_qb_projection_presentation(_loaded(ambiguous), _matchup(player_id=None, player_name="John Doe"), _props())
    assert result.status == "projection_ambiguous"


def test_inputs_are_not_mutated_and_no_decision_or_model_side_effects():
    matchup, props = _matchup(), _props()
    matchup_before, props_before = matchup.copy(deep=True), props.copy(deep=True)
    result = build_qb_projection_presentation(_loaded(), matchup, props)
    assert result.status == "matched"
    pd.testing.assert_series_equal(matchup, matchup_before)
    pd.testing.assert_frame_equal(props, props_before)
    assert not hasattr(result, "decision")
    assert not any(hasattr(value, "fit") or hasattr(value, "predict") for value in result.__dict__.values())
