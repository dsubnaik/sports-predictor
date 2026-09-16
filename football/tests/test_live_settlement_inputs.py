"""Tests for deliberate live settlement-input retrieval with injected sources."""

from copy import deepcopy

import pandas as pd
import pytest

import football.results.live_settlement_inputs as live_inputs
from football.data.build_quarterback_dataset import OUTPUT_COLUMNS as QB_COLUMNS
from football.data.build_running_back_dataset import OUTPUT_COLUMNS as RB_COLUMNS
from football.data.build_schedule_dataset import OUTPUT_COLUMNS as SCHEDULE_COLUMNS
from football.results.live_settlement_inputs import (
    LiveSettlementInputsValidationError,
    load_live_settlement_inputs,
)


def schedules() -> pd.DataFrame:
    return pd.DataFrame({
        "season": [2026], "week": [1], "game_id": ["2026_01_KC_LAC"],
        "game_type": ["REG"], "gameday": ["2026-09-10"], "gametime": ["20:20"],
        "home_team": ["LAC"], "away_team": ["KC"],
        "home_score": [21.0], "away_score": [17.0],
    })


def player_stats() -> pd.DataFrame:
    return pd.DataFrame({
        "season": [2026, 2026], "week": [1, 1], "season_type": ["REG", "REG"],
        "game_id": ["2026_01_KC_LAC", "2026_01_KC_LAC"],
        "player_id": ["qb-kc", "rb-kc"],
        "player_display_name": ["Quarterback", "Running Back"],
        "position": ["QB", "RB"], "team": ["KC", "KC"],
        "opponent_team": ["LAC", "LAC"], "attempts": [35, 0],
        "completions": [24, 0], "passing_yards": [285, 0],
        "passing_tds": [3, 0], "passing_interceptions": [0, 0],
        "carries": [0, 18], "rushing_yards": [0, 84], "rushing_tds": [0, 1],
        "receptions": [0, 4], "targets": [0, 5], "receiving_yards": [0, 36],
        "receiving_tds": [0, 1],
    })


def test_module_exposes_loader_without_eagerly_invoking_it():
    assert callable(live_inputs.load_live_settlement_inputs)


def test_valid_load_normalizes_all_inputs_once_and_preserves_scores_payload():
    calls = []
    schedule_source = schedules()
    stats_source = player_stats()
    payload = [{"id": "completed", "completed": True}, {"id": "live", "completed": False}]
    original_schedule = schedule_source.copy(deep=True)
    original_stats = stats_source.copy(deep=True)
    original_payload = deepcopy(payload)

    def schedule_loader(seasons):
        calls.append(("schedule", seasons))
        return schedule_source

    def stats_loader(seasons):
        calls.append(("stats", seasons))
        return stats_source

    def scores_fetcher(*, days_from):
        calls.append(("scores", days_from))
        return payload

    result = load_live_settlement_inputs(
        2026,
        schedules_loader=schedule_loader,
        player_stats_loader=stats_loader,
        scores_fetcher=scores_fetcher,
    )

    assert calls == [("schedule", [2026]), ("stats", [2026]), ("scores", 3)]
    assert result.season == 2026
    assert result.normalized_schedule.columns.tolist() == SCHEDULE_COLUMNS
    assert len(result.normalized_schedule) == 2
    assert result.normalized_schedule[["home_score", "away_score"]].values.tolist() == [[21.0, 17.0], [21.0, 17.0]]
    assert result.quarterback_games.columns.tolist() == QB_COLUMNS
    assert result.quarterback_games.loc[0, "passing_yards"] == 285
    assert result.running_back_games.columns.tolist() == RB_COLUMNS
    assert result.running_back_games.loc[0, "rushing_yards"] == 84
    assert result.scores_payload == payload
    assert result.scores_payload is not payload
    pd.testing.assert_frame_equal(schedule_source, original_schedule)
    pd.testing.assert_frame_equal(stats_source, original_stats)
    assert payload == original_payload


def test_explicit_days_from_is_forwarded_once():
    calls = []
    load_live_settlement_inputs(
        2026,
        schedules_loader=lambda _seasons: schedules(),
        player_stats_loader=lambda _seasons: player_stats(),
        scores_fetcher=lambda *, days_from: calls.append(days_from) or [],
        days_from=1,
    )
    assert calls == [1]


@pytest.mark.parametrize("season", [0, -1, True, False, 2026.0, "2026"])
def test_invalid_season_fails_before_external_calls(season):
    with pytest.raises(LiveSettlementInputsValidationError, match="season"):
        load_live_settlement_inputs(
            season,
            schedules_loader=pytest.fail,
            player_stats_loader=pytest.fail,
            scores_fetcher=pytest.fail,
        )


@pytest.mark.parametrize("days_from", [0, -1, 4, True, False, 1.0, "1"])
def test_invalid_days_from_fails_before_external_calls(days_from):
    with pytest.raises(LiveSettlementInputsValidationError, match="days_from"):
        load_live_settlement_inputs(
            2026,
            schedules_loader=pytest.fail,
            player_stats_loader=pytest.fail,
            scores_fetcher=pytest.fail,
            days_from=days_from,
        )


def test_empty_valid_sources_produce_empty_normalized_frames():
    empty_schedule = schedules().iloc[0:0].copy()
    empty_stats = player_stats().iloc[0:0].copy()
    result = load_live_settlement_inputs(
        2026,
        schedules_loader=lambda _seasons: empty_schedule,
        player_stats_loader=lambda _seasons: empty_stats,
        scores_fetcher=lambda **_kwargs: [],
    )
    assert result.normalized_schedule.empty and result.normalized_schedule.columns.tolist() == SCHEDULE_COLUMNS
    assert result.quarterback_games.empty and result.quarterback_games.columns.tolist() == QB_COLUMNS
    assert result.running_back_games.empty and result.running_back_games.columns.tolist() == RB_COLUMNS


def test_loader_and_builder_failures_return_no_partial_inputs():
    with pytest.raises(RuntimeError, match="schedule failure"):
        load_live_settlement_inputs(
            2026,
            schedules_loader=lambda _seasons: (_ for _ in ()).throw(RuntimeError("schedule failure")),
            player_stats_loader=pytest.fail,
            scores_fetcher=pytest.fail,
        )

    bad_stats = player_stats().drop(columns=["rushing_yards"])
    with pytest.raises(ValueError, match="missing required columns"):
        load_live_settlement_inputs(
            2026,
            schedules_loader=lambda _seasons: schedules(),
            player_stats_loader=lambda _seasons: bad_stats,
            scores_fetcher=pytest.fail,
        )


def test_conflicting_duplicate_rows_propagate_builder_failure_without_scores_call():
    conflicting = pd.concat([player_stats(), player_stats().iloc[[1]].assign(rushing_yards=99)], ignore_index=True)
    with pytest.raises(ValueError, match="Conflicting running-back game records"):
        load_live_settlement_inputs(
            2026,
            schedules_loader=lambda _seasons: schedules(),
            player_stats_loader=lambda _seasons: conflicting,
            scores_fetcher=pytest.fail,
        )


@pytest.mark.parametrize("source_name", ["schedule", "stats"])
def test_mixed_season_source_is_rejected_after_normalization_without_scores_call(source_name):
    schedule_source = schedules()
    stats_source = player_stats()
    if source_name == "schedule":
        schedule_source.loc[0, "season"] = 2025
    else:
        stats_source.loc[:, "season"] = 2025
    with pytest.raises(LiveSettlementInputsValidationError, match="other than requested season"):
        load_live_settlement_inputs(
            2026,
            schedules_loader=lambda _seasons: schedule_source,
            player_stats_loader=lambda _seasons: stats_source,
            scores_fetcher=pytest.fail,
        )
