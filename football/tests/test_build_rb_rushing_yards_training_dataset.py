import numpy as np
import pandas as pd
import pytest

from football.data.build_running_back_dataset import OUTPUT_COLUMNS as RB_COLUMNS
from football.training.build_rb_rushing_yards_dataset import (
    FEATURE_COLUMNS,
    OUTPUT_COLUMNS,
    TARGET_COLUMN,
    build_rb_rushing_yards_training_dataset,
)


def rb_row(season, week, game_id, player_id, team, opponent, carries, yards, targets=0):
    return {
        "season": season, "week": week, "game_id": game_id, "player_id": player_id,
        "player_name": player_id, "team": team, "opponent": opponent, "position": "RB",
        "rushing_attempts": carries, "rushing_yards": yards, "rushing_touchdowns": 0,
        "receptions": min(targets, 1), "targets": targets,
        "receiving_yards": targets * 5, "receiving_touchdowns": 0,
    }


def schedule_for(rows):
    records = []
    for season, week, game_id in sorted({(r["season"], r["week"], r["game_id"]) for r in rows}):
        teams = {(r["team"], r["opponent"]) for r in rows if (r["season"], r["week"], r["game_id"]) == (season, week, game_id)}
        for team, opponent in teams:
            records.append({"season": season, "week": week, "game_id": game_id, "team": team, "opponent": opponent, "home_away": "home"})
            records.append({"season": season, "week": week, "game_id": game_id, "team": opponent, "opponent": team, "home_away": "away"})
    return pd.DataFrame(records).drop_duplicates()


def build(rows):
    data = pd.DataFrame(rows, columns=RB_COLUMNS)
    return build_rb_rushing_yards_training_dataset(data, schedule_for(rows))


def target(dataset, player_id, week):
    return dataset.loc[(dataset.player_id == player_id) & (dataset.week == week)].iloc[0]


def test_week_one_uses_only_immediately_prior_season_and_later_weeks_use_current_season():
    rows = [
        rb_row(2023, 18, "old", "rb", "A", "D", 20, 200),
        rb_row(2024, 18, "prior", "rb", "A", "D", 10, 100),
        rb_row(2025, 1, "w1", "rb", "A", "D", 5, 50),
        rb_row(2025, 2, "w2", "rb", "A", "D", 6, 60),
    ]
    result = build(rows)
    assert target(result, "rb", 1).rb_season_rushing_yards_avg == 100
    assert target(result, "rb", 2).rb_season_rushing_yards_avg == 50


def test_target_week_future_usage_and_role_changes_do_not_change_features_or_inclusion():
    rows = [
        rb_row(2025, 1, "g1", "rb", "A", "D", 10, 40, 2),
        rb_row(2025, 2, "g2", "rb", "A", "D", 1, 7, 0),
        rb_row(2025, 2, "g2", "mate", "A", "D", 20, 80, 6),
        rb_row(2025, 3, "g3", "rb", "A", "D", 9, 90, 3),
    ]
    changed = [dict(row) for row in rows]
    changed[1].update(rushing_attempts=99, targets=99)
    changed[2].update(rushing_attempts=0, targets=0)
    changed[3].update(rushing_attempts=1, rushing_yards=-10, targets=0)
    first, second = build(rows), build(changed)
    first_row, second_row = target(first, "rb", 2), target(second, "rb", 2)
    assert first_row[FEATURE_COLUMNS].to_dict() == second_row[FEATURE_COLUMNS].to_dict()
    assert first_row[TARGET_COLUMN] == second_row[TARGET_COLUMN] == 7


def test_multiple_zero_and_low_volume_backs_are_retained_but_missing_player_games_are_not_created():
    rows = [
        rb_row(2025, 1, "g", "zero", "A", "D", 0, 0),
        rb_row(2025, 1, "g", "low", "A", "D", 1, -2, 1),
    ]
    result = build(rows)
    assert set(result.player_id) == {"zero", "low"}
    assert len(result) == 2
    assert target(result, "zero", 1)[TARGET_COLUMN] == 0
    assert target(result, "low", 1)[TARGET_COLUMN] == -2


def test_historical_backfield_shares_use_all_teammates_and_missing_history_stays_missing():
    rows = [
        rb_row(2024, 18, "prior", "rb", "A", "D", 4, 20, 1),
        rb_row(2024, 18, "prior", "mate", "A", "D", 6, 30, 3),
        rb_row(2025, 1, "target", "rb", "A", "D", 2, 10),
        rb_row(2025, 1, "cold", "B", "E", 0, 0, 0),
    ]
    result = build(rows)
    row = target(result, "rb", 1)
    assert row.rb_season_carry_share_avg == pytest.approx(0.4)
    assert row.rb_season_opportunity_share_avg == pytest.approx(5 / 14)
    cold = target(result, "B", 1)
    assert cold.rb_missing_history and cold.rb_season_history_games == 0
    assert pd.isna(cold.rb_season_rushing_yards_avg)
    assert cold.defense_missing_history and cold.defense_season_history_games == 0


def test_weighted_defensive_ypc_zero_denominator_and_team_change_metadata():
    rows = [
        rb_row(2024, 17, "d1", "other1", "X", "D", 2, 20),
        rb_row(2024, 18, "d2", "other2", "Y", "D", 8, 24),
        rb_row(2024, 18, "oldteam", "rb", "A", "Q", 3, 12),
        rb_row(2025, 1, "target", "rb", "B", "D", 1, 4),
        rb_row(2025, 1, "zero_target", "new", "C", "Z", 1, 3),
        rb_row(2024, 18, "zero_d", "other3", "M", "Z", 0, 0),
    ]
    result = build(rows)
    row = target(result, "rb", 1)
    assert row.defense_season_rb_yards_per_carry_allowed == pytest.approx(4.4)
    assert row.rb_prior_latest_team == "A" and row.player_team_changed
    assert pd.isna(target(result, "new", 1).defense_season_rb_yards_per_carry_allowed)


def test_duplicates_schedule_validation_determinism_and_inputs_are_not_mutated():
    rows = [rb_row(2025, 1, "g", "rb", "A", "D", 2, 8)]
    data = pd.DataFrame(rows, columns=RB_COLUMNS)
    schedule = schedule_for(rows)
    original_data, original_schedule = data.copy(deep=True), schedule.copy(deep=True)
    expected = build_rb_rushing_yards_training_dataset(data, schedule)
    actual = build_rb_rushing_yards_training_dataset(data.sample(frac=1), schedule.sample(frac=1))
    pd.testing.assert_frame_equal(expected, actual)
    pd.testing.assert_frame_equal(data, original_data)
    pd.testing.assert_frame_equal(schedule, original_schedule)
    duplicate = pd.concat([data, data.assign(rushing_yards=9)], ignore_index=True)
    with pytest.raises(ValueError, match="Conflicting running-back"):
        build_rb_rushing_yards_training_dataset(duplicate, schedule)
    bad_schedule = schedule.loc[schedule.team != "A"].copy()
    with pytest.raises(ValueError, match="missing schedule context"):
        build_rb_rushing_yards_training_dataset(data, bad_schedule)


def test_invalid_targets_are_excluded_and_empty_valid_input_has_documented_schema():
    rows = [rb_row(2025, 1, "g", "bad", "A", "D", 1, np.nan)]
    assert build(rows).empty
    assert build(rows).columns.tolist() == OUTPUT_COLUMNS
    empty = pd.DataFrame(columns=RB_COLUMNS)
    schedule = pd.DataFrame(columns=["season", "week", "game_id", "team", "opponent", "home_away"])
    result = build_rb_rushing_yards_training_dataset(empty, schedule)
    assert result.empty and result.columns.tolist() == OUTPUT_COLUMNS
