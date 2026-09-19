import pandas as pd
import pytest

from football.data.build_quarterback_dataset import OUTPUT_COLUMNS as QB_COLUMNS
from football.training.build_qb_passing_yards_dataset import (
    FEATURE_COLUMNS,
    METADATA_COLUMNS,
    OUTPUT_COLUMNS,
    TARGET_COLUMN,
    TARGET_KEY,
    build_qb_passing_yards_training_dataset,
)


def _inputs():
    qb_rows = []
    schedule_rows = []

    def add_game(season, week, game_id, away, home, away_qb, home_qb, away_yards, home_yards):
        schedule_rows.extend(
            [
                {"season": season, "week": week, "game_id": game_id, "team": home, "opponent": away, "home_away": "home"},
                {"season": season, "week": week, "game_id": game_id, "team": away, "opponent": home, "home_away": "away"},
            ]
        )
        for team, opponent, player_id, yards in [
            (away, home, away_qb, away_yards),
            (home, away, home_qb, home_yards),
        ]:
            qb_rows.append(
                {
                    "season": season, "week": week, "game_id": game_id,
                    "player_id": player_id, "player_name": f"{player_id} name",
                    "team": team, "opponent": opponent, "passing_attempts": yards // 10,
                    "completions": yards // 20, "passing_yards": yards,
                    "passing_touchdowns": 1, "interceptions": 0,
                }
            )

    # A's Week 1 target must use only 2025 history; later 2026 games are traps.
    add_game(2025, 1, "2025_01_A_B", "A", "B", "qb_a", "qb_b", 100, 180)
    add_game(2025, 2, "2025_02_A_C", "A", "C", "qb_a", "qb_c", 200, 170)
    add_game(2026, 1, "2026_01_A_D", "A", "D", "qb_a", "qb_d", 300, 190)
    add_game(2026, 1, "2026_01_X_E", "X", "E", "qb_x", "qb_e", 160, 210)
    add_game(2026, 2, "2026_02_A_E", "A", "E", "qb_a", "qb_e2", 400, 999)
    # Same-week A game is intentionally synthetic: it must not enter A's W2 history.
    add_game(2026, 2, "2026_02_A_Z", "A", "Z", "qb_a", "qb_z", 900, 220)
    add_game(2026, 2, "2026_02_Y_E", "Y", "E", "qb_y", "qb_e3", 999, 230)
    add_game(2026, 3, "2026_03_A_F", "A", "F", "qb_a", "qb_f", 500, 200)
    add_game(2026, 3, "2026_03_Q_E", "Q", "E", "qb_q", "qb_e4", 777, 201)
    add_game(2026, 4, "2026_04_A_G", "A", "G", "qb_a", "qb_g", 600, 205)
    add_game(2026, 5, "2026_05_A_H", "A", "H", "qb_a", "qb_h", 700, 206)
    return pd.DataFrame(qb_rows, columns=QB_COLUMNS), pd.DataFrame(schedule_rows)


def _row(data, game_id, player_id="qb_a"):
    return data.loc[(data["game_id"] == game_id) & (data["player_id"] == player_id)].iloc[0]


def test_builds_one_row_per_qb_target_with_contract_and_context():
    qbs, schedules = _inputs()
    result = build_qb_passing_yards_training_dataset(qbs, schedules)

    assert result.columns.tolist() == OUTPUT_COLUMNS
    assert len(result) == len(qbs)
    assert not result.duplicated(TARGET_KEY).any()
    target = _row(result, "2026_02_A_E")
    assert target[TARGET_COLUMN] == 400.0
    assert target[["team", "opponent", "home_away"]].to_dict() == {
        "team": "A", "opponent": "E", "home_away": "away"
    }
    forbidden_outcomes = {
        "passing_attempts", "completions", "passing_yards",
        "passing_touchdowns", "interceptions", "home_score", "away_score",
    }
    assert not forbidden_outcomes.intersection(FEATURE_COLUMNS)


def test_week_cutoffs_exclude_target_same_week_and_future_for_qb_and_defense():
    qbs, schedules = _inputs()
    result = build_qb_passing_yards_training_dataset(qbs, schedules)

    week1 = _row(result, "2026_01_A_D")
    assert week1["qb_season_passing_yards_avg"] == 150.0
    assert week1["qb_season_passing_attempts_avg"] == 15.0
    assert week1["qb_season_history_games"] == 2

    week2 = _row(result, "2026_02_A_E")
    assert week2["qb_season_passing_yards_avg"] == 300.0
    assert week2["qb_last3_passing_yards_avg"] == 300.0
    # E allowed 160 in Week 1; E's 999 allowed in target week and 777 later
    # cannot appear in this target's defense features.
    assert week2["defense_season_passing_yards_allowed_avg"] == 160.0
    assert week2["defense_last3_passing_yards_allowed_avg"] == 160.0
    assert week2["defense_season_passing_attempts_allowed_avg"] == 16.0


def test_last_three_uses_the_latest_three_eligible_prior_games_only():
    qbs, schedules = _inputs()
    result = build_qb_passing_yards_training_dataset(qbs, schedules)
    week5 = _row(result, "2026_05_A_H")

    # Week 5's season window is 2026 weeks 1-4, including both Week 2 rows;
    # the last three are 900, 500, and 600 by stable week/game ordering.
    assert week5["qb_season_passing_yards_avg"] == 540.0
    assert week5["qb_last3_passing_yards_avg"] == pytest.approx(2000 / 3)
    assert week5["qb_last3_history_games"] == 3


def test_missing_history_is_a_cold_start_without_imputation():
    qbs, schedules = _inputs()
    qbs = qbs.loc[qbs["season"] == 2026].copy()
    result = build_qb_passing_yards_training_dataset(qbs, schedules.loc[schedules["season"] == 2026])
    row = _row(result, "2026_01_A_D")

    assert pd.isna(row["qb_season_passing_yards_avg"])
    assert row["qb_season_history_games"] == 0
    assert row["qb_missing_history"]
    assert pd.isna(row["defense_season_passing_yards_allowed_avg"])
    assert row["defense_season_history_games"] == 0
    assert row["defense_missing_history"]


def test_keeps_multiple_target_game_qbs_without_selecting_a_target_starter():
    qbs, schedules = _inputs()
    backup = qbs.loc[qbs["game_id"] == "2026_02_A_E"].iloc[[0]].copy()
    backup["player_id"] = "qb_a_backup"
    backup["player_name"] = "qb_a_backup name"
    backup["passing_attempts"] = 3
    backup["passing_yards"] = 25

    result = build_qb_passing_yards_training_dataset(
        pd.concat([qbs, backup], ignore_index=True), schedules
    )

    target_rows = result.loc[result["game_id"] == "2026_02_A_E"]
    assert set(target_rows["player_id"]) == {"qb_a", "qb_a_backup", "qb_e2"}
    assert _row(result, "2026_02_A_E", "qb_a_backup")[TARGET_COLUMN] == 25.0


def test_excludes_nonfinite_target_passing_yards():
    qbs, schedules = _inputs()
    qbs["passing_yards"] = qbs["passing_yards"].astype(float)
    qbs.loc[qbs["game_id"] == "2026_05_A_H", "passing_yards"] = float("inf")

    result = build_qb_passing_yards_training_dataset(qbs, schedules)

    assert "2026_05_A_H" not in result["game_id"].tolist()


def test_is_deterministic_does_not_mutate_and_collapses_exact_duplicates():
    qbs, schedules = _inputs()
    original_qbs = qbs.copy(deep=True)
    original_schedules = schedules.copy(deep=True)
    duplicate_qbs = pd.concat([qbs, qbs.iloc[[0]]], ignore_index=True)

    expected = build_qb_passing_yards_training_dataset(qbs, schedules)
    actual = build_qb_passing_yards_training_dataset(
        duplicate_qbs.sample(frac=1, random_state=4),
        schedules.sample(frac=1, random_state=9),
    )

    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(qbs, original_qbs)
    pd.testing.assert_frame_equal(schedules, original_schedules)
    assert actual[TARGET_KEY].values.tolist() == sorted(actual[TARGET_KEY].values.tolist())


def test_rejects_conflicts_missing_columns_and_ambiguous_schedule_context():
    qbs, schedules = _inputs()
    conflicting = pd.concat([qbs, qbs.iloc[[0]].assign(passing_yards=333)], ignore_index=True)
    with pytest.raises(ValueError, match="Conflicting quarterback-game records"):
        build_qb_passing_yards_training_dataset(conflicting, schedules)

    with pytest.raises(ValueError, match="missing required columns"):
        build_qb_passing_yards_training_dataset(qbs.drop(columns="passing_yards"), schedules)

    ambiguous = pd.concat(
        [schedules, schedules.iloc[[0]].assign(opponent="NOT_B")], ignore_index=True
    )
    with pytest.raises(ValueError, match="Conflicting normalized schedule rows"):
        build_qb_passing_yards_training_dataset(qbs, ambiguous)

    with pytest.raises(ValueError, match="missing schedule context"):
        build_qb_passing_yards_training_dataset(qbs, schedules.iloc[1:])
