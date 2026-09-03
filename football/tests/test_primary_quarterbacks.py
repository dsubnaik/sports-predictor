import pandas as pd
import pytest

from football.data.build_quarterback_dataset import OUTPUT_COLUMNS
from football.features.primary_quarterbacks import (
    PRIMARY_QUARTERBACK_COLUMNS,
    identify_primary_quarterbacks,
)


def make_quarterback_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_KC_LAC",
                "player_id": "qb_kc_starter",
                "player_name": "Kansas City Starter",
                "team": "KC",
                "opponent": "LAC",
                "passing_attempts": 32,
                "completions": 21,
                "passing_yards": 275,
                "passing_touchdowns": 2,
                "interceptions": 0,
            },
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_KC_LAC",
                "player_id": "qb_kc_backup",
                "player_name": "Kansas City Backup",
                "team": "KC",
                "opponent": "LAC",
                "passing_attempts": 5,
                "completions": 3,
                "passing_yards": 30,
                "passing_touchdowns": 0,
                "interceptions": 0,
            },
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_KC_LAC",
                "player_id": "qb_lac_starter",
                "player_name": "Los Angeles Starter",
                "team": "LAC",
                "opponent": "KC",
                "passing_attempts": 28,
                "completions": 18,
                "passing_yards": 230,
                "passing_touchdowns": 1,
                "interceptions": 1,
            },
            {
                "season": 2026,
                "week": 2,
                "game_id": "2026_02_BUF_MIA",
                "player_id": "qb_buf_low",
                "player_name": "Buffalo Low",
                "team": "BUF",
                "opponent": "MIA",
                "passing_attempts": 14,
                "completions": 9,
                "passing_yards": 100,
                "passing_touchdowns": 0,
                "interceptions": 1,
            },
            {
                "season": 2026,
                "week": 2,
                "game_id": "2026_02_BUF_MIA",
                "player_id": "qb_mia_primary",
                "player_name": "Miami Primary",
                "team": "MIA",
                "opponent": "BUF",
                "passing_attempts": 20,
                "completions": 13,
                "passing_yards": 160,
                "passing_touchdowns": 1,
                "interceptions": 0,
            },
            {
                "season": 2026,
                "week": 2,
                "game_id": "2026_02_BUF_MIA",
                "player_id": "qb_mia_split",
                "player_name": "Miami Split",
                "team": "MIA",
                "opponent": "BUF",
                "passing_attempts": 14,
                "completions": 8,
                "passing_yards": 95,
                "passing_touchdowns": 0,
                "interceptions": 0,
            },
            {
                "season": 2026,
                "week": 3,
                "game_id": "2026_03_BAL_CIN",
                "player_id": "qb_bal_a",
                "player_name": "Baltimore A",
                "team": "BAL",
                "opponent": "CIN",
                "passing_attempts": 16,
                "completions": 10,
                "passing_yards": 120,
                "passing_touchdowns": 1,
                "interceptions": 0,
            },
            {
                "season": 2026,
                "week": 3,
                "game_id": "2026_03_BAL_CIN",
                "player_id": "qb_bal_b",
                "player_name": "Baltimore B",
                "team": "BAL",
                "opponent": "CIN",
                "passing_attempts": 16,
                "completions": 11,
                "passing_yards": 130,
                "passing_touchdowns": 1,
                "interceptions": 0,
            },
            {
                "season": 2026,
                "week": 4,
                "game_id": "2026_04_DAL_NYG",
                "player_id": "qb_dal_only",
                "player_name": "Dallas Only",
                "team": "DAL",
                "opponent": "NYG",
                "passing_attempts": 33,
                "completions": 22,
                "passing_yards": 260,
                "passing_touchdowns": 2,
                "interceptions": 1,
            },
            {
                "season": 2026,
                "week": 5,
                "game_id": "2026_05_PHI_WAS",
                "player_id": "qb_phi_primary",
                "player_name": "Philadelphia Primary",
                "team": "PHI",
                "opponent": "WAS",
                "passing_attempts": 25,
                "completions": 16,
                "passing_yards": 210,
                "passing_touchdowns": 1,
                "interceptions": 1,
            },
            {
                "season": 2026,
                "week": 5,
                "game_id": "2026_05_PHI_WAS",
                "player_id": "qb_phi_zero",
                "player_name": "Philadelphia Zero",
                "team": "PHI",
                "opponent": "WAS",
                "passing_attempts": 0,
                "completions": 0,
                "passing_yards": 0,
                "passing_touchdowns": 0,
                "interceptions": 0,
            },
        ],
        columns=OUTPUT_COLUMNS,
    )


def get_team_game(result: pd.DataFrame, team: str) -> pd.Series:
    return result.loc[result["team"] == team].iloc[0]


def test_identify_primary_quarterbacks_selects_normal_starter_over_backup():
    result = identify_primary_quarterbacks(make_quarterback_dataset())

    row = get_team_game(result, "KC")

    assert row["player_id"] == "qb_kc_starter"
    assert row["primary_attempts"] == 32
    assert row["secondary_attempts"] == 5
    assert row["quarterbacks_with_attempts"] == 2


def test_identify_primary_quarterbacks_returns_one_row_per_team_game():
    data = make_quarterback_dataset()

    result = identify_primary_quarterbacks(data)

    expected_team_games = data[
        ["season", "week", "game_id", "team"]
    ].drop_duplicates()

    assert len(result) == len(expected_team_games)
    assert not result.duplicated(
        subset=["season", "week", "game_id", "team"]
    ).any()


def test_identify_primary_quarterbacks_keeps_multiple_teams_in_same_game():
    result = identify_primary_quarterbacks(make_quarterback_dataset())

    teams = result.loc[result["game_id"] == "2026_01_KC_LAC", "team"].tolist()

    assert teams == ["KC", "LAC"]


def test_identify_primary_quarterbacks_flags_low_attempt_primary_qb():
    result = identify_primary_quarterbacks(make_quarterback_dataset())

    row = get_team_game(result, "BUF")

    assert row["low_attempt_primary_qb"]


def test_identify_primary_quarterbacks_flags_similar_attempt_split():
    result = identify_primary_quarterbacks(make_quarterback_dataset())

    row = get_team_game(result, "MIA")

    assert row["similar_attempt_split"]


def test_identify_primary_quarterbacks_flags_exact_tie_and_breaks_by_player_id():
    result = identify_primary_quarterbacks(make_quarterback_dataset())

    row = get_team_game(result, "BAL")

    assert row["player_id"] == "qb_bal_a"
    assert row["exact_attempt_tie"]
    assert row["secondary_attempts"] == 16


def test_identify_primary_quarterbacks_handles_team_with_only_one_quarterback():
    result = identify_primary_quarterbacks(make_quarterback_dataset())

    row = get_team_game(result, "DAL")

    assert row["player_id"] == "qb_dal_only"
    assert row["secondary_attempts"] == 0
    assert row["quarterbacks_with_attempts"] == 1
    assert not row["similar_attempt_split"]
    assert not row["exact_attempt_tie"]


def test_identify_primary_quarterbacks_ignores_zero_attempt_rows_for_split_counts():
    result = identify_primary_quarterbacks(make_quarterback_dataset())

    row = get_team_game(result, "PHI")

    assert row["player_id"] == "qb_phi_primary"
    assert row["secondary_attempts"] == 0
    assert row["quarterbacks_with_attempts"] == 1


def test_identify_primary_quarterbacks_handles_all_zero_attempt_team_game():
    data = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 6,
                "game_id": "2026_06_CHI_DET",
                "player_id": "qb_chi_a",
                "player_name": "Chicago A",
                "team": "CHI",
                "opponent": "DET",
                "passing_attempts": 0,
                "completions": 0,
                "passing_yards": 0,
                "passing_touchdowns": 0,
                "interceptions": 0,
            },
            {
                "season": 2026,
                "week": 6,
                "game_id": "2026_06_CHI_DET",
                "player_id": "qb_chi_b",
                "player_name": "Chicago B",
                "team": "CHI",
                "opponent": "DET",
                "passing_attempts": 0,
                "completions": 0,
                "passing_yards": 0,
                "passing_touchdowns": 0,
                "interceptions": 0,
            },
        ],
        columns=OUTPUT_COLUMNS,
    )

    result = identify_primary_quarterbacks(data)
    row = get_team_game(result, "CHI")

    assert row["player_id"] == "qb_chi_a"
    assert row["primary_attempts"] == 0
    assert row["secondary_attempts"] == 0
    assert row["quarterbacks_with_attempts"] == 0
    assert row["low_attempt_primary_qb"]
    assert not row["similar_attempt_split"]
    assert row["exact_attempt_tie"]


def test_identify_primary_quarterbacks_validates_invalid_thresholds():
    data = make_quarterback_dataset()

    with pytest.raises(ValueError, match="low_attempt_threshold"):
        identify_primary_quarterbacks(data, low_attempt_threshold=-1)

    with pytest.raises(ValueError, match="low_attempt_threshold"):
        identify_primary_quarterbacks(data, low_attempt_threshold="15")

    with pytest.raises(ValueError, match="similar_attempt_ratio"):
        identify_primary_quarterbacks(data, similar_attempt_ratio=-0.01)

    with pytest.raises(ValueError, match="similar_attempt_ratio"):
        identify_primary_quarterbacks(data, similar_attempt_ratio=1.01)

    with pytest.raises(ValueError, match="similar_attempt_ratio"):
        identify_primary_quarterbacks(data, similar_attempt_ratio=True)


def test_identify_primary_quarterbacks_validates_missing_columns():
    data = make_quarterback_dataset().drop(columns=["passing_attempts", "team"])

    with pytest.raises(ValueError) as error:
        identify_primary_quarterbacks(data)

    message = str(error.value)

    assert "Quarterback data is missing required columns" in message
    assert "passing_attempts" in message
    assert "team" in message


def test_identify_primary_quarterbacks_does_not_mutate_input():
    data = make_quarterback_dataset()
    original = data.copy(deep=True)

    identify_primary_quarterbacks(data)

    pd.testing.assert_frame_equal(data, original)


def test_identify_primary_quarterbacks_outputs_expected_columns():
    result = identify_primary_quarterbacks(make_quarterback_dataset())

    assert result.columns.tolist() == PRIMARY_QUARTERBACK_COLUMNS
