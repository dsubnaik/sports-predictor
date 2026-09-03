import pandas as pd
import pytest

from football.features.defense_vs_quarterbacks import (
    DEFENSE_VS_QUARTERBACK_COLUMNS,
    build_defense_vs_primary_quarterback_logs,
)
from football.features.primary_quarterbacks import PRIMARY_QUARTERBACK_COLUMNS


def make_primary_quarterback_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
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
                "primary_attempts": 20,
                "secondary_attempts": 14,
                "quarterbacks_with_attempts": 2,
                "low_attempt_primary_qb": False,
                "similar_attempt_split": True,
                "exact_attempt_tie": False,
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
                "primary_attempts": 28,
                "secondary_attempts": 0,
                "quarterbacks_with_attempts": 1,
                "low_attempt_primary_qb": False,
                "similar_attempt_split": False,
                "exact_attempt_tie": False,
            },
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
                "primary_attempts": 32,
                "secondary_attempts": 5,
                "quarterbacks_with_attempts": 2,
                "low_attempt_primary_qb": False,
                "similar_attempt_split": False,
                "exact_attempt_tie": False,
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
                "primary_attempts": 16,
                "secondary_attempts": 16,
                "quarterbacks_with_attempts": 2,
                "low_attempt_primary_qb": False,
                "similar_attempt_split": True,
                "exact_attempt_tie": True,
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
                "primary_attempts": 14,
                "secondary_attempts": 0,
                "quarterbacks_with_attempts": 1,
                "low_attempt_primary_qb": True,
                "similar_attempt_split": False,
                "exact_attempt_tie": False,
            },
        ],
        columns=PRIMARY_QUARTERBACK_COLUMNS,
    )


def get_defense_game(result: pd.DataFrame, defense: str) -> pd.Series:
    return result.loc[result["defense"] == defense].iloc[0]


def test_build_defense_vs_quarterbacks_maps_offense_to_defense_perspective():
    result = build_defense_vs_primary_quarterback_logs(
        make_primary_quarterback_rows()
    )

    row = get_defense_game(result, "LAC")

    assert row["offense_team"] == "KC"
    assert row["opposing_qb_id"] == "qb_kc_starter"
    assert row["opposing_qb_name"] == "Kansas City Starter"
    assert row["passing_attempts_allowed"] == 32


def test_build_defense_vs_quarterbacks_keeps_both_defenses_from_same_game():
    result = build_defense_vs_primary_quarterback_logs(
        make_primary_quarterback_rows()
    )

    defenses = result.loc[
        result["game_id"] == "2026_01_KC_LAC",
        "defense",
    ].tolist()

    assert defenses == ["KC", "LAC"]


def test_build_defense_vs_quarterbacks_returns_one_row_per_defense_game():
    result = build_defense_vs_primary_quarterback_logs(
        make_primary_quarterback_rows()
    )

    expected_defense_games = result[
        ["season", "week", "game_id", "defense"]
    ].drop_duplicates()

    assert len(result) == len(expected_defense_games)
    assert not result.duplicated(
        subset=["season", "week", "game_id", "defense"]
    ).any()


def test_build_defense_vs_quarterbacks_preserves_passing_statistics():
    result = build_defense_vs_primary_quarterback_logs(
        make_primary_quarterback_rows()
    )

    row = get_defense_game(result, "KC")

    assert row["passing_attempts_allowed"] == 28
    assert row["completions_allowed"] == 18
    assert row["passing_yards_allowed"] == 230
    assert row["passing_touchdowns_allowed"] == 1
    assert row["opposing_interceptions"] == 1


def test_build_defense_vs_quarterbacks_preserves_data_quality_flags():
    result = build_defense_vs_primary_quarterback_logs(
        make_primary_quarterback_rows()
    )

    low_attempt_row = get_defense_game(result, "MIA")
    split_row = get_defense_game(result, "BUF")
    tie_row = get_defense_game(result, "CIN")

    assert low_attempt_row["low_attempt_primary_qb"]
    assert not low_attempt_row["similar_attempt_split"]
    assert split_row["similar_attempt_split"]
    assert tie_row["exact_attempt_tie"]
    assert tie_row["quarterbacks_with_attempts"] == 2


def test_build_defense_vs_quarterbacks_validates_missing_columns():
    data = make_primary_quarterback_rows().drop(
        columns=["opponent", "low_attempt_primary_qb"]
    )

    with pytest.raises(ValueError) as error:
        build_defense_vs_primary_quarterback_logs(data)

    message = str(error.value)

    assert "Primary quarterback data is missing required columns" in message
    assert "opponent" in message
    assert "low_attempt_primary_qb" in message


def test_build_defense_vs_quarterbacks_rejects_conflicting_defense_games():
    data = make_primary_quarterback_rows()
    conflict = data.iloc[[0]].copy()
    conflict["team"] = "NE"
    conflict["player_id"] = "qb_ne_other"
    data = pd.concat([data, conflict], ignore_index=True)

    with pytest.raises(ValueError) as error:
        build_defense_vs_primary_quarterback_logs(data)

    message = str(error.value)

    assert "Conflicting defense-game records found" in message
    assert "BUF" in message
    assert "2026_02_BUF_MIA" in message


def test_build_defense_vs_quarterbacks_sorts_chronologically():
    result = build_defense_vs_primary_quarterback_logs(
        make_primary_quarterback_rows()
    )

    assert result[["season", "week", "game_id", "defense"]].to_dict(
        orient="records"
    ) == [
        {
            "season": 2026,
            "week": 1,
            "game_id": "2026_01_KC_LAC",
            "defense": "KC",
        },
        {
            "season": 2026,
            "week": 1,
            "game_id": "2026_01_KC_LAC",
            "defense": "LAC",
        },
        {
            "season": 2026,
            "week": 2,
            "game_id": "2026_02_BUF_MIA",
            "defense": "BUF",
        },
        {
            "season": 2026,
            "week": 2,
            "game_id": "2026_02_BUF_MIA",
            "defense": "MIA",
        },
        {
            "season": 2026,
            "week": 3,
            "game_id": "2026_03_BAL_CIN",
            "defense": "CIN",
        },
    ]


def test_build_defense_vs_quarterbacks_does_not_mutate_input():
    data = make_primary_quarterback_rows()
    original = data.copy(deep=True)

    build_defense_vs_primary_quarterback_logs(data)

    pd.testing.assert_frame_equal(data, original)


def test_build_defense_vs_quarterbacks_outputs_expected_columns():
    result = build_defense_vs_primary_quarterback_logs(
        make_primary_quarterback_rows()
    )

    assert result.columns.tolist() == DEFENSE_VS_QUARTERBACK_COLUMNS
