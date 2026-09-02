import pandas as pd
import pytest

from football.data.build_quarterback_dataset import (
    OUTPUT_COLUMNS,
    build_quarterback_dataset,
)


def make_quarterback_rows() -> pd.DataFrame:
    """Create unsorted quarterback-game rows."""

    return pd.DataFrame(
        {
            "season": [2026, 2026, 2025],
            "week": [2, 1, 18],
            "game_id": [
                "2026_02_BAL_CIN",
                "2026_01_KC_LAC",
                "2025_18_BUF_MIA",
            ],
            "player_id": ["qb_bal", "qb_kc", "qb_buf"],
            "player_name": [
                "Baltimore QB",
                "Kansas City QB",
                "Buffalo QB",
            ],
            "team": ["BAL", "KC", "BUF"],
            "opponent": ["CIN", "LAC", "MIA"],
            "passing_attempts": [31, 35, 28],
            "completions": [20, 24, 19],
            "passing_yards": [240, 285, 221],
            "passing_touchdowns": [2, 3, 1],
            "interceptions": [1, 0, 1],
            "ignored_column": ["x", "y", "w"],
        }
    )


def test_build_quarterback_dataset_outputs_expected_columns():
    result = build_quarterback_dataset(make_quarterback_rows())

    assert result.columns.tolist() == OUTPUT_COLUMNS


def test_build_quarterback_dataset_returns_one_row_per_quarterback_game():
    data = make_quarterback_rows()
    data = pd.concat([data, data.iloc[[1]]], ignore_index=True)

    result = build_quarterback_dataset(data)

    assert not result.duplicated(
        subset=["season", "week", "game_id", "player_id"]
    ).any()
    assert len(result) == 3


def test_build_quarterback_dataset_removes_identical_duplicates():
    data = make_quarterback_rows()
    data = pd.concat([data, data.iloc[[1]]], ignore_index=True)

    result = build_quarterback_dataset(data)

    duplicate_game = result[
        (result["season"] == 2026)
        & (result["week"] == 1)
        & (result["game_id"] == "2026_01_KC_LAC")
        & (result["player_id"] == "qb_kc")
    ].iloc[0]

    assert duplicate_game["player_name"] == "Kansas City QB"
    assert duplicate_game["passing_yards"] == 285


def test_build_quarterback_dataset_rejects_conflicting_duplicates():
    data = make_quarterback_rows()
    conflicting_row = data.iloc[[1]].copy()
    conflicting_row.loc[:, "passing_yards"] = 300
    data = pd.concat([data, conflicting_row], ignore_index=True)

    with pytest.raises(ValueError) as error:
        build_quarterback_dataset(data)

    message = str(error.value)

    assert "Conflicting quarterback-game records found" in message
    assert "2026_01_KC_LAC" in message
    assert "qb_kc" in message


def test_build_quarterback_dataset_sorts_chronologically():
    result = build_quarterback_dataset(make_quarterback_rows())

    assert result[
        ["season", "week", "game_id", "player_id"]
    ].values.tolist() == [
        [2025, 18, "2025_18_BUF_MIA", "qb_buf"],
        [2026, 1, "2026_01_KC_LAC", "qb_kc"],
        [2026, 2, "2026_02_BAL_CIN", "qb_bal"],
    ]


def test_build_quarterback_dataset_validates_missing_columns():
    data = make_quarterback_rows().drop(
        columns=[
            "passing_yards",
            "interceptions",
        ]
    )

    with pytest.raises(ValueError) as error:
        build_quarterback_dataset(data)

    message = str(error.value)

    assert "Quarterback data is missing required columns" in message
    assert "passing_yards" in message
    assert "interceptions" in message


def test_build_quarterback_dataset_does_not_mutate_input():
    data = make_quarterback_rows()
    original = data.copy(deep=True)

    build_quarterback_dataset(data)

    pd.testing.assert_frame_equal(data, original)
