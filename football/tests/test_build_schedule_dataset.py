import pandas as pd
import pytest

from football.data.build_schedule_dataset import (
    OUTPUT_COLUMNS,
    normalize_schedule_dataset,
)


def make_schedule_rows() -> pd.DataFrame:
    """Create unsorted fake nflverse schedule rows."""

    return pd.DataFrame(
        {
            "season": [2026, 2026, 2026, 2025],
            "week": [2, 1, 20, 18],
            "game_id": [
                "2026_02_BAL_CIN",
                "2026_01_KC_LAC",
                "2026_20_KC_BUF",
                "2025_18_BUF_MIA",
            ],
            "game_type": ["REG", "REG", "DIV", "REG"],
            "gameday": [
                "2026-09-20",
                "2026-09-10",
                "2027-01-17",
                "2026-01-04",
            ],
            "gametime": ["13:00", "20:20", "18:30", "16:25"],
            "home_team": ["CIN", "LAC", "BUF", "MIA"],
            "away_team": ["BAL", "KC", "KC", "BUF"],
            "ignored_column": ["a", "b", "c", "d"],
        }
    )


def test_normalize_schedule_dataset_maps_nflverse_columns_to_contract():
    result = normalize_schedule_dataset(make_schedule_rows())

    assert result.columns.tolist() == OUTPUT_COLUMNS
    assert result.iloc[0].to_dict() == {
        "season": 2025,
        "week": 18,
        "game_id": "2025_18_BUF_MIA",
        "game_date": "2026-01-04",
        "game_time": "16:25",
        "team": "MIA",
        "opponent": "BUF",
        "home_away": "home",
    }


def test_normalize_schedule_dataset_returns_two_rows_per_game():
    result = normalize_schedule_dataset(make_schedule_rows())

    counts = result.groupby("game_id").size().to_dict()

    assert counts == {
        "2025_18_BUF_MIA": 2,
        "2026_01_KC_LAC": 2,
        "2026_02_BAL_CIN": 2,
    }


def test_normalize_schedule_dataset_creates_home_and_away_perspectives():
    result = normalize_schedule_dataset(make_schedule_rows())
    game_rows = result[result["game_id"] == "2026_01_KC_LAC"]

    assert game_rows.to_dict(orient="records") == [
        {
            "season": 2026,
            "week": 1,
            "game_id": "2026_01_KC_LAC",
            "game_date": "2026-09-10",
            "game_time": "20:20",
            "team": "LAC",
            "opponent": "KC",
            "home_away": "home",
        },
        {
            "season": 2026,
            "week": 1,
            "game_id": "2026_01_KC_LAC",
            "game_date": "2026-09-10",
            "game_time": "20:20",
            "team": "KC",
            "opponent": "LAC",
            "home_away": "away",
        },
    ]


def test_normalize_schedule_dataset_handles_multiple_games_and_weeks():
    result = normalize_schedule_dataset(make_schedule_rows())

    assert result[["season", "week", "game_id"]].drop_duplicates().values.tolist() == [
        [2025, 18, "2025_18_BUF_MIA"],
        [2026, 1, "2026_01_KC_LAC"],
        [2026, 2, "2026_02_BAL_CIN"],
    ]


def test_normalize_schedule_dataset_excludes_postseason_games():
    result = normalize_schedule_dataset(make_schedule_rows())

    assert "2026_20_KC_BUF" not in result["game_id"].tolist()


def test_normalize_schedule_dataset_validates_missing_source_columns():
    data = make_schedule_rows().drop(columns=["gameday", "home_team"])

    with pytest.raises(ValueError) as error:
        normalize_schedule_dataset(data)

    message = str(error.value)

    assert "NFL schedule source data is missing required columns" in message
    assert "gameday" in message
    assert "home_team" in message


def test_normalize_schedule_dataset_removes_identical_duplicates():
    data = make_schedule_rows()
    data = pd.concat([data, data.iloc[[1]]], ignore_index=True)

    result = normalize_schedule_dataset(data)

    assert len(result[result["game_id"] == "2026_01_KC_LAC"]) == 2


def test_normalize_schedule_dataset_coalesces_non_conflicting_duplicate_game_ids():
    data = make_schedule_rows()
    duplicate_row = data.iloc[[1]].copy()
    duplicate_row.loc[:, "ignored_column"] = "different ignored value"
    data = pd.concat([data, duplicate_row], ignore_index=True)

    result = normalize_schedule_dataset(data)

    assert len(result[result["game_id"] == "2026_01_KC_LAC"]) == 2


def test_normalize_schedule_dataset_rejects_conflicting_duplicate_game_ids():
    data = make_schedule_rows()
    conflicting_row = data.iloc[[1]].copy()
    conflicting_row.loc[:, "gametime"] = "21:00"
    data = pd.concat([data, conflicting_row], ignore_index=True)

    with pytest.raises(ValueError) as error:
        normalize_schedule_dataset(data)

    message = str(error.value)

    assert "Conflicting schedule records found" in message
    assert "2026_01_KC_LAC" in message


@pytest.mark.parametrize("team_column", ["home_team", "away_team"])
def test_normalize_schedule_dataset_rejects_missing_teams(team_column):
    data = make_schedule_rows()
    data.loc[1, team_column] = None

    with pytest.raises(ValueError) as error:
        normalize_schedule_dataset(data)

    message = str(error.value)

    assert "missing home_team or away_team" in message
    assert "2026_01_KC_LAC" in message


def test_normalize_schedule_dataset_rejects_blank_teams():
    data = make_schedule_rows()
    data.loc[1, "away_team"] = " "

    with pytest.raises(ValueError) as error:
        normalize_schedule_dataset(data)

    assert "missing home_team or away_team" in str(error.value)


def test_normalize_schedule_dataset_rejects_same_home_and_away_team():
    data = make_schedule_rows()
    data.loc[1, "away_team"] = "LAC"

    with pytest.raises(ValueError) as error:
        normalize_schedule_dataset(data)

    message = str(error.value)

    assert "same home_team and away_team" in message
    assert "2026_01_KC_LAC" in message


def test_normalize_schedule_dataset_uses_deterministic_ordering():
    result = normalize_schedule_dataset(make_schedule_rows())

    assert result[["season", "week", "game_id", "home_away"]].values.tolist() == [
        [2025, 18, "2025_18_BUF_MIA", "home"],
        [2025, 18, "2025_18_BUF_MIA", "away"],
        [2026, 1, "2026_01_KC_LAC", "home"],
        [2026, 1, "2026_01_KC_LAC", "away"],
        [2026, 2, "2026_02_BAL_CIN", "home"],
        [2026, 2, "2026_02_BAL_CIN", "away"],
    ]


def test_normalize_schedule_dataset_returns_empty_output_schema_without_regular_season():
    data = make_schedule_rows()
    data.loc[:, "game_type"] = "POST"

    result = normalize_schedule_dataset(data)

    assert result.empty
    assert result.columns.tolist() == OUTPUT_COLUMNS


def test_normalize_schedule_dataset_does_not_mutate_input():
    data = make_schedule_rows()
    original = data.copy(deep=True)

    normalize_schedule_dataset(data)

    pd.testing.assert_frame_equal(data, original)
