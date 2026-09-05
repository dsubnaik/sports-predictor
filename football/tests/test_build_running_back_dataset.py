import pandas as pd
import pytest

from football.data.build_running_back_dataset import (
    OUTPUT_COLUMNS,
    build_running_back_dataset,
)


def make_player_stats_source() -> pd.DataFrame:
    """Create unsorted nflverse-style player stat rows."""

    return pd.DataFrame(
        {
            "season": [2026, 2026, 2026, 2026, 2025, 2026, 2026],
            "week": [2, 1, 1, 1, 18, 20, 1],
            "season_type": ["REG", "REG", "REG", "REG", "REG", "POST", "REG"],
            "game_id": [
                "2026_02_BAL_CIN",
                "2026_01_KC_LAC",
                "2026_01_KC_LAC",
                "2026_01_KC_LAC",
                "2025_18_BUF_MIA",
                "2026_20_KC_BUF",
                "2026_01_KC_LAC",
            ],
            "player_id": [
                "rb_bal",
                "rb_kc_1",
                "rb_kc_2",
                "wr_kc",
                "rb_buf",
                "rb_post",
                "fb_kc",
            ],
            "player_display_name": [
                "Baltimore RB",
                "Kansas City RB One",
                "Kansas City RB Two",
                "Kansas City WR",
                "Buffalo RB",
                "Postseason RB",
                "Kansas City FB",
            ],
            "position": ["RB", "RB", "RB", "WR", "RB", "RB", "FB"],
            "team": ["BAL", "KC", "KC", "KC", "BUF", "KC", "KC"],
            "opponent_team": ["CIN", "LAC", "LAC", "LAC", "MIA", "BUF", "LAC"],
            "carries": [16, 18, 0, 2, 12, 20, 5],
            "rushing_yards": [91, 84, 0, 14, 73, 101, 21],
            "rushing_tds": [1, 1, 0, 0, 0, 2, 0],
            "receptions": [2, 4, 3, 5, 1, 3, 1],
            "targets": [3, 5, 4, 7, 2, 4, 1],
            "receiving_yards": [17, 36, 22, 44, 9, 28, 6],
            "receiving_tds": [0, 1, 0, 1, 0, 0, 0],
            "ignored_column": ["a", "b", "c", "d", "e", "f", "g"],
        }
    )


def test_build_running_back_dataset_maps_normal_rb_row_to_expected_schema():
    result = build_running_back_dataset(make_player_stats_source())

    assert result.columns.tolist() == OUTPUT_COLUMNS
    assert result.iloc[0].to_dict() == {
        "season": 2025,
        "week": 18,
        "game_id": "2025_18_BUF_MIA",
        "player_id": "rb_buf",
        "player_name": "Buffalo RB",
        "team": "BUF",
        "opponent": "MIA",
        "position": "RB",
        "rushing_attempts": 12,
        "rushing_yards": 73,
        "rushing_touchdowns": 0,
        "receptions": 1,
        "targets": 2,
        "receiving_yards": 9,
        "receiving_touchdowns": 0,
    }


def test_build_running_back_dataset_keeps_multiple_rbs_for_same_team_game():
    result = build_running_back_dataset(make_player_stats_source())
    game_rows = result[
        (result["game_id"] == "2026_01_KC_LAC")
        & (result["team"] == "KC")
    ]

    assert game_rows["player_id"].tolist() == ["rb_kc_1", "rb_kc_2"]


def test_build_running_back_dataset_preserves_rb_with_zero_rushing_attempts():
    result = build_running_back_dataset(make_player_stats_source())

    zero_attempt_row = result[result["player_id"] == "rb_kc_2"].iloc[0]

    assert zero_attempt_row["rushing_attempts"] == 0
    assert zero_attempt_row["receiving_yards"] == 22


def test_build_running_back_dataset_excludes_non_rb_with_rushing_attempts():
    result = build_running_back_dataset(make_player_stats_source())

    assert "wr_kc" not in result["player_id"].tolist()


def test_build_running_back_dataset_uses_supported_rb_position_only():
    result = build_running_back_dataset(make_player_stats_source())

    assert set(result["position"]) == {"RB"}
    assert "fb_kc" not in result["player_id"].tolist()


def test_build_running_back_dataset_preserves_receiving_fields():
    result = build_running_back_dataset(make_player_stats_source())
    row = result[result["player_id"] == "rb_kc_1"].iloc[0]

    assert row["receptions"] == 4
    assert row["targets"] == 5
    assert row["receiving_yards"] == 36
    assert row["receiving_touchdowns"] == 1


def test_build_running_back_dataset_excludes_non_regular_season_rows():
    result = build_running_back_dataset(make_player_stats_source())

    assert "rb_post" not in result["player_id"].tolist()


def test_build_running_back_dataset_removes_identical_duplicates():
    data = make_player_stats_source()
    data = pd.concat([data, data.iloc[[1]]], ignore_index=True)

    result = build_running_back_dataset(data)

    assert result["player_id"].tolist().count("rb_kc_1") == 1


def test_build_running_back_dataset_rejects_conflicting_duplicates():
    data = make_player_stats_source()
    conflicting_row = data.iloc[[1]].copy()
    conflicting_row.loc[:, "rushing_yards"] = 99
    data = pd.concat([data, conflicting_row], ignore_index=True)

    with pytest.raises(ValueError) as error:
        build_running_back_dataset(data)

    message = str(error.value)

    assert "Conflicting running-back game records found" in message
    assert "2026_01_KC_LAC" in message
    assert "rb_kc_1" in message


def test_build_running_back_dataset_does_not_aggregate_conflicting_duplicates():
    data = make_player_stats_source()
    conflicting_row = data.iloc[[1]].copy()
    conflicting_row.loc[:, "carries"] = 20
    data = pd.concat([data, conflicting_row], ignore_index=True)

    with pytest.raises(ValueError):
        build_running_back_dataset(data)


def test_build_running_back_dataset_checks_duplicates_only_after_rb_filtering():
    data = make_player_stats_source()
    conflicting_non_rb = data.iloc[[3]].copy()
    conflicting_non_rb.loc[:, "rushing_yards"] = 99
    conflicting_post = data.iloc[[5]].copy()
    conflicting_post.loc[:, "rushing_yards"] = 130
    data = pd.concat(
        [data, conflicting_non_rb, conflicting_post],
        ignore_index=True,
    )

    result = build_running_back_dataset(data)

    assert result["player_id"].tolist() == [
        "rb_buf",
        "rb_kc_1",
        "rb_kc_2",
        "rb_bal",
    ]


def test_build_running_back_dataset_does_not_mutate_input():
    data = make_player_stats_source()
    original = data.copy(deep=True)

    build_running_back_dataset(data)

    pd.testing.assert_frame_equal(data, original)


def test_build_running_back_dataset_uses_deterministic_ordering():
    data = make_player_stats_source().sample(frac=1, random_state=42)

    result = build_running_back_dataset(data)

    assert result[
        ["season", "week", "game_id", "team", "player_id"]
    ].values.tolist() == [
        [2025, 18, "2025_18_BUF_MIA", "BUF", "rb_buf"],
        [2026, 1, "2026_01_KC_LAC", "KC", "rb_kc_1"],
        [2026, 1, "2026_01_KC_LAC", "KC", "rb_kc_2"],
        [2026, 2, "2026_02_BAL_CIN", "BAL", "rb_bal"],
    ]


def test_build_running_back_dataset_returns_empty_schema_without_rbs():
    data = make_player_stats_source()
    data.loc[:, "position"] = "WR"

    result = build_running_back_dataset(data)

    assert result.empty
    assert result.columns.tolist() == OUTPUT_COLUMNS


def test_build_running_back_dataset_returns_empty_schema_without_regular_season_rbs():
    data = make_player_stats_source()
    data.loc[:, "season_type"] = "POST"

    result = build_running_back_dataset(data)

    assert result.empty
    assert result.columns.tolist() == OUTPUT_COLUMNS


def test_build_running_back_dataset_validates_missing_source_columns():
    data = make_player_stats_source().drop(
        columns=[
            "carries",
            "receiving_tds",
        ]
    )

    with pytest.raises(ValueError) as error:
        build_running_back_dataset(data)

    message = str(error.value)

    assert "NFL player stats source data is missing required columns" in message
    assert "carries" in message
    assert "receiving_tds" in message


def test_build_running_back_dataset_validates_before_selecting_or_renaming():
    data = pd.DataFrame(columns=[])

    with pytest.raises(ValueError) as error:
        build_running_back_dataset(data)

    message = str(error.value)

    assert "NFL player stats source data is missing required columns" in message
    assert "player_display_name" in message
    assert "opponent_team" in message


def test_build_running_back_dataset_handles_entirely_empty_input():
    data = make_player_stats_source().iloc[0:0].copy()

    result = build_running_back_dataset(data)

    assert result.empty
    assert result.columns.tolist() == OUTPUT_COLUMNS
