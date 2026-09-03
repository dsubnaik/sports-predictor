import pandas as pd
import pytest

from football.data.build_quarterback_dataset import OUTPUT_COLUMNS
from football.data.fetch_nflverse import (
    load_depth_charts,
    load_player_game_stats,
    load_schedules,
    normalize_quarterback_game_stats,
)


class FakePolarsFrame:
    def __init__(self, data):
        self.data = data
        self.to_pandas_called = False

    def to_pandas(self):
        self.to_pandas_called = True
        return self.data.copy()


def make_player_stats_source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2026, 2026, 2026],
            "week": [1, 1, 20],
            "season_type": ["REG", "REG", "POST"],
            "game_id": [
                "2026_01_KC_LAC",
                "2026_01_KC_LAC",
                "2026_20_KC_BUF",
            ],
            "player_id": ["qb_kc", "rb_kc", "qb_kc"],
            "player_display_name": [
                "Kansas City QB",
                "Kansas City RB",
                "Kansas City QB",
            ],
            "position": ["QB", "RB", "QB"],
            "team": ["KC", "KC", "KC"],
            "opponent_team": ["LAC", "LAC", "BUF"],
            "attempts": [35, 0, 31],
            "completions": [24, 0, 21],
            "passing_yards": [285, 0, 244],
            "passing_tds": [3, 0, 2],
            "passing_interceptions": [0, 0, 1],
            "ignored_column": ["keep", "out", "post"],
        }
    )


def test_normalize_quarterback_game_stats_maps_source_columns_to_contract():
    result = normalize_quarterback_game_stats(make_player_stats_source())

    assert result.columns.tolist() == OUTPUT_COLUMNS
    assert result.iloc[0].to_dict() == {
        "season": 2026,
        "week": 1,
        "game_id": "2026_01_KC_LAC",
        "player_id": "qb_kc",
        "player_name": "Kansas City QB",
        "team": "KC",
        "opponent": "LAC",
        "passing_attempts": 35,
        "completions": 24,
        "passing_yards": 285,
        "passing_touchdowns": 3,
        "interceptions": 0,
    }


def test_normalize_quarterback_game_stats_excludes_non_quarterbacks():
    result = normalize_quarterback_game_stats(make_player_stats_source())

    assert "rb_kc" not in result["player_id"].tolist()


def test_normalize_quarterback_game_stats_excludes_postseason_rows():
    result = normalize_quarterback_game_stats(make_player_stats_source())

    assert result["week"].tolist() == [1]


def test_normalize_quarterback_game_stats_validates_missing_source_columns():
    data = make_player_stats_source().drop(
        columns=["passing_interceptions", "opponent_team"]
    )

    with pytest.raises(ValueError) as error:
        normalize_quarterback_game_stats(data)

    message = str(error.value)

    assert "NFL player stats source data is missing required columns" in message
    assert "opponent_team" in message
    assert "passing_interceptions" in message


def test_load_player_game_stats_converts_polars_to_pandas_at_boundary():
    fake_polars = FakePolarsFrame(make_player_stats_source())

    def loader(**kwargs):
        return fake_polars

    result = load_player_game_stats([2026], loader=loader)

    assert fake_polars.to_pandas_called
    assert isinstance(result, pd.DataFrame)


def test_load_player_game_stats_passes_requested_seasons_to_loader():
    calls = []

    def loader(**kwargs):
        calls.append(kwargs)
        return FakePolarsFrame(make_player_stats_source())

    load_player_game_stats([2025, 2026], loader=loader)

    assert calls == [{"seasons": [2025, 2026], "summary_level": "week"}]


def test_load_schedules_passes_requested_seasons_to_loader():
    calls = []
    schedules = pd.DataFrame({"season": [2026], "game_id": ["2026_01_KC_LAC"]})

    def loader(**kwargs):
        calls.append(kwargs)
        return FakePolarsFrame(schedules)

    result = load_schedules([2025, 2026], loader=loader)

    assert calls == [{"seasons": [2025, 2026]}]
    assert isinstance(result, pd.DataFrame)


def test_load_depth_charts_passes_requested_seasons_to_loader():
    calls = []
    depth_charts = pd.DataFrame({"season": [2026], "team": ["KC"]})

    def loader(**kwargs):
        calls.append(kwargs)
        return FakePolarsFrame(depth_charts)

    result = load_depth_charts([2025, 2026], loader=loader)

    assert calls == [{"seasons": [2025, 2026]}]
    assert isinstance(result, pd.DataFrame)


def test_normalize_quarterback_game_stats_does_not_mutate_input():
    data = make_player_stats_source()
    original = data.copy(deep=True)

    normalize_quarterback_game_stats(data)

    pd.testing.assert_frame_equal(data, original)
