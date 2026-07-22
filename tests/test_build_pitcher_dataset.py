"""Tests for building the MLB pitcher training dataset."""

import pandas as pd

from data.build_pitcher_dataset import filter_to_official_starts


def test_filter_to_official_starts():
    pitch_data = pd.DataFrame(
        {
            "game_pk": [1, 1, 1, 2],
            "pitcher": [100, 100, 200, 100],
            "description": [
                "called_strike",
                "ball",
                "swinging_strike",
                "foul",
            ],
        }
    )

    starters = [
        {
            "game_pk": 1,
            "game_date": "2026-04-01",
            "pitcher_id": 100,
            "pitcher_name": "Official Starter",
            "home_away": "away",
        }
    ]

    result = filter_to_official_starts(pitch_data, starters)

    assert len(result) == 2
    assert result["game_pk"].tolist() == [1, 1]
    assert result["pitcher"].tolist() == [100, 100]