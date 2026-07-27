"""Tests for aggregating pitch-level Statcast data into pitcher starts."""

import pandas as pd

from data.fetch_statcast import aggregate_to_starts


def test_aggregate_to_starts_keeps_two_starters_in_same_game_separate():
    pitch_data = pd.DataFrame(
        {
            "game_pk": [123, 123, 123, 123],
            "game_date": ["2026-04-01"] * 4,
            "pitcher": [100, 100, 200, 200],
            "player_name": [
                "Away Starter",
                "Away Starter",
                "Home Starter",
                "Home Starter",
            ],
            "events": [None, "strikeout", "strikeout", "strikeout"],
            "description": [
                "ball",
                "swinging_strike",
                "called_strike",
                "swinging_strike_blocked",
            ],
            "release_speed": [94.0, 96.0, 90.0, 92.0],
            "release_spin_rate": [2200.0, 2400.0, 2100.0, 2300.0],
        }
    )

    result = aggregate_to_starts(pitch_data)

    assert len(result) == 2
    assert result["pitcher_id"].tolist() == [100, 200]

    away_starter = result.loc[result["pitcher_id"] == 100].iloc[0]
    assert away_starter["strikeouts"] == 1
    assert away_starter["swinging_strikes"] == 1
    assert away_starter["total_pitches"] == 2
    assert away_starter["avg_velocity"] == 95.0
    assert away_starter["avg_spin_rate"] == 2300.0
    assert away_starter["swstr_pct"] == 0.5

    home_starter = result.loc[result["pitcher_id"] == 200].iloc[0]
    assert home_starter["strikeouts"] == 2
    assert home_starter["swinging_strikes"] == 1
    assert home_starter["total_pitches"] == 2
    assert home_starter["avg_velocity"] == 91.0
    assert home_starter["avg_spin_rate"] == 2200.0
    assert home_starter["swstr_pct"] == 0.5