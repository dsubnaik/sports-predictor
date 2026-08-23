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
            "home_away": [
                "away",
                "away",
                "home",
                "home",
            ],
            "team_id": [
                10,
                10,
                20,
                20,
            ],
            "team_name": [
                "Away Team",
                "Away Team",
                "Home Team",
                "Home Team",
            ],
            "opponent_id": [
                20,
                20,
                10,
                10,
            ],
            "opponent_name": [
                "Home Team",
                "Home Team",
                "Away Team",
                "Away Team",
            ],
            "events": [
                None,
                "strikeout",
                "strikeout",
                "strikeout",
            ],
            "description": [
                "ball",
                "swinging_strike",
                "called_strike",
                "swinging_strike_blocked",
            ],
            "release_speed": [
                94.0,
                96.0,
                90.0,
                92.0,
            ],
            "release_spin_rate": [
                2200.0,
                2400.0,
                2100.0,
                2300.0,
            ],
        }
    )