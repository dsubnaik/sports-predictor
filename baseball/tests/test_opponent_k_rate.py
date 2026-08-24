import pandas as pd
import pytest

from baseball.features.engineer import add_opponent_k_rate


def test_add_opponent_k_rate_matches_correct_opponent():
    start_data = pd.DataFrame(
        {
            "game_pk": [1, 1],
            "game_date": pd.to_datetime(
                [
                    "2026-04-03",
                    "2026-04-03",
                ]
            ),
            "pitcher_id": [100, 200],
            "opponent_id": [20, 10],
        }
    )

    team_k_history = pd.DataFrame(
        {
            "batting_team_id": [10, 20],
            "game_date": pd.to_datetime(
                [
                    "2026-04-03",
                    "2026-04-03",
                ]
            ),
            "team_k_rate": [0.20, 0.30],
        }
    )

    result = add_opponent_k_rate(
        start_data=start_data,
        team_k_history=team_k_history,
    )

    pitcher_100 = result[
        result["pitcher_id"] == 100
    ].iloc[0]

    pitcher_200 = result[
        result["pitcher_id"] == 200
    ].iloc[0]

    assert pitcher_100["opponent_k_rate"] == pytest.approx(0.30)
    assert pitcher_200["opponent_k_rate"] == pytest.approx(0.20)