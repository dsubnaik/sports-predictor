import pandas as pd
import pytest

from features.engineer import build_team_k_history


def test_build_team_k_history_uses_only_prior_games():
    pitch_rows = []

    # Team 10 is the away batting team.
    # Day 1: 2 strikeouts in 4 plate appearances.
    day_1_events = [
        "strikeout",
        "single",
        "strikeout",
        "walk",
    ]

    for at_bat_number, event in enumerate(day_1_events, start=1):
        pitch_rows.append(
            {
                "game_pk": 1,
                "game_date": "2026-04-01",
                "inning_topbot": "Top",
                "at_bat_number": at_bat_number,
                "events": event,
            }
        )

    # Day 2: 1 strikeout in 4 plate appearances.
    day_2_events = [
        "single",
        "strikeout",
        "walk",
        "groundout",
    ]

    for at_bat_number, event in enumerate(day_2_events, start=1):
        pitch_rows.append(
            {
                "game_pk": 2,
                "game_date": "2026-04-02",
                "inning_topbot": "Top",
                "at_bat_number": at_bat_number,
                "events": event,
            }
        )

    # Day 3 results should NOT affect Day 3's team_k_rate.
    day_3_events = [
        "strikeout",
        "strikeout",
        "strikeout",
        "strikeout",
    ]

    for at_bat_number, event in enumerate(day_3_events, start=1):
        pitch_rows.append(
            {
                "game_pk": 3,
                "game_date": "2026-04-03",
                "inning_topbot": "Top",
                "at_bat_number": at_bat_number,
                "events": event,
            }
        )

    pitch_data = pd.DataFrame(pitch_rows)

    starters = []

    for game_pk, game_date in [
        (1, "2026-04-01"),
        (2, "2026-04-02"),
        (3, "2026-04-03"),
    ]:
        starters.extend(
            [
                {
                    "game_pk": game_pk,
                    "game_date": game_date,
                    "pitcher_id": 100 + game_pk,
                    "pitcher_name": "Away Starter",
                    "home_away": "away",
                    "team_id": 10,
                    "team_name": "Away Team",
                    "opponent_id": 20,
                    "opponent_name": "Home Team",
                },
                {
                    "game_pk": game_pk,
                    "game_date": game_date,
                    "pitcher_id": 200 + game_pk,
                    "pitcher_name": "Home Starter",
                    "home_away": "home",
                    "team_id": 20,
                    "team_name": "Home Team",
                    "opponent_id": 10,
                    "opponent_name": "Away Team",
                },
            ]
        )

    result = build_team_k_history(
        pitch_data=pitch_data,
        starters=starters,
    )

    team_10 = result[
        result["batting_team_id"] == 10
    ].sort_values("game_date")

    # No previous games exist before Day 1.
    assert pd.isna(team_10.iloc[0]["team_k_rate"])

    # Before Day 2:
    # Day 1 = 2 strikeouts / 4 PA = 0.50
    assert team_10.iloc[1]["team_k_rate"] == pytest.approx(0.50)

    # Before Day 3:
    # Days 1 + 2 = 3 strikeouts / 8 PA = 0.375
    assert team_10.iloc[2]["team_k_rate"] == pytest.approx(0.375)