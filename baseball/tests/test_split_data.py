import pandas as pd

from baseball.training.split_data import chronological_split


def test_chronological_split():
    data = pd.DataFrame(
        {
            "game_date": [
                "2026-07-20",
                "2026-06-15",
                "2026-07-05",
                "2026-05-01",
            ],
            "strikeouts": [7, 5, 6, 4],
        }
    )

    train_data, validation_data, test_data = chronological_split(
        data,
        train_end_date="2026-06-30",
        validation_end_date="2026-07-14",
    )

    assert len(train_data) == 2
    assert len(validation_data) == 1
    assert len(test_data) == 1

    assert train_data["game_date"].max() <= pd.Timestamp("2026-06-30")
    assert validation_data["game_date"].min() > pd.Timestamp("2026-06-30")
    assert validation_data["game_date"].max() <= pd.Timestamp("2026-07-14")
    assert test_data["game_date"].min() > pd.Timestamp("2026-07-14")
