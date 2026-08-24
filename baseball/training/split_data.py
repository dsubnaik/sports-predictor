"""Functions for splitting model data chronologically."""

import pandas as pd


def chronological_split(data, train_end_date, validation_end_date):
    """
    Split pitcher-start data into chronological train, validation,
    and test sets.
    """
    data = data.copy()

    data["game_date"] = pd.to_datetime(data["game_date"])
    train_end_date = pd.to_datetime(train_end_date)
    validation_end_date = pd.to_datetime(validation_end_date)

    if train_end_date >= validation_end_date:
        raise ValueError(
            "train_end_date must be earlier than validation_end_date."
        )

    data = data.sort_values("game_date")

    train_data = data[
        data["game_date"] <= train_end_date
    ]

    validation_data = data[
        (data["game_date"] > train_end_date)
        & (data["game_date"] <= validation_end_date)
    ]

    test_data = data[
        data["game_date"] > validation_end_date
    ]

    return train_data, validation_data, test_data