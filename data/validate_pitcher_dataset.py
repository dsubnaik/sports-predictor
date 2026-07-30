"""Validate the processed official starter training dataset."""

from pathlib import Path

import pandas as pd


DATASET_PATH = Path(
    "data/processed/pitcher_training_2026.csv"
)

REQUIRED_COLUMNS = {
    "game_pk",
    "pitcher_id",
    "game_date",
    "pitcher_name",
    "strikeouts",
    "swinging_strikes",
    "total_pitches",
    "avg_velocity",
    "swstr_pct",
    "rolling_k",
    "rolling_swstr",
    "rolling_velocity",
    "rolling_pitches",
}


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the processed pitcher training dataset."""

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    data = pd.read_csv(path)

    return data


def validate_required_columns(data: pd.DataFrame) -> None:
    """Raise an error when required columns are missing."""

    data_columns = set(data.columns)

    missing_columns = REQUIRED_COLUMNS.difference(data_columns)

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )


def validate_unique_starts(data: pd.DataFrame) -> None:
    """Ensure each pitcher has only one row per game."""

    duplicate_starts = data[
        data.duplicated(
            subset=["game_pk", "pitcher_id"],
            keep=False,
        )
    ]

    if not duplicate_starts.empty:
        raise ValueError(
            f"Found {len(duplicate_starts)} duplicate game/pitcher rows."
        )


def validate_missing_values(data: pd.DataFrame) -> None:
    """Ensure model features contain no missing values."""

    rolling_columns = [
        "rolling_k",
        "rolling_swstr",
        "rolling_velocity",
        "rolling_pitches",
    ]

    # The first sum counts missing values in each column.
    # The second sum adds the column totals together.
    missing_values = (
        data[rolling_columns]
        .isna()
        .sum()
        .sum()
    )

    if missing_values:
        raise ValueError(
            f"Found {missing_values} missing values in rolling columns."
        )


def summarize_dataset(data: pd.DataFrame) -> None:
    """Print a summary of the pitcher dataset."""

    print("====================================")
    print("Pitcher Dataset Summary")
    print("====================================")

    print(f"Rows: {len(data)}")
    print(f"Unique Pitchers: {data['pitcher_id'].nunique()}")
    print(f"Unique Games: {data['game_pk'].nunique()}")
    print(f"Earliest Game: {data['game_date'].min()}")
    print(f"Latest Game: {data['game_date'].max()}")
    print(f"Average Strikeouts: {data['strikeouts'].mean()}")
    print(f"Minimum Strikeouts: {data['strikeouts'].min()}")
    print(f"Maximum Strikeouts: {data['strikeouts'].max()}")
    print(
        "Average Pitches per Start: "
        f"{data['total_pitches'].mean()}"
    )
    print(f"Average Rolling K: {data['rolling_k'].mean()}")


def main() -> None:
    """Load, validate, and summarize the pitcher dataset."""

    data = load_dataset(DATASET_PATH)

    validate_required_columns(data)
    validate_unique_starts(data)
    validate_missing_values(data)

    summarize_dataset(data)

    print("Dataset validation passed.")


if __name__ == "__main__":
    main()