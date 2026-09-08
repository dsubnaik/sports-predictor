"""Annotate completed running-back game rows with same-game usage.

The usage totals, shares, and leader flags in this module are retrospective
descriptors of completed games. They are not valid direct inputs for predicting
that same game. Downstream predictive or report features must shift or filter
these rows to games strictly before the target report week.
"""

import pandas as pd

from football.data.build_running_back_dataset import (
    OUTPUT_COLUMNS,
    RUNNING_BACK_GAME_KEYS,
)


LOW_VOLUME_RB_OPPORTUNITY_THRESHOLD = 3

TEAM_GAME_KEYS = [
    "season",
    "week",
    "game_id",
    "team",
]

USAGE_COLUMNS = [
    "opportunities",
    "team_rb_rushing_attempts",
    "team_rb_targets",
    "team_rb_opportunities",
    "carry_share",
    "target_share",
    "opportunity_share",
    "lead_team_rusher",
    "lead_targeted_rb",
    "backfield_opportunity_leader",
    "tied_backfield_opportunity_leader",
    "low_volume_rb",
    "shared_backfield",
]

RUNNING_BACK_USAGE_COLUMNS = [
    *OUTPUT_COLUMNS,
    *USAGE_COLUMNS,
]

NUMERIC_USAGE_INPUT_COLUMNS = [
    "rushing_attempts",
    "targets",
]


def annotate_running_back_usage(data: pd.DataFrame) -> pd.DataFrame:
    """Return normalized RB rows annotated with retrospective usage context.

    The input must follow the normalized schema returned by
    ``build_running_back_dataset``. Every valid RB row is preserved, including
    zero-usage and low-volume backs.

    The appended totals, shares, and leader flags describe same-game backfield
    usage after the game is complete. Do not use them directly to predict that
    same game; leakage-safe downstream features must use only prior games, such
    as rows from weeks strictly before the target report week.
    """

    _validate_running_back_data(data)

    running_back_data = data.loc[:, OUTPUT_COLUMNS].copy()

    if running_back_data.empty:
        return pd.DataFrame(columns=RUNNING_BACK_USAGE_COLUMNS)

    running_back_data = _validated_unique_running_back_games(running_back_data)

    running_back_data["opportunities"] = (
        running_back_data["rushing_attempts"] + running_back_data["targets"]
    )

    # These are same-game team totals for completed historical games only.
    running_back_data["team_rb_rushing_attempts"] = running_back_data.groupby(
        TEAM_GAME_KEYS,
        sort=False,
    )["rushing_attempts"].transform("sum")
    running_back_data["team_rb_targets"] = running_back_data.groupby(
        TEAM_GAME_KEYS,
        sort=False,
    )["targets"].transform("sum")
    running_back_data["team_rb_opportunities"] = running_back_data.groupby(
        TEAM_GAME_KEYS,
        sort=False,
    )["opportunities"].transform("sum")

    running_back_data["carry_share"] = _safe_share(
        running_back_data["rushing_attempts"],
        running_back_data["team_rb_rushing_attempts"],
    )
    running_back_data["target_share"] = _safe_share(
        running_back_data["targets"],
        running_back_data["team_rb_targets"],
    )
    running_back_data["opportunity_share"] = _safe_share(
        running_back_data["opportunities"],
        running_back_data["team_rb_opportunities"],
    )

    max_carries = running_back_data.groupby(TEAM_GAME_KEYS, sort=False)[
        "rushing_attempts"
    ].transform("max")
    max_targets = running_back_data.groupby(TEAM_GAME_KEYS, sort=False)[
        "targets"
    ].transform("max")
    max_opportunities = running_back_data.groupby(TEAM_GAME_KEYS, sort=False)[
        "opportunities"
    ].transform("max")

    running_back_data["lead_team_rusher"] = (
        (running_back_data["rushing_attempts"] == max_carries)
        & (max_carries > 0)
    )
    running_back_data["lead_targeted_rb"] = (
        (running_back_data["targets"] == max_targets)
        & (max_targets > 0)
    )
    running_back_data["backfield_opportunity_leader"] = (
        (running_back_data["opportunities"] == max_opportunities)
        & (max_opportunities > 0)
    )

    top_opportunity_count = running_back_data.groupby(TEAM_GAME_KEYS, sort=False)[
        "backfield_opportunity_leader"
    ].transform("sum")
    running_back_data["tied_backfield_opportunity_leader"] = (
        running_back_data["backfield_opportunity_leader"]
        & (top_opportunity_count > 1)
    )

    running_back_data["_positive_opportunities"] = (
        running_back_data["opportunities"] > 0
    )
    positive_opportunity_count = running_back_data.groupby(TEAM_GAME_KEYS, sort=False)[
        "_positive_opportunities"
    ].transform("sum")
    running_back_data["low_volume_rb"] = (
        running_back_data["opportunities"] <= LOW_VOLUME_RB_OPPORTUNITY_THRESHOLD
    )
    running_back_data["shared_backfield"] = positive_opportunity_count >= 2

    running_back_data = running_back_data.drop(columns=["_positive_opportunities"])

    return running_back_data.sort_values(
        by=[
            "season",
            "week",
            "game_id",
            "team",
            "player_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True).loc[:, RUNNING_BACK_USAGE_COLUMNS]


def _safe_share(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Return finite shares, using 0.0 for zero-denominator rows."""

    shares = pd.Series(0.0, index=numerator.index)
    valid_denominator = denominator != 0
    shares.loc[valid_denominator] = (
        numerator.loc[valid_denominator] / denominator.loc[valid_denominator]
    )

    return shares


def _validated_unique_running_back_games(data: pd.DataFrame) -> pd.DataFrame:
    """Return unique RB-games or raise on conflicting duplicate keys."""

    unique_games = data.drop_duplicates().copy()
    conflicting_duplicates = unique_games[
        unique_games.duplicated(
            subset=RUNNING_BACK_GAME_KEYS,
            keep=False,
        )
    ]

    if not conflicting_duplicates.empty:
        conflicting_keys = (
            conflicting_duplicates.loc[:, RUNNING_BACK_GAME_KEYS]
            .drop_duplicates()
            .sort_values(
                by=RUNNING_BACK_GAME_KEYS,
                kind="mergesort",
            )
            .to_dict(orient="records")
        )

        raise ValueError(
            "Conflicting running-back game records found for keys: "
            f"{conflicting_keys}"
        )

    return unique_games


def _validate_running_back_data(data: pd.DataFrame) -> None:
    """Raise clear errors for invalid normalized running-back data."""

    missing_columns = set(OUTPUT_COLUMNS).difference(data.columns)

    if missing_columns:
        raise ValueError(
            "Running back data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if data.empty:
        return

    invalid_numeric_columns = [
        column
        for column in NUMERIC_USAGE_INPUT_COLUMNS
        if not pd.api.types.is_numeric_dtype(data[column])
    ]

    if invalid_numeric_columns:
        raise ValueError(
            "Running back usage columns must be numeric: "
            f"{sorted(invalid_numeric_columns)}"
        )

    missing_usage = data.loc[:, NUMERIC_USAGE_INPUT_COLUMNS].isna().any()
    missing_usage_columns = missing_usage[missing_usage].index.tolist()

    if missing_usage_columns:
        raise ValueError(
            "Running back usage columns cannot contain missing values: "
            f"{sorted(missing_usage_columns)}"
        )

    negative_usage = (data.loc[:, NUMERIC_USAGE_INPUT_COLUMNS] < 0).any()
    negative_usage_columns = negative_usage[negative_usage].index.tolist()

    if negative_usage_columns:
        raise ValueError(
            "Running back usage columns cannot contain negative values: "
            f"{sorted(negative_usage_columns)}"
        )
