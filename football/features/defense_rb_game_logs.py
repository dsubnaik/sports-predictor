"""Build retrospective defense-game logs from opposing RB group production.

These logs describe completed historical games. They are not valid direct
features for predicting the same game. Downstream defense metrics must filter
them to games strictly before the target report week.
"""

import numpy as np
import pandas as pd

from football.data.build_running_back_dataset import RUNNING_BACK_GAME_KEYS
from football.features.running_back_usage import RUNNING_BACK_USAGE_COLUMNS


DEFENSE_RB_GAME_KEYS = [
    "season",
    "week",
    "game_id",
    "defense",
]

OFFENSE_RB_GAME_KEYS = [
    "season",
    "week",
    "game_id",
    "offense_team",
]

DEFENSE_RB_GAME_LOG_COLUMNS = [
    "season",
    "week",
    "game_id",
    "defense",
    "offense_team",
    "rb_rushing_attempts_allowed",
    "rb_rushing_yards_allowed",
    "rb_rushing_touchdowns_allowed",
    "rb_receptions_allowed",
    "rb_targets_allowed",
    "rb_receiving_yards_allowed",
    "rb_receiving_touchdowns_allowed",
    "rb_opportunities_allowed",
    "rb_players_used",
]

IDENTITY_COLUMNS = [
    "season",
    "week",
    "game_id",
    "player_id",
    "team",
    "opponent",
]

TEXT_IDENTITY_COLUMNS = [
    "game_id",
    "player_id",
    "team",
    "opponent",
]

NUMERIC_IDENTITY_COLUMNS = [
    "season",
    "week",
]

PRODUCTION_COLUMNS = [
    "rushing_attempts",
    "rushing_yards",
    "rushing_touchdowns",
    "receptions",
    "targets",
    "receiving_yards",
    "receiving_touchdowns",
    "opportunities",
]

NON_NEGATIVE_PRODUCTION_COLUMNS = [
    "rushing_attempts",
    "rushing_touchdowns",
    "receptions",
    "targets",
    "receiving_touchdowns",
    "opportunities",
]


def build_defense_rb_game_logs(annotated_rbs: pd.DataFrame) -> pd.DataFrame:
    """Return one retrospective RB-production row per defense and game.

    ``annotated_rbs`` must follow the complete schema returned by
    ``annotate_running_back_usage``. All represented opposing RBs are included
    in each defense-game total. The repeated ``team_rb_*`` usage columns are
    never summed. Each input ``opponent`` becomes ``defense``, and its ``team``
    becomes ``offense_team``.

    The result describes completed same-game production and must not be used
    directly to predict that game. Downstream predictive features must use
    only defense-game rows strictly before the target report week.
    """

    _validate_annotated_running_backs(annotated_rbs)

    running_backs = annotated_rbs.loc[:, RUNNING_BACK_USAGE_COLUMNS].copy()
    if running_backs.empty:
        return pd.DataFrame(columns=DEFENSE_RB_GAME_LOG_COLUMNS)

    running_backs = _validated_unique_running_back_games(running_backs)
    running_backs = running_backs.rename(
        columns={
            "opponent": "defense",
            "team": "offense_team",
        }
    )
    _validate_defense_game_context(running_backs)

    defense_logs = (
        running_backs.groupby(
            [*DEFENSE_RB_GAME_KEYS, "offense_team"],
            as_index=False,
            sort=False,
        )
        .agg(
            rb_rushing_attempts_allowed=("rushing_attempts", "sum"),
            rb_rushing_yards_allowed=("rushing_yards", "sum"),
            rb_rushing_touchdowns_allowed=("rushing_touchdowns", "sum"),
            rb_receptions_allowed=("receptions", "sum"),
            rb_targets_allowed=("targets", "sum"),
            rb_receiving_yards_allowed=("receiving_yards", "sum"),
            rb_receiving_touchdowns_allowed=("receiving_touchdowns", "sum"),
            rb_opportunities_allowed=("opportunities", "sum"),
            rb_players_used=("player_id", "nunique"),
        )
        .loc[:, DEFENSE_RB_GAME_LOG_COLUMNS]
    )

    return defense_logs.sort_values(
        by=DEFENSE_RB_GAME_KEYS,
        kind="mergesort",
    ).reset_index(drop=True)


def _validate_annotated_running_backs(data: pd.DataFrame) -> None:
    """Raise clear errors for invalid annotated running-back rows."""

    missing_columns = set(RUNNING_BACK_USAGE_COLUMNS).difference(data.columns)

    if missing_columns:
        raise ValueError(
            "Annotated running back data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if data.empty:
        return

    missing_identities = data.loc[:, IDENTITY_COLUMNS].isna().any()
    missing_identity_columns = missing_identities[
        missing_identities
    ].index.tolist()
    if missing_identity_columns:
        raise ValueError(
            "Defense RB identity columns cannot contain missing values: "
            f"{sorted(missing_identity_columns)}"
        )

    blank_identities = data.loc[:, TEXT_IDENTITY_COLUMNS].apply(
        lambda values: values.astype("string").str.strip().eq("").any()
    )
    blank_identity_columns = blank_identities[blank_identities].index.tolist()
    if blank_identity_columns:
        raise ValueError(
            "Defense RB identity columns cannot contain blank values: "
            f"{sorted(blank_identity_columns)}"
        )

    _validate_numeric_identity_columns(data)
    _validate_production_columns(data)

    invalid_positions = data["position"].ne("RB").fillna(True)
    if invalid_positions.any():
        raise ValueError(
            "Annotated running back data must contain only position == 'RB' rows"
        )

    expected_opportunities = data["rushing_attempts"] + data["targets"]
    opportunities_match = np.array_equal(
        data["opportunities"].to_numpy(dtype=float),
        expected_opportunities.to_numpy(dtype=float),
    )
    if not opportunities_match:
        raise ValueError(
            "Running back opportunities must equal rushing_attempts plus targets"
        )


def _validate_numeric_identity_columns(data: pd.DataFrame) -> None:
    """Require positive, finite whole-number season and week identities."""

    invalid_numeric_columns = [
        column
        for column in NUMERIC_IDENTITY_COLUMNS
        if (
            _contains_boolean(data[column])
            or not pd.api.types.is_numeric_dtype(data[column])
        )
    ]
    if invalid_numeric_columns:
        raise ValueError(
            "Defense RB season and week columns must be numeric: "
            f"{sorted(invalid_numeric_columns)}"
        )

    identity_values = data.loc[:, NUMERIC_IDENTITY_COLUMNS].to_numpy(dtype=float)
    non_finite_columns = data.loc[:, NUMERIC_IDENTITY_COLUMNS].columns[
        ~np.isfinite(identity_values).all(axis=0)
    ].tolist()
    if non_finite_columns:
        raise ValueError(
            "Defense RB season and week columns must contain finite values: "
            f"{sorted(non_finite_columns)}"
        )

    invalid_values = (
        (identity_values < 1)
        | (identity_values != np.floor(identity_values))
    ).any(axis=0)
    invalid_value_columns = data.loc[:, NUMERIC_IDENTITY_COLUMNS].columns[
        invalid_values
    ].tolist()
    if invalid_value_columns:
        raise ValueError(
            "Defense RB season and week columns must contain positive integers: "
            f"{sorted(invalid_value_columns)}"
        )


def _validate_production_columns(data: pd.DataFrame) -> None:
    """Validate individual RB production used by the aggregation."""

    invalid_numeric_columns = [
        column
        for column in PRODUCTION_COLUMNS
        if (
            _contains_boolean(data[column])
            or not pd.api.types.is_numeric_dtype(data[column])
        )
    ]
    if invalid_numeric_columns:
        raise ValueError(
            "Defense RB production columns must be numeric: "
            f"{sorted(invalid_numeric_columns)}"
        )

    missing_values = data.loc[:, PRODUCTION_COLUMNS].isna().any()
    missing_value_columns = missing_values[missing_values].index.tolist()
    if missing_value_columns:
        raise ValueError(
            "Defense RB production columns cannot contain missing values: "
            f"{sorted(missing_value_columns)}"
        )

    production_values = data.loc[:, PRODUCTION_COLUMNS].to_numpy(dtype=float)
    non_finite_columns = data.loc[:, PRODUCTION_COLUMNS].columns[
        ~np.isfinite(production_values).all(axis=0)
    ].tolist()
    if non_finite_columns:
        raise ValueError(
            "Defense RB production columns must contain finite values: "
            f"{sorted(non_finite_columns)}"
        )

    negative_values = (
        data.loc[:, NON_NEGATIVE_PRODUCTION_COLUMNS] < 0
    ).any()
    negative_value_columns = negative_values[negative_values].index.tolist()
    if negative_value_columns:
        raise ValueError(
            "Defense RB count columns cannot contain negative values: "
            f"{sorted(negative_value_columns)}"
        )


def _contains_boolean(values: pd.Series) -> bool:
    """Return whether a numeric-contract series contains boolean values."""

    return bool(
        values.map(lambda value: isinstance(value, (bool, np.bool_))).any()
    )


def _validated_unique_running_back_games(data: pd.DataFrame) -> pd.DataFrame:
    """Return exact-unique RB games or reject conflicting player-game rows."""

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


def _validate_defense_game_context(data: pd.DataFrame) -> None:
    """Require one-to-one defense and offense_team game mappings."""

    offense_counts = data.groupby(
        DEFENSE_RB_GAME_KEYS,
        sort=False,
    )["offense_team"].nunique()
    conflicting_defense_keys = offense_counts[offense_counts > 1]

    if not conflicting_defense_keys.empty:
        conflicts = (
            conflicting_defense_keys.index.to_frame(index=False)
            .sort_values(by=DEFENSE_RB_GAME_KEYS, kind="mergesort")
            .to_dict(orient="records")
        )
        raise ValueError(
            "Conflicting defense-game records found for keys: "
            f"{conflicts}"
        )

    defense_counts = data.groupby(
        OFFENSE_RB_GAME_KEYS,
        sort=False,
    )["defense"].nunique()
    conflicting_offense_keys = defense_counts[defense_counts > 1]

    if not conflicting_offense_keys.empty:
        conflicts = (
            conflicting_offense_keys.index.to_frame(index=False)
            .sort_values(by=OFFENSE_RB_GAME_KEYS, kind="mergesort")
            .to_dict(orient="records")
        )
        raise ValueError(
            "Conflicting offense-game records found for keys: "
            f"{conflicts}"
        )
