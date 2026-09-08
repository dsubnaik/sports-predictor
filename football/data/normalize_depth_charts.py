"""Normalize current nflverse depth charts for RB participant resolution.

The canonical dataset defined here is intentionally scoped to expected
running-back participant resolution. It preserves every position so a
resolver can observe complete snapshots, but only RB player identities and
depth metadata are required to be complete.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


NFLVERSE_DATED_DEPTH_CHART_COLUMNS = [
    "dt",
    "team",
    "player_name",
    "gsis_id",
    "pos_abb",
    "pos_slot",
    "pos_rank",
]

CANONICAL_DEPTH_CHART_COLUMNS = [
    "team",
    "player_id",
    "player_name",
    "position",
    "snapshot_timestamp",
    "depth_position",
    "depth_rank",
]

_CURRENT_SCHEMA_MARKERS = {"dt", "pos_abb", "pos_slot", "pos_rank"}
_CANONICAL_SCHEMA_MARKERS = {"snapshot_timestamp", "player_id"}
_LEGACY_SCHEMA_MARKERS = {
    "season",
    "week",
    "game_type",
    "season_type",
    "club_code",
    "depth_team",
    "full_name",
}


def normalize_nflverse_depth_charts(
    depth_charts: pd.DataFrame,
) -> pd.DataFrame:
    """Convert one 2025+ nflverse dated frame to the RB canonical schema.

    Frames containing populated rows from current, canonical, or legacy
    schema families at the same time are rejected. Normalize each source
    family separately before concatenating canonical results.
    """

    if not isinstance(depth_charts, pd.DataFrame):
        raise TypeError("NFL depth-chart data must be a pandas DataFrame")

    _reject_duplicate_columns(depth_charts, "NFL depth-chart data")
    _reject_mixed_source_schemas(depth_charts)
    _require_columns(
        depth_charts,
        NFLVERSE_DATED_DEPTH_CHART_COLUMNS,
        "2025+ nflverse depth-chart data",
    )

    normalized = depth_charts.loc[
        :, NFLVERSE_DATED_DEPTH_CHART_COLUMNS
    ].rename(
        columns={
            "dt": "snapshot_timestamp",
            "gsis_id": "player_id",
            "pos_abb": "position",
            "pos_slot": "depth_position",
            "pos_rank": "depth_rank",
        }
    ).copy()

    if normalized.empty:
        return pd.DataFrame(columns=CANONICAL_DEPTH_CHART_COLUMNS)

    _validate_text(normalized, "team", "NFL depth-chart data")
    _validate_text(normalized, "position", "NFL depth-chart data")
    normalized["snapshot_timestamp"] = _parse_timestamps(
        normalized["snapshot_timestamp"],
        "NFL depth-chart data",
    )

    rb_mask = normalized["position"].eq("RB")
    rbs = normalized.loc[rb_mask].copy()
    _validate_text(rbs, "player_id", "NFL RB depth-chart data")
    _validate_text(rbs, "player_name", "NFL RB depth-chart data")
    rb_ranks = _positive_whole_numbers(
        rbs["depth_rank"],
        "NFL RB depth-chart depth_rank",
        allow_missing=True,
    )
    rb_positions = _nullable_categorical_text(
        rbs["depth_position"],
        "NFL RB depth-chart depth_position",
    )
    normalized["depth_rank"] = _coerce_nullable_positive_whole_numbers(
        normalized["depth_rank"]
    )
    normalized["depth_position"] = _coerce_nullable_categorical_text(
        normalized["depth_position"]
    )
    normalized.loc[rb_mask, "depth_rank"] = rb_ranks.tolist()
    normalized.loc[rb_mask, "depth_position"] = rb_positions.tolist()

    normalized = normalized.drop_duplicates(
        subset=CANONICAL_DEPTH_CHART_COLUMNS
    ).copy()
    _reject_conflicting_rb_snapshot_rows(normalized)

    normalized["_rank_sort"] = pd.to_numeric(
        normalized["depth_rank"], errors="coerce"
    )
    normalized["_position_sort"] = (
        normalized["depth_position"].astype("string").fillna("")
    )
    normalized["_player_id_sort"] = (
        normalized["player_id"].astype("string").fillna("")
    )
    normalized["_player_name_sort"] = (
        normalized["player_name"].astype("string").fillna("")
    )
    normalized = normalized.sort_values(
        [
            "team",
            "snapshot_timestamp",
            "position",
            "_rank_sort",
            "_position_sort",
            "_player_id_sort",
            "_player_name_sort",
        ],
        kind="mergesort",
        na_position="last",
    )

    return normalized.loc[:, CANONICAL_DEPTH_CHART_COLUMNS].reset_index(
        drop=True
    )


def _reject_mixed_source_schemas(data: pd.DataFrame) -> None:
    families = []
    if _has_populated_columns(data, _CURRENT_SCHEMA_MARKERS):
        families.append("2025+ nflverse dated")
    if _has_populated_columns(data, _CANONICAL_SCHEMA_MARKERS):
        families.append("canonical")
    if _has_populated_columns(data, _LEGACY_SCHEMA_MARKERS):
        families.append("legacy weekly")

    if len(families) > 1:
        raise ValueError(
            "Mixed depth-chart schemas detected "
            f"({', '.join(families)}); normalize each source schema "
            "separately before concatenation"
        )

    if families and families[0] != "2025+ nflverse dated":
        raise ValueError(
            "Unsupported depth-chart schema. This normalizer accepts only "
            "the 2025+ nflverse dated schema"
        )


def _has_populated_columns(data: pd.DataFrame, columns: set[str]) -> bool:
    present = sorted(columns.intersection(data.columns))
    return bool(present) and bool(data.loc[:, present].notna().any().any())


def _reject_conflicting_rb_snapshot_rows(data: pd.DataFrame) -> None:
    rbs = data.loc[data["position"].eq("RB")].copy()
    if rbs.empty:
        return

    identity = [
        "team",
        "snapshot_timestamp",
        "player_id",
        "position",
        "depth_position",
    ]
    conflicts = rbs.loc[rbs.duplicated(identity, keep=False)]
    if conflicts.empty:
        return

    keys = (
        conflicts.loc[:, identity]
        .drop_duplicates()
        .assign(
            depth_position=lambda values: values["depth_position"].astype(
                "string"
            )
        )
        .sort_values(identity, kind="mergesort", na_position="last")
        .to_dict("records")
    )
    raise ValueError(
        "Conflicting NFL RB depth-chart rows found for participant snapshots: "
        f"{keys}"
    )


def _parse_timestamps(values: pd.Series, label: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", format="mixed", utc=True)
    invalid = parsed.isna() & values.notna()
    if invalid.any():
        invalid_values = (
            values.loc[invalid].astype(str).drop_duplicates().sort_values().tolist()
        )
        raise ValueError(
            f"{label} contains invalid snapshot timestamps: {invalid_values}"
        )
    if parsed.isna().any():
        raise ValueError(f"{label} contains missing snapshot timestamps")
    return parsed


def _positive_whole_numbers(
    values: pd.Series,
    label: str,
    *,
    allow_missing: bool,
) -> pd.Series:
    booleans = values.map(lambda value: isinstance(value, (bool, np.bool_)))
    converted = pd.to_numeric(values, errors="coerce")
    invalid_missing = converted.isna() & ~values.isna()
    invalid_required = converted.isna() if not allow_missing else False
    finite = pd.Series(
        np.isfinite(converted.fillna(0).astype(float)), index=values.index
    )
    invalid = (
        booleans
        | invalid_missing
        | invalid_required
        | ~finite
        | (converted.notna() & (converted.lt(1) | converted.mod(1).ne(0)))
    )
    if invalid.any():
        suffix = " or missing values" if allow_missing else ""
        raise ValueError(f"{label} must contain positive whole numbers{suffix}")
    return converted.astype("Int64")


def _nullable_categorical_text(values: pd.Series, label: str) -> pd.Series:
    normalized = pd.Series(pd.NA, index=values.index, dtype="string")
    populated = values.notna().astype(bool)
    text_values = values.map(lambda value: isinstance(value, str)).astype(bool)
    invalid_type = populated & ~text_values
    if invalid_type.any():
        raise ValueError(
            f"{label} must contain categorical text or missing values"
        )

    if populated.any():
        stripped = values.loc[populated].astype("string").str.strip()
        nonblank = stripped.ne("")
        normalized.loc[stripped.index[nonblank]] = stripped.loc[nonblank]
    return normalized


def _coerce_nullable_categorical_text(values: pd.Series) -> pd.Series:
    normalized = pd.Series(pd.NA, index=values.index, dtype="string")
    text_values = values.map(lambda value: isinstance(value, str)).astype(bool)
    if text_values.any():
        stripped = values.loc[text_values].astype("string").str.strip()
        nonblank = stripped.ne("")
        normalized.loc[stripped.index[nonblank]] = stripped.loc[nonblank]
    return normalized


def _coerce_nullable_positive_whole_numbers(values: pd.Series) -> pd.Series:
    normalized = pd.Series(pd.NA, index=values.index, dtype="Int64")
    booleans = values.map(lambda value: isinstance(value, (bool, np.bool_)))
    converted = pd.to_numeric(values, errors="coerce")
    finite = pd.Series(
        np.isfinite(converted.fillna(0).astype(float)), index=values.index
    )
    valid = (
        ~booleans
        & converted.notna()
        & finite
        & converted.ge(1)
        & converted.mod(1).eq(0)
    )
    normalized.loc[valid] = converted.loc[valid].astype("Int64")
    return normalized


def _validate_text(data: pd.DataFrame, column: str, label: str) -> None:
    if data.empty:
        return
    invalid_type = ~data[column].map(lambda value: isinstance(value, str))
    blank = data[column].astype("string").str.strip().eq("").fillna(True)
    if bool((invalid_type | blank).any()):
        raise ValueError(f"{label} {column} values must be nonblank strings")


def _require_columns(
    data: pd.DataFrame,
    required: list[str],
    label: str,
) -> None:
    missing = sorted(set(required).difference(data.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _reject_duplicate_columns(data: pd.DataFrame, label: str) -> None:
    duplicates = data.columns[data.columns.duplicated()].tolist()
    if duplicates:
        raise ValueError(f"{label} contains duplicate columns: {duplicates}")
