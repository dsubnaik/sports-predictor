"""Resolve expected NFL running-back participants from canonical snapshots.

Depth charts identify expected participants, not guaranteed game-day actives.
Every player listed at exactly ``RB`` is retained; usage is not considered.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd

from football.data.normalize_depth_charts import CANONICAL_DEPTH_CHART_COLUMNS


REQUESTED_TEAM_COLUMNS = ["team"]
OVERRIDE_COLUMNS = [
    "season",
    "report_week",
    "team",
    "player_id",
    "player_name",
    "participant_order",
]
OPTIONAL_OVERRIDE_COLUMNS = [
    "selection_notes",
    "depth_chart_date",
    "depth_chart_week",
    "depth_position",
    "depth_rank",
]
OUTPUT_COLUMNS = [
    "report_season",
    "report_week",
    "team",
    "player_id",
    "player_name",
    "position",
    "participant_order",
    "selection_source",
    "depth_chart_date",
    "depth_chart_week",
    "depth_position",
    "depth_rank",
    "resolution_missing",
    "selection_notes",
]


def resolve_expected_running_backs(
    requested_teams: pd.DataFrame,
    depth_charts: pd.DataFrame,
    report_season: int,
    report_week: int,
    as_of_date: object,
    manual_overrides: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return all expected RBs or one diagnostic row per unresolved team.

    ``depth_charts`` must use ``CANONICAL_DEPTH_CHART_COLUMNS``. Use
    ``normalize_nflverse_depth_charts`` at the data boundary for raw 2025+
    nflverse depth charts.
    """

    _validate_report_value(report_season, "report_season")
    _validate_report_value(report_week, "report_week")
    parsed_as_of_date = _parse_scalar_date(as_of_date, "as_of_date")
    _require_columns(requested_teams, REQUESTED_TEAM_COLUMNS, "Requested team data")
    _require_exact_canonical_schema(depth_charts)

    teams = requested_teams.loc[:, REQUESTED_TEAM_COLUMNS].copy()
    _validate_text(teams, "team", "Requested team data")
    teams = (
        teams.drop_duplicates()
        .sort_values("team", kind="mergesort")
        .reset_index(drop=True)
    )
    if teams.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    manual_result = _validated_manual_overrides(
        manual_overrides,
        teams,
        report_season,
        report_week,
        parsed_as_of_date,
    )
    overridden_teams = set(manual_result["team"])
    automatic_teams = teams.loc[~teams["team"].isin(overridden_teams)].copy()

    automatic_result = _resolve_automatic_participants(
        automatic_teams,
        depth_charts,
        report_season,
        report_week,
        parsed_as_of_date,
    )
    result = pd.concat([automatic_result, manual_result], ignore_index=True)

    result["participant_order"] = pd.array(
        result["participant_order"], dtype="Int64"
    )
    result["depth_chart_week"] = pd.array(
        result["depth_chart_week"], dtype="Int64"
    )
    result["depth_rank"] = pd.array(result["depth_rank"], dtype="Int64")
    result["depth_position"] = result["depth_position"].astype("string")

    result["_order_sort"] = pd.to_numeric(
        result["participant_order"], errors="coerce"
    )
    result["_player_id_sort"] = result["player_id"].astype("string").fillna("")
    return (
        result.sort_values(
            [
                "report_season",
                "report_week",
                "team",
                "_order_sort",
                "_player_id_sort",
            ],
            kind="mergesort",
            na_position="last",
        )
        .drop(columns=["_order_sort", "_player_id_sort"])
        .loc[:, OUTPUT_COLUMNS]
        .reset_index(drop=True)
    )


def _resolve_automatic_participants(
    teams: pd.DataFrame,
    depth_charts: pd.DataFrame,
    report_season: int,
    report_week: int,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    if teams.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    depth = depth_charts.loc[
        depth_charts["team"].isin(teams["team"]),
        CANONICAL_DEPTH_CHART_COLUMNS,
    ].copy()
    _validate_text(depth, "team", "Canonical depth-chart data")
    _validate_text(depth, "position", "Canonical depth-chart data")
    depth["_snapshot_timestamp"] = _parse_timestamps(
        depth["snapshot_timestamp"], "Canonical depth-chart data"
    )
    depth["_snapshot_date"] = (
        depth["_snapshot_timestamp"].dt.normalize().dt.tz_localize(None)
    )

    eligible = depth.loc[depth["_snapshot_date"].le(as_of_date)].copy()
    if eligible.empty:
        selected = eligible
    else:
        latest_dates = eligible.groupby("team", sort=False)[
            "_snapshot_date"
        ].transform("max")
        selected = eligible.loc[eligible["_snapshot_date"].eq(latest_dates)].copy()

    _validate_selected_rb_rows(selected)
    chosen = _choose_unambiguous_same_day_snapshots(selected)
    rbs = chosen.loc[chosen["position"].eq("RB")].copy()
    rbs = _unique_rb_participants(rbs)
    rbs = _order_participants(rbs)
    return _build_automatic_output(
        teams,
        chosen,
        rbs,
        report_season,
        report_week,
    )


def _validate_selected_rb_rows(selected: pd.DataFrame) -> None:
    rbs = selected.loc[selected["position"].eq("RB")].copy()
    _validate_text(rbs, "player_id", "Selected RB depth-chart data")
    _validate_text(rbs, "player_name", "Selected RB depth-chart data")
    rb_ranks = _positive_whole_numbers(
        rbs["depth_rank"],
        "Selected RB depth-chart depth_rank",
        allow_missing=True,
    )
    rb_positions = _nullable_categorical_text(
        rbs["depth_position"],
        "Selected RB depth-chart depth_position",
    )
    selected["depth_rank"] = selected["depth_rank"].astype("object")
    selected["depth_position"] = selected["depth_position"].astype("object")
    selected.loc[rbs.index, "depth_rank"] = rb_ranks.astype("object").tolist()
    selected.loc[rbs.index, "depth_position"] = rb_positions.astype(
        "object"
    ).tolist()


def _choose_unambiguous_same_day_snapshots(
    selected: pd.DataFrame,
) -> pd.DataFrame:
    if selected.empty:
        return selected.copy()

    chosen_groups = []
    conflicts = []
    for (team, snapshot_date), team_rows in selected.groupby(
        ["team", "_snapshot_date"], sort=True
    ):
        signatures = []
        timestamp_groups = []
        for timestamp, snapshot in team_rows.groupby(
            "_snapshot_timestamp", sort=True
        ):
            signatures.append(_rb_projection_signature(snapshot))
            timestamp_groups.append((timestamp, snapshot))

        if len(set(signatures)) > 1:
            conflicts.append(
                {"team": team, "depth_chart_date": snapshot_date}
            )
            continue

        chosen_groups.append(timestamp_groups[-1][1])

    if conflicts:
        raise ValueError(
            "Ambiguous same-day NFL depth-chart snapshots contain conflicting "
            f"RB projections: {conflicts}"
        )

    if not chosen_groups:
        return selected.iloc[0:0].copy()
    return pd.concat(chosen_groups, ignore_index=False).copy()


def _rb_projection_signature(snapshot: pd.DataFrame) -> tuple[tuple[str, ...], ...]:
    columns = [
        "player_id",
        "player_name",
        "position",
        "depth_position",
        "depth_rank",
    ]
    rbs = snapshot.loc[snapshot["position"].eq("RB"), columns].drop_duplicates()
    values = rbs.astype("object").where(rbs.notna(), "<missing>")
    string_values = values.map(lambda value: str(value))
    return tuple(sorted(string_values.itertuples(index=False, name=None)))


def _unique_rb_participants(rbs: pd.DataFrame) -> pd.DataFrame:
    if rbs.empty:
        return rbs.copy()

    comparable = [
        "team",
        "player_id",
        "player_name",
        "position",
        "depth_position",
        "depth_rank",
        "_snapshot_date",
    ]
    unique = rbs.drop_duplicates(subset=comparable).copy()
    conflicts = unique.loc[unique.duplicated(["team", "player_id"], keep=False)]
    if conflicts.empty:
        return unique

    keys = (
        conflicts.loc[:, ["team", "player_id"]]
        .drop_duplicates()
        .sort_values(["team", "player_id"], kind="mergesort")
        .to_dict("records")
    )
    raise ValueError(
        "Conflicting RB depth-chart participant records found for keys: "
        f"{keys}"
    )


def _order_participants(rbs: pd.DataFrame) -> pd.DataFrame:
    ordered = rbs.copy()
    if ordered.empty:
        ordered["participant_order"] = pd.Series(dtype="Int64")
        return ordered

    ordered["_depth_position_sort"] = ordered["depth_position"].astype("string")
    ordered["_player_id_sort"] = ordered["player_id"].astype("string")
    ordered = ordered.sort_values(
        ["team", "depth_rank", "_depth_position_sort", "_player_id_sort"],
        kind="mergesort",
        na_position="last",
    )
    ordered["participant_order"] = (
        ordered.groupby("team", sort=False).cumcount() + 1
    )
    return ordered.drop(columns=["_depth_position_sort", "_player_id_sort"])


def _build_automatic_output(
    teams: pd.DataFrame,
    selected: pd.DataFrame,
    rbs: pd.DataFrame,
    report_season: int,
    report_week: int,
) -> pd.DataFrame:
    records = []
    for team in teams["team"]:
        team_rbs = rbs.loc[rbs["team"].eq(team)]
        for row in team_rbs.to_dict("records"):
            records.append(
                {
                    "report_season": report_season,
                    "report_week": report_week,
                    "team": team,
                    "player_id": row["player_id"],
                    "player_name": row["player_name"],
                    "position": "RB",
                    "participant_order": row["participant_order"],
                    "selection_source": "depth_chart",
                    "depth_chart_date": row["_snapshot_date"],
                    "depth_chart_week": pd.NA,
                    "depth_position": row["depth_position"],
                    "depth_rank": row["depth_rank"],
                    "resolution_missing": False,
                    "selection_notes": "",
                }
            )

        if team_rbs.empty:
            context = selected.loc[selected["team"].eq(team)]
            snapshot_found = not context.empty
            records.append(
                {
                    "report_season": report_season,
                    "report_week": report_week,
                    "team": team,
                    "player_id": pd.NA,
                    "player_name": pd.NA,
                    "position": "RB",
                    "participant_order": pd.NA,
                    "selection_source": "unresolved",
                    "depth_chart_date": (
                        context["_snapshot_date"].iloc[0]
                        if snapshot_found
                        else pd.NaT
                    ),
                    "depth_chart_week": pd.NA,
                    "depth_position": pd.NA,
                    "depth_rank": pd.NA,
                    "resolution_missing": True,
                    "selection_notes": (
                        "Selected depth-chart snapshot contains no RB participants"
                        if snapshot_found
                        else "No eligible depth-chart snapshot for requested team"
                    ),
                }
            )

    return pd.DataFrame(records, columns=OUTPUT_COLUMNS)


def _validated_manual_overrides(
    data: pd.DataFrame | None,
    teams: pd.DataFrame,
    report_season: int,
    report_week: int,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Manual RB override data must be a pandas DataFrame")

    _require_columns(data, OVERRIDE_COLUMNS, "Manual RB override data")
    columns = [
        *OVERRIDE_COLUMNS,
        *[column for column in OPTIONAL_OVERRIDE_COLUMNS if column in data.columns],
    ]
    overrides = data.loc[:, columns].copy()
    overrides = overrides.loc[
        overrides["season"].eq(report_season)
        & overrides["report_week"].eq(report_week)
    ].copy()
    if overrides.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    for column in ["team", "player_id", "player_name"]:
        _validate_text(overrides, column, "Manual RB override data")

    unknown_teams = sorted(set(overrides["team"]) - set(teams["team"]))
    if unknown_teams:
        raise ValueError(
            f"Manual RB override teams were not requested: {unknown_teams}"
        )

    overrides["participant_order"] = _positive_whole_numbers(
        overrides["participant_order"],
        "Manual RB override participant_order",
        allow_missing=False,
    )
    _reject_duplicate_override_values(overrides)

    for column in OPTIONAL_OVERRIDE_COLUMNS:
        if column not in overrides:
            overrides[column] = "" if column == "selection_notes" else pd.NA
    overrides["selection_notes"] = overrides["selection_notes"].fillna("")
    overrides["depth_rank"] = _positive_whole_numbers(
        overrides["depth_rank"],
        "Manual RB override depth_rank",
        allow_missing=True,
    )
    overrides["depth_position"] = _nullable_categorical_text(
        overrides["depth_position"],
        "Manual RB override depth_position",
    )
    overrides["depth_chart_week"] = _positive_whole_numbers(
        overrides["depth_chart_week"],
        "Manual RB override depth_chart_week",
        allow_missing=True,
    )

    future_weeks = overrides["depth_chart_week"].notna() & overrides[
        "depth_chart_week"
    ].gt(report_week)
    if future_weeks.any():
        conflicts = _override_conflicts(
            overrides.loc[future_weeks], "depth_chart_week"
        )
        raise ValueError(
            "Manual RB override depth_chart_week cannot be later than "
            f"report_week: {conflicts}"
        )

    parsed_dates = pd.Series(pd.NaT, index=overrides.index, dtype="datetime64[ns]")
    populated_dates = overrides["depth_chart_date"].notna()
    if populated_dates.any():
        parsed_dates.loc[populated_dates] = _parse_date_series(
            overrides.loc[populated_dates, "depth_chart_date"],
            "Manual RB override data",
        )
    overrides["depth_chart_date"] = parsed_dates

    future_dates = overrides["depth_chart_date"].notna() & overrides[
        "depth_chart_date"
    ].gt(as_of_date)
    if future_dates.any():
        conflicts = _override_conflicts(
            overrides.loc[future_dates], "depth_chart_date"
        )
        raise ValueError(
            "Manual RB override depth_chart_date cannot be later than "
            f"as_of_date: {conflicts}"
        )

    manual = pd.DataFrame(
        {
            "report_season": report_season,
            "report_week": report_week,
            "team": overrides["team"],
            "player_id": overrides["player_id"],
            "player_name": overrides["player_name"],
            "position": "RB",
            "participant_order": overrides["participant_order"],
            "selection_source": "manual_override",
            "depth_chart_date": overrides["depth_chart_date"],
            "depth_chart_week": overrides["depth_chart_week"],
            "depth_position": overrides["depth_position"],
            "depth_rank": overrides["depth_rank"],
            "resolution_missing": False,
            "selection_notes": overrides["selection_notes"],
        }
    )
    return manual.loc[:, OUTPUT_COLUMNS]


def _reject_duplicate_override_values(overrides: pd.DataFrame) -> None:
    duplicate_ids = overrides.duplicated(["team", "player_id"], keep=False)
    if duplicate_ids.any():
        keys = (
            overrides.loc[duplicate_ids, ["team", "player_id"]]
            .drop_duplicates()
            .sort_values(["team", "player_id"], kind="mergesort")
            .to_dict("records")
        )
        raise ValueError(f"Duplicate manual RB override identities found: {keys}")

    duplicate_orders = overrides.duplicated(
        ["team", "participant_order"], keep=False
    )
    if duplicate_orders.any():
        keys = (
            overrides.loc[
                duplicate_orders, ["team", "participant_order"]
            ]
            .drop_duplicates()
            .sort_values(["team", "participant_order"], kind="mergesort")
            .to_dict("records")
        )
        raise ValueError(
            "Conflicting manual RB override participant orders found: "
            f"{keys}"
        )


def _override_conflicts(data: pd.DataFrame, column: str) -> list[dict]:
    return (
        data.sort_values(
            ["team", "participant_order", "player_id"], kind="mergesort"
        )
        .loc[:, ["team", "player_id", column]]
        .to_dict("records")
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


def _parse_date_series(values: pd.Series, label: str) -> pd.Series:
    parsed = _parse_timestamps(values, label)
    return parsed.dt.normalize().dt.tz_localize(None)


def _parse_scalar_date(value: object, name: str) -> pd.Timestamp:
    if value is None:
        raise ValueError(f"{name} is required and must be a valid date")
    try:
        parsed = pd.to_datetime(value, errors="raise", utc=True)
    except (TypeError, ValueError):
        raise ValueError(f"{name} is required and must be a valid date") from None
    if pd.isna(parsed):
        raise ValueError(f"{name} is required and must be a valid date")
    return pd.Timestamp(parsed).normalize().tz_localize(None)


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


def _validate_text(data: pd.DataFrame, column: str, label: str) -> None:
    if data.empty:
        return
    invalid_type = ~data[column].map(lambda value: isinstance(value, str))
    blank = data[column].astype("string").str.strip().eq("").fillna(True)
    if bool((invalid_type | blank).any()):
        raise ValueError(f"{label} {column} values must be nonblank strings")


def _require_exact_canonical_schema(data: pd.DataFrame) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Canonical depth-chart data must be a pandas DataFrame")
    duplicate_columns = data.columns[data.columns.duplicated()].tolist()
    if duplicate_columns:
        raise ValueError(
            "Canonical depth-chart data contains duplicate columns: "
            f"{duplicate_columns}"
        )
    missing = sorted(set(CANONICAL_DEPTH_CHART_COLUMNS).difference(data.columns))
    extras = sorted(set(data.columns).difference(CANONICAL_DEPTH_CHART_COLUMNS))
    if missing or extras:
        raise ValueError(
            "Canonical depth-chart data must contain exactly "
            f"{CANONICAL_DEPTH_CHART_COLUMNS}; missing={missing}, "
            f"unexpected={extras}"
        )


def _require_columns(data: pd.DataFrame, required: list[str], label: str) -> None:
    missing = sorted(set(required).difference(data.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _validate_report_value(value: object, name: str) -> None:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < 1
    ):
        raise ValueError(f"{name} must be an integer greater than or equal to 1")
