"""Resolve expected starting quarterbacks for scheduled NFL matchups."""

from __future__ import annotations

from numbers import Integral

import pandas as pd


SCHEDULE_COLUMNS = [
    "season",
    "week",
    "game_id",
    "game_date",
    "game_time",
    "team",
    "opponent",
    "home_away",
]

DEPTH_CHART_COLUMNS = [
    "season",
    "week",
    "game_type",
    "depth_chart_date",
    "team",
    "player_id",
    "player_name",
    "position",
    "depth_rank",
]

LEGACY_DEPTH_CHART_COLUMNS = [
    column for column in DEPTH_CHART_COLUMNS if column != "depth_chart_date"
]

OVERRIDE_COLUMNS = [
    "season",
    "report_week",
    "team",
    "player_id",
    "player_name",
    "selection_notes",
]

OUTPUT_COLUMNS = [
    "season",
    "report_week",
    "game_id",
    "game_date",
    "game_time",
    "team",
    "opponent",
    "home_away",
    "expected_player_id",
    "expected_player_name",
    "selection_source",
    "depth_chart_date",
    "depth_rank",
    "starter_uncertain",
    "selection_notes",
]

DEPTH_CHART_SOURCE_MAPPINGS = [
    {
        "required": {
            "season",
            "week",
            "game_type",
            "club_code",
            "gsis_id",
            "full_name",
            "position",
            "depth_team",
        },
        "columns": {
            "season": "season",
            "week": "week",
            "game_type": "game_type",
            "club_code": "team",
            "gsis_id": "player_id",
            "full_name": "player_name",
            "position": "position",
            "depth_team": "depth_rank",
        },
        "dated_snapshot": False,
    },
    {
        "required": {
            "season",
            "week",
            "game_type",
            "team",
            "gsis_id",
            "player_name",
            "pos_abb",
            "pos_rank",
        },
        "columns": {
            "season": "season",
            "week": "week",
            "game_type": "game_type",
            "team": "team",
            "gsis_id": "player_id",
            "player_name": "player_name",
            "pos_abb": "position",
            "pos_rank": "depth_rank",
        },
        "dated_snapshot": False,
    },
    {
        "required": {
            "season",
            "week",
            "season_type",
            "team",
            "gsis_id",
            "player_name",
            "pos_abb",
            "pos_rank",
        },
        "columns": {
            "season": "season",
            "week": "week",
            "season_type": "game_type",
            "team": "team",
            "gsis_id": "player_id",
            "player_name": "player_name",
            "pos_abb": "position",
            "pos_rank": "depth_rank",
        },
        "dated_snapshot": False,
    },
    {
        "required": {
            "season",
            "week",
            "season_type",
            "team",
            "player_id",
            "player_name",
            "position",
            "depth_rank",
        },
        "columns": {
            "season": "season",
            "week": "week",
            "season_type": "game_type",
            "team": "team",
            "player_id": "player_id",
            "player_name": "player_name",
            "position": "position",
            "depth_rank": "depth_rank",
        },
        "dated_snapshot": False,
    },
    {
        "required": {
            "dt",
            "team",
            "gsis_id",
            "player_name",
            "pos_abb",
            "pos_rank",
        },
        "columns": {
            "dt": "depth_chart_date",
            "team": "team",
            "gsis_id": "player_id",
            "player_name": "player_name",
            "pos_abb": "position",
            "pos_rank": "depth_rank",
        },
        "dated_snapshot": True,
    },
    {
        "required": set(LEGACY_DEPTH_CHART_COLUMNS),
        "columns": {column: column for column in LEGACY_DEPTH_CHART_COLUMNS},
        "dated_snapshot": False,
    },
    {
        "required": set(DEPTH_CHART_COLUMNS),
        "columns": {column: column for column in DEPTH_CHART_COLUMNS},
        "dated_snapshot": False,
    },
]


def resolve_expected_quarterbacks(
    schedule_rows: pd.DataFrame,
    depth_charts: pd.DataFrame,
    report_season: int,
    report_week: int,
    as_of_date: object = None,
    manual_overrides: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return one expected-quarterback row for each scheduled team."""

    _validate_report_value(report_season, "report_season")
    _validate_report_value(report_week, "report_week")
    parsed_as_of_date = _parse_scalar_date(as_of_date, "as_of_date")
    _validate_required_columns(
        schedule_rows,
        SCHEDULE_COLUMNS,
        "Schedule data",
    )

    scheduled_teams = schedule_rows.loc[
        (schedule_rows["season"] == report_season)
        & (schedule_rows["week"] == report_week),
        SCHEDULE_COLUMNS,
    ].copy()

    scheduled_teams["report_week"] = report_week

    if scheduled_teams.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    selected_depth_qbs = _select_depth_chart_quarterbacks(
        depth_charts,
        report_season,
        report_week,
        scheduled_teams["team"],
        parsed_as_of_date,
    )

    result = scheduled_teams.merge(
        selected_depth_qbs,
        on="team",
        how="left",
    )
    result["depth_rank"] = pd.to_numeric(result["depth_rank"], errors="coerce")

    result["expected_player_id"] = result["player_id"]
    result["expected_player_name"] = result["player_name"]
    result["selection_source"] = "depth_chart"
    result["selection_notes"] = ""

    unresolved_mask = result["expected_player_id"].isna()
    result.loc[unresolved_mask, "selection_source"] = "unresolved"
    result.loc[unresolved_mask, "starter_uncertain"] = True
    result.loc[unresolved_mask, "selection_notes"] = (
        "No quarterback found on requested regular-season depth chart"
    )

    if manual_overrides is not None:
        result = _apply_manual_overrides(
            result,
            manual_overrides,
            report_season,
            report_week,
        )

    result = result.loc[:, OUTPUT_COLUMNS].sort_values(
        by=["game_id", "team"],
        kind="mergesort",
    )

    return result.reset_index(drop=True)


def _select_depth_chart_quarterbacks(
    depth_charts: pd.DataFrame,
    report_season: int,
    report_week: int,
    scheduled_teams: pd.Series,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    depth_data, is_dated_snapshot = _normalize_depth_charts(depth_charts)

    if is_dated_snapshot:
        requested_depth_rows = _select_latest_dated_snapshots(
            depth_data,
            scheduled_teams,
            as_of_date,
        )
    else:
        requested_depth_rows = depth_data.loc[
            (depth_data["season"] == report_season)
            & (depth_data["week"] == report_week)
            & (depth_data["game_type"] == "REG"),
            DEPTH_CHART_COLUMNS,
        ].copy()

    requested_qbs = requested_depth_rows.loc[
        requested_depth_rows["position"] == "QB",
        DEPTH_CHART_COLUMNS,
    ].copy()

    if requested_qbs.empty:
        return _empty_selected_depth_chart()

    requested_qbs["depth_rank"] = pd.to_numeric(
        requested_qbs["depth_rank"],
        errors="coerce",
    )
    requested_qbs = requested_qbs.dropna(subset=["depth_rank"])
    requested_qbs = requested_qbs.drop_duplicates(
        subset=[
            "season",
            "week",
            "game_type",
            "team",
            "player_id",
            "player_name",
            "position",
            "depth_rank",
        ]
    )

    if requested_qbs.empty:
        return _empty_selected_depth_chart()

    best_rank = requested_qbs.groupby("team", sort=False)["depth_rank"].transform(
        "min"
    )
    candidates = requested_qbs.loc[
        requested_qbs["depth_rank"] == best_rank
    ].copy()
    candidates["_player_id_sort"] = candidates["player_id"].astype(str)
    candidates = candidates.sort_values(
        by=["team", "_player_id_sort"],
        kind="mergesort",
    )

    selected = candidates.drop_duplicates(subset=["team"], keep="first").copy()
    tie_counts = (
        candidates.groupby("team", sort=False)["player_id"]
        .nunique()
        .rename("best_depth_player_count")
        .reset_index()
    )
    selected = selected.merge(tie_counts, on="team", how="left")
    selected["starter_uncertain"] = selected["best_depth_player_count"] > 1

    return selected.loc[
        :,
        [
            "team",
            "player_id",
            "player_name",
            "depth_chart_date",
            "depth_rank",
            "starter_uncertain",
        ],
    ]


def _empty_selected_depth_chart() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "team",
            "player_id",
            "player_name",
            "depth_chart_date",
            "depth_rank",
            "starter_uncertain",
        ]
    )


def _select_latest_dated_snapshots(
    depth_data: pd.DataFrame,
    scheduled_teams: pd.Series,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    snapshot_rows = depth_data.loc[
        depth_data["team"].isin(scheduled_teams) &
        (depth_data["depth_chart_date"] <= as_of_date),
        DEPTH_CHART_COLUMNS,
    ].copy()

    if snapshot_rows.empty:
        return snapshot_rows

    latest_snapshot_dates = snapshot_rows.groupby("team", sort=False)[
        "depth_chart_date"
    ].transform("max")

    return snapshot_rows.loc[
        snapshot_rows["depth_chart_date"] == latest_snapshot_dates
    ].copy()


def _normalize_depth_charts(depth_charts: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    for mapping in DEPTH_CHART_SOURCE_MAPPINGS:
        if mapping["required"].issubset(depth_charts.columns):
            normalized = depth_charts.loc[:, mapping["columns"].keys()].rename(
                columns=mapping["columns"]
            )
            if mapping["dated_snapshot"]:
                normalized["season"] = pd.NA
                normalized["week"] = pd.NA
                normalized["game_type"] = pd.NA
                normalized["depth_chart_date"] = _parse_depth_chart_dates(
                    normalized["depth_chart_date"]
                )
            elif "depth_chart_date" not in normalized.columns:
                normalized["depth_chart_date"] = pd.NA

            return normalized.loc[:, DEPTH_CHART_COLUMNS].copy(), mapping[
                "dated_snapshot"
            ]

    required_options = [
        sorted(mapping["required"]) for mapping in DEPTH_CHART_SOURCE_MAPPINGS
    ]
    raise ValueError(
        "NFL depth-chart data is missing required columns for a supported "
        f"source shape. Expected one of: {required_options}"
    )


def _parse_depth_chart_dates(dates: pd.Series) -> pd.Series:
    parsed_dates = pd.to_datetime(
        dates,
        errors="coerce",
        format="mixed",
        utc=True,
    )
    invalid_mask = parsed_dates.isna() & dates.notna()

    if invalid_mask.any():
        invalid_values = dates.loc[invalid_mask].astype(str).unique().tolist()
        raise ValueError(
            "NFL depth-chart data contains invalid dt values: "
            f"{invalid_values}"
        )

    if parsed_dates.isna().any():
        raise ValueError("NFL depth-chart data contains missing dt values")

    return parsed_dates.dt.normalize().dt.tz_localize(None)


def _parse_scalar_date(value: object, name: str) -> pd.Timestamp:
    if value is None:
        raise ValueError(f"{name} is required and must be a valid date")

    try:
        parsed_date = pd.to_datetime(value, errors="raise", utc=True)
    except (TypeError, ValueError):
        raise ValueError(f"{name} is required and must be a valid date") from None

    if pd.isna(parsed_date):
        raise ValueError(f"{name} is required and must be a valid date")

    return pd.Timestamp(parsed_date).normalize().tz_localize(None)


def _apply_manual_overrides(
    result: pd.DataFrame,
    manual_overrides: pd.DataFrame,
    report_season: int,
    report_week: int,
) -> pd.DataFrame:
    _validate_required_columns(
        manual_overrides,
        OVERRIDE_COLUMNS,
        "Manual quarterback override data",
    )

    optional_columns = ["depth_rank", "depth_chart_date"]
    override_columns = [
        *OVERRIDE_COLUMNS,
        *[column for column in optional_columns if column in manual_overrides.columns],
    ]
    overrides = manual_overrides.loc[:, override_columns].copy()
    overrides = overrides.loc[
        (overrides["season"] == report_season)
        & (overrides["report_week"] == report_week)
    ].copy()

    if overrides.empty:
        return result

    _validate_conflicting_overrides(overrides)

    overrides = overrides.drop_duplicates(
        subset=[
            column
            for column in overrides.columns
            if column in [*OVERRIDE_COLUMNS, "depth_rank", "depth_chart_date"]
        ]
    )
    if "depth_rank" not in overrides.columns:
        overrides["depth_rank"] = pd.NA
    overrides["depth_rank"] = pd.to_numeric(
        overrides["depth_rank"],
        errors="coerce",
    )
    if "depth_chart_date" not in overrides.columns:
        overrides["depth_chart_date"] = pd.NaT
    else:
        override_dates = pd.Series(pd.NaT, index=overrides.index)
        override_date_mask = overrides["depth_chart_date"].notna()
        if override_date_mask.any():
            override_dates.loc[override_date_mask] = _parse_depth_chart_dates(
                overrides.loc[override_date_mask, "depth_chart_date"]
            )
        overrides["depth_chart_date"] = override_dates

    overrides = overrides.rename(
        columns={
            "player_id": "override_player_id",
            "player_name": "override_player_name",
            "depth_rank": "override_depth_rank",
            "depth_chart_date": "override_depth_chart_date",
            "selection_notes": "override_selection_notes",
        }
    )

    result = result.merge(
        overrides.loc[
            :,
            [
                "team",
                "override_player_id",
                "override_player_name",
                "override_depth_rank",
                "override_depth_chart_date",
                "override_selection_notes",
            ],
        ],
        on="team",
        how="left",
    )

    override_mask = result["override_player_id"].notna()
    result.loc[override_mask, "expected_player_id"] = result.loc[
        override_mask,
        "override_player_id",
    ]
    result.loc[override_mask, "expected_player_name"] = result.loc[
        override_mask,
        "override_player_name",
    ]
    result.loc[override_mask, "depth_rank"] = result.loc[
        override_mask,
        "override_depth_rank",
    ]
    result.loc[override_mask, "depth_chart_date"] = result.loc[
        override_mask,
        "override_depth_chart_date",
    ]
    result.loc[override_mask, "selection_source"] = "manual_override"
    result.loc[override_mask, "starter_uncertain"] = False
    result.loc[override_mask, "selection_notes"] = result.loc[
        override_mask,
        "override_selection_notes",
    ]

    return result.drop(
        columns=[
            "override_player_id",
            "override_player_name",
            "override_depth_rank",
            "override_depth_chart_date",
            "override_selection_notes",
        ]
    )


def _validate_conflicting_overrides(overrides: pd.DataFrame) -> None:
    duplicate_keys = overrides.loc[
        overrides.duplicated(
            subset=["season", "report_week", "team"],
            keep=False,
        )
    ]

    if duplicate_keys.empty:
        return

    conflicting_keys = []
    for key, rows in duplicate_keys.groupby(
        ["season", "report_week", "team"],
        sort=True,
    ):
        comparable_columns = [
            column
            for column in [
                "player_id",
                "player_name",
                "selection_notes",
                "depth_rank",
                "depth_chart_date",
            ]
            if column in rows.columns
        ]
        if len(rows.loc[:, comparable_columns].drop_duplicates()) > 1:
            conflicting_keys.append(
                {
                    "season": key[0],
                    "report_week": key[1],
                    "team": key[2],
                }
            )

    if conflicting_keys:
        raise ValueError(
            "Conflicting manual quarterback overrides found for keys: "
            f"{conflicting_keys}"
        )


def _validate_required_columns(
    data: pd.DataFrame,
    required_columns: list[str],
    label: str,
) -> None:
    missing_columns = set(required_columns).difference(data.columns)

    if missing_columns:
        raise ValueError(
            f"{label} is missing required columns: {sorted(missing_columns)}"
        )


def _validate_report_value(value: int, name: str) -> None:
    """Require positive integer report season and week values."""

    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be an integer greater than or equal to 1")
