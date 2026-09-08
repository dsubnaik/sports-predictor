import inspect

import numpy as np
import pandas as pd
import pytest

from football.data.normalize_depth_charts import (
    CANONICAL_DEPTH_CHART_COLUMNS,
    normalize_nflverse_depth_charts,
)
from football.features.expected_running_backs import (
    OUTPUT_COLUMNS,
    resolve_expected_running_backs,
)


def teams(*values):
    return pd.DataFrame({"team": list(values)})


def canonical(rows):
    return pd.DataFrame(rows, columns=CANONICAL_DEPTH_CHART_COLUMNS)


def empty_canonical():
    return pd.DataFrame(columns=CANONICAL_DEPTH_CHART_COLUMNS)


def overrides(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "season",
            "report_week",
            "team",
            "player_id",
            "player_name",
            "participant_order",
        ],
    )


def resolve(
    requested=None,
    depth=None,
    manual=None,
    *,
    report_season=2026,
    report_week=1,
    as_of_date="2026-09-09",
):
    return resolve_expected_running_backs(
        teams("KC") if requested is None else requested,
        canonical(
            [
                (
                    "KC",
                    "rb-1",
                    "Starter",
                    "RB",
                    "2026-09-08T12:00:00Z",
                    "RB",
                    1,
                )
            ]
        )
        if depth is None
        else depth,
        report_season,
        report_week,
        as_of_date,
        manual_overrides=manual,
    )


def test_public_signature_requires_as_of_date():
    signature = inspect.signature(resolve_expected_running_backs)

    assert signature.parameters["as_of_date"].default is inspect.Parameter.empty


def test_one_team_one_rb_produces_expected_participant():
    result = resolve()

    assert result.columns.tolist() == OUTPUT_COLUMNS
    assert result[["team", "player_id", "selection_source"]].values.tolist() == [
        ["KC", "rb-1", "depth_chart"]
    ]
    assert result["resolution_missing"].tolist() == [False]


def test_all_rbs_and_backup_are_preserved_in_depth_order():
    depth = canonical(
        [
            ("KC", "rb-3", "Third", "RB", "2026-09-08", "RB", pd.NA),
            ("KC", "rb-2", "Backup", "RB", "2026-09-08", "RB", 2),
            ("KC", "rb-1", "Starter", "RB", "2026-09-08", "RB", 1),
            ("KC", "fb-1", "Fullback", "FB", "2026-09-08", "FB", 1),
            ("KC", "wr-1", "Rushing WR", "WR", "2026-09-08", "WR", 1),
        ]
    )

    result = resolve(depth=depth)

    assert result["player_id"].tolist() == ["rb-1", "rb-2", "rb-3"]
    assert result["participant_order"].tolist() == [1, 2, 3]
    assert pd.isna(result.loc[result["player_id"].eq("rb-3"), "depth_rank"]).all()


def test_depth_rank_controls_order_before_categorical_depth_position():
    depth = canonical(
        [
            ("KC", "rb-3", "Third", "RB", "2026-09-08", "A", 2),
            ("KC", "rb-2", "Second", "RB", "2026-09-08", "Z", 1),
            ("KC", "rb-1", "First", "RB", "2026-09-08", "A", 1),
        ]
    )

    result = resolve(depth=depth)

    assert result["player_id"].tolist() == ["rb-1", "rb-2", "rb-3"]
    assert result["depth_position"].tolist() == ["A", "Z", "A"]
    assert str(result["depth_rank"].dtype) == "Int64"


def test_categorical_and_missing_depth_positions_survive_resolution():
    depth = canonical(
        [
            ("KC", "rb-1", "Primary", "RB", "2026-09-08", "RB", 1),
            ("KC", "rb-2", "Passing", "RB", "2026-09-08", "3DRB", 2),
            ("KC", "rb-3", "Unknown Slot", "RB", "2026-09-08", "   ", 3),
        ]
    )

    result = resolve(depth=depth)

    assert result["depth_position"].tolist()[:2] == ["RB", "3DRB"]
    assert pd.isna(result["depth_position"].iloc[2])


def test_same_name_different_ids_remain_separate():
    depth = canonical(
        [
            ("KC", "rb-1", "Same Name", "RB", "2026-09-08", "RB", 1),
            ("KC", "rb-2", "Same Name", "RB", "2026-09-08", "RB", 2),
        ]
    )

    assert resolve(depth=depth)["player_id"].tolist() == ["rb-1", "rb-2"]


def test_latest_eligible_snapshot_is_selected_independently_per_team():
    depth = canonical(
        [
            ("KC", "kc-old", "Old KC", "RB", "2026-09-06", "RB", 1),
            ("KC", "kc-new", "New KC", "RB", "2026-09-08", "RB", 1),
            ("KC", "kc-future", "Future KC", "RB", "2026-09-10", "RB", 1),
            ("BUF", "buf-old", "Old BUF", "RB", "2026-09-05", "RB", 1),
            ("BUF", "buf-new", "New BUF", "RB", "2026-09-07", "RB", 1),
        ]
    )

    result = resolve(teams("KC", "BUF"), depth)

    assert result[["team", "player_id"]].values.tolist() == [
        ["BUF", "buf-new"],
        ["KC", "kc-new"],
    ]
    assert result["depth_chart_date"].tolist() == [
        pd.Timestamp("2026-09-07"),
        pd.Timestamp("2026-09-08"),
    ]


def test_snapshot_selection_precedes_rb_filtering_and_never_uses_stale_rb():
    depth = canonical(
        [
            ("KC", "rb-old", "Old Runner", "RB", "2026-09-07", "RB", 1),
            ("KC", None, None, "QB", "2026-09-08", "QB", 1),
        ]
    )

    result = resolve(depth=depth)

    assert result["selection_source"].tolist() == ["unresolved"]
    assert result["depth_chart_date"].tolist() == [pd.Timestamp("2026-09-08")]
    assert "no RB participants" in result["selection_notes"].item()


def test_identical_same_day_rb_projections_ignore_non_rb_changes():
    depth = canonical(
        [
            ("KC", "rb-1", "Runner", "RB", "2026-09-08T08:00:00Z", "RB", 1),
            ("KC", "qb-old", "Old QB", "QB", "2026-09-08T08:00:00Z", "QB", 1),
            ("KC", "rb-1", "Runner", "RB", "2026-09-08T16:00:00Z", "RB", 1),
            ("KC", None, None, "QB", "2026-09-08T16:00:00Z", "QB", 1),
        ]
    )

    result = resolve(depth=depth)

    assert result["player_id"].tolist() == ["rb-1"]
    assert result["depth_chart_date"].tolist() == [pd.Timestamp("2026-09-08")]


@pytest.mark.parametrize(
    "later_row",
    [
        ("KC", None, None, "QB", "2026-09-08T16:00:00Z", "QB", 1),
        ("KC", "rb-1", "Runner", "WR", "2026-09-08T16:00:00Z", "WR", 1),
        ("KC", "rb-1", "Runner", "RB", "2026-09-08T16:00:00Z", "3DRB", 1),
        ("KC", "rb-1", "Renamed", "RB", "2026-09-08T16:00:00Z", "RB", 1),
    ],
)
def test_removed_reclassified_or_changed_same_day_rb_is_ambiguous(later_row):
    depth = canonical(
        [
            ("KC", "rb-1", "Runner", "RB", "2026-09-08T08:00:00Z", "RB", 1),
            later_row,
        ]
    )

    with pytest.raises(ValueError, match="Ambiguous same-day.*RB projections"):
        resolve(depth=depth)


def test_same_day_empty_rb_projection_is_not_ignored_or_stale():
    depth = canonical(
        [
            ("KC", "rb-1", "Runner", "RB", "2026-09-08T08:00:00Z", "RB", 1),
            ("KC", None, None, "WR", "2026-09-08T16:00:00Z", "WR", 1),
        ]
    )

    with pytest.raises(ValueError, match="Ambiguous same-day"):
        resolve(depth=depth)


def test_changed_same_day_categorical_depth_position_is_ambiguous():
    depth = canonical(
        [
            ("KC", "rb-1", "Runner", "RB", "2026-09-08T08:00:00Z", "RB", 1),
            ("KC", "rb-1", "Runner", "RB", "2026-09-08T16:00:00Z", "3DRB", 1),
        ]
    )

    with pytest.raises(ValueError, match="Ambiguous same-day.*RB projections"):
        resolve(depth=depth)


def test_non_rb_missing_identity_does_not_block_resolution():
    depth = canonical(
        [
            ("KC", "rb-1", "Runner", "RB", "2026-09-08", "RB", 1),
            ("KC", None, None, "QB", "2026-09-08", None, None),
        ]
    )

    assert resolve(depth=depth)["player_id"].tolist() == ["rb-1"]


@pytest.mark.parametrize("column", ["player_id", "player_name"])
def test_selected_rb_missing_identity_raises(column):
    depth = canonical(
        [("KC", "rb-1", "Runner", "RB", "2026-09-08", "RB", 1)]
    )
    depth.loc[0, column] = None

    with pytest.raises(ValueError, match="Selected RB.*nonblank strings"):
        resolve(depth=depth)


@pytest.mark.parametrize("rank_dtype", ["Int64", "Float64"])
def test_nullable_rank_dtypes_are_supported_and_missing_sorts_last(rank_dtype):
    depth = canonical(
        [
            ("KC", "rb-2", "Unranked", "RB", "2026-09-08", "RB", pd.NA),
            ("KC", "rb-1", "Ranked", "RB", "2026-09-08", "3DRB", 1),
        ]
    )
    depth["depth_rank"] = depth["depth_rank"].astype(rank_dtype)

    result = resolve(depth=depth)

    assert result["player_id"].tolist() == ["rb-1", "rb-2"]
    assert pd.isna(result["depth_rank"].iloc[1])


def test_categorical_depth_position_is_not_accepted_as_depth_rank():
    depth = canonical(
        [("KC", "rb-1", "Runner", "RB", "2026-09-08", "RB", "RB")]
    )

    with pytest.raises(ValueError, match="depth_rank.*positive whole numbers"):
        resolve(depth=depth)


def test_no_eligible_snapshot_and_empty_canonical_data_are_unresolved():
    future = canonical(
        [("KC", "future", "Future", "RB", "2026-09-10", "RB", 1)]
    )

    for depth in [future, empty_canonical()]:
        result = resolve(depth=depth)
        assert result["selection_source"].tolist() == ["unresolved"]
        assert pd.isna(result["depth_chart_date"].item())
        assert "No eligible" in result["selection_notes"].item()


def test_unrelated_malformed_canonical_rows_do_not_affect_requested_team():
    depth = canonical(
        [
            ("KC", "rb-1", "Runner", "RB", "2026-09-08", "RB", 1),
            (None, None, None, None, "not-a-date", None, None),
            ("BUF", None, None, "RB", "not-a-date", "bad", "bad"),
        ]
    )

    assert resolve(depth=depth)["player_id"].tolist() == ["rb-1"]


def test_malformed_requested_automatic_team_still_raises():
    depth = canonical(
        [("KC", "rb-1", "Runner", "RB", "not-a-date", "RB", 1)]
    )

    with pytest.raises(ValueError, match="invalid snapshot timestamps"):
        resolve(depth=depth)


def test_conflicting_duplicate_rb_participant_rows_raise():
    depth = canonical(
        [
            ("KC", "rb-1", "Runner", "RB", "2026-09-08", "RB", 1),
            ("KC", "rb-1", "Runner", "RB", "2026-09-08", "3DRB", 2),
        ]
    )

    with pytest.raises(ValueError, match="Conflicting RB depth-chart"):
        resolve(depth=depth)


def test_raw_aliases_and_extra_columns_are_rejected():
    raw = pd.DataFrame(
        {
            "dt": ["2026-09-08"],
            "team": ["KC"],
            "gsis_id": ["rb-1"],
            "player_name": ["Runner"],
            "pos_abb": ["RB"],
            "pos_slot": ["RB"],
            "pos_rank": [1],
        }
    )

    with pytest.raises(ValueError, match="must contain exactly"):
        resolve(depth=raw)

    extra = canonical(
        [("KC", "rb-1", "Runner", "RB", "2026-09-08", "RB", 1)]
    ).assign(dt="2026-09-08")
    with pytest.raises(ValueError, match=r"unexpected=\['dt'\]"):
        resolve(depth=extra)


def test_manual_override_can_replace_team_and_preserve_multiple_players():
    depth = canonical(
        [
            ("KC", "auto", "Automatic", "RB", "2026-09-08", "RB", 1),
            ("BUF", "buf", "Buffalo", "RB", "2026-09-08", "RB", 1),
        ]
    )
    manual = overrides(
        [
            (2026, 1, "KC", "manual-2", "Second Manual", 2),
            (2026, 1, "KC", "manual-1", "First Manual", 1),
        ]
    )

    result = resolve(teams("KC", "BUF"), depth, manual)

    assert result[["team", "player_id", "selection_source"]].values.tolist() == [
        ["BUF", "buf", "depth_chart"],
        ["KC", "manual-1", "manual_override"],
        ["KC", "manual-2", "manual_override"],
    ]
    assert result.loc[result["team"].eq("KC"), "selection_notes"].eq("").all()


def test_valid_override_bypasses_its_own_malformed_automatic_rows():
    depth = canonical(
        [("KC", None, None, "RB", "not-a-date", "bad", "bad")]
    )
    manual = overrides([(2026, 1, "KC", "manual", "Manual", 1)])

    result = resolve(depth=depth, manual=manual)

    assert result[["player_id", "selection_source"]].values.tolist() == [
        ["manual", "manual_override"]
    ]


def test_override_for_one_team_does_not_suppress_other_team_error():
    depth = canonical(
        [
            ("KC", None, None, "RB", "not-a-date", None, None),
            ("BUF", "bad", "Bad", "RB", "not-a-date", "RB", 1),
        ]
    )
    manual = overrides([(2026, 1, "KC", "manual", "Manual", 1)])

    with pytest.raises(ValueError, match="invalid snapshot timestamps"):
        resolve(teams("KC", "BUF"), depth, manual)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("depth_chart_date", "2026-09-10", "cannot be later than as_of_date"),
        ("depth_chart_week", 2, "cannot be later than report_week"),
        ("depth_chart_week", 0, "positive whole numbers"),
        ("depth_chart_week", 1.5, "positive whole numbers"),
        ("depth_chart_week", True, "positive whole numbers"),
        ("depth_chart_week", np.inf, "positive whole numbers"),
    ],
)
def test_invalid_or_future_override_provenance_raises(column, value, message):
    manual = overrides([(2026, 1, "KC", "manual", "Manual", 1)])
    manual[column] = value

    with pytest.raises(ValueError, match=message):
        resolve(manual=manual)


def test_valid_override_provenance_is_preserved_and_both_cutoffs_apply():
    manual = overrides([(2026, 2, "KC", "manual", "Manual", 1)])
    manual["depth_chart_date"] = "2026-09-09"
    manual["depth_chart_week"] = 1
    manual["selection_notes"] = "coach announcement"

    result = resolve(manual=manual, report_week=2)

    assert result["depth_chart_date"].tolist() == [pd.Timestamp("2026-09-09")]
    assert result["depth_chart_week"].tolist() == [1]
    assert result["selection_notes"].tolist() == ["coach announcement"]


def test_manual_override_accepts_categorical_depth_position():
    manual = overrides([(2026, 1, "KC", "manual", "Manual", 1)])
    manual["depth_position"] = " 3DRB "
    manual["depth_rank"] = 2

    result = resolve(manual=manual)

    assert result["depth_position"].tolist() == ["3DRB"]
    assert result["depth_rank"].tolist() == [2]


@pytest.mark.parametrize("value", [12, True, np.inf])
def test_manual_override_rejects_noncategorical_depth_position(value):
    manual = overrides([(2026, 1, "KC", "manual", "Manual", 1)])
    manual["depth_position"] = value

    with pytest.raises(ValueError, match="categorical text or missing values"):
        resolve(manual=manual)


def test_duplicate_override_identity_and_order_raise():
    duplicate_id = overrides(
        [
            (2026, 1, "KC", "same", "First", 1),
            (2026, 1, "KC", "same", "Second", 2),
        ]
    )
    with pytest.raises(ValueError, match="Duplicate manual RB override identities"):
        resolve(manual=duplicate_id)

    duplicate_order = overrides(
        [
            (2026, 1, "KC", "one", "First", 1),
            (2026, 1, "KC", "two", "Second", 1),
        ]
    )
    with pytest.raises(ValueError, match="Conflicting manual RB override"):
        resolve(manual=duplicate_order)


def test_irrelevant_override_context_is_not_validated_or_applied():
    manual = overrides([(2027, 2, "OTHER", None, None, 0)])
    manual["depth_chart_date"] = "2099-01-01"
    manual["depth_chart_week"] = 99

    assert resolve(manual=manual)["player_id"].tolist() == ["rb-1"]


@pytest.mark.parametrize("value", [None, "", "   ", 12])
def test_requested_team_values_must_be_nonblank_strings(value):
    with pytest.raises(ValueError, match="Requested team data.*nonblank strings"):
        resolve(requested=teams(value))


@pytest.mark.parametrize("name", ["report_season", "report_week"])
@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "1"])
def test_invalid_report_values_raise(name, value):
    kwargs = {name: value}
    with pytest.raises(ValueError, match=f"{name} must be an integer"):
        resolve(**kwargs)


@pytest.mark.parametrize("value", [None, "not-a-date", pd.NaT])
def test_as_of_date_is_required_and_valid(value):
    with pytest.raises(ValueError, match="as_of_date is required"):
        resolve(as_of_date=value)


def test_empty_requested_teams_return_documented_empty_schema():
    result = resolve(requested=teams(), depth=empty_canonical())

    assert result.empty
    assert result.columns.tolist() == OUTPUT_COLUMNS


def test_output_is_deterministic_and_all_inputs_remain_unchanged():
    requested = teams("KC", "BUF")
    depth = canonical(
        [
            ("KC", "kc-2", "KC Two", "RB", "2026-09-08", "3DRB", 2),
            ("BUF", "buf-1", "BUF One", "RB", "2026-09-07", "RB", 1),
            ("KC", "kc-1", "KC One", "RB", "2026-09-08", "RB", 1),
        ]
    )
    manual = overrides([(2025, 1, "OTHER", None, None, 0)])
    originals = [value.copy(deep=True) for value in (requested, depth, manual)]

    forward = resolve(requested, depth, manual)
    reverse = resolve(
        requested.iloc[::-1], depth.iloc[::-1], manual.iloc[::-1]
    )

    pd.testing.assert_frame_equal(forward, reverse)
    for current, original in zip((requested, depth, manual), originals):
        pd.testing.assert_frame_equal(current, original)


def test_realistic_current_nflverse_fixture_flows_end_to_end():
    raw = pd.DataFrame(
        [
            {
                "dt": "2026-09-08T12:00:00Z",
                "team": "KC",
                "player_name": "Primary Runner",
                "espn_id": "1001",
                "gsis_id": "00-001001",
                "pos_grp_id": "1",
                "pos_grp": "Offense",
                "pos_id": "2",
                "pos_name": "Running Back",
                "pos_abb": "RB",
                "pos_slot": "RB",
                "pos_rank": 1,
            },
            {
                "dt": "2026-09-08T12:00:00Z",
                "team": "KC",
                "player_name": "Passing Runner",
                "espn_id": "1002",
                "gsis_id": "00-001002",
                "pos_grp_id": "1",
                "pos_grp": "Offense",
                "pos_id": "3",
                "pos_name": "Running Back",
                "pos_abb": "RB",
                "pos_slot": "3DRB",
                "pos_rank": 2,
            },
        ]
    )
    original = raw.copy(deep=True)

    normalized = normalize_nflverse_depth_charts(raw)
    result = resolve(depth=normalized)

    assert result[["player_id", "depth_position", "depth_rank"]].values.tolist() == [
        ["00-001001", "RB", 1],
        ["00-001002", "3DRB", 2],
    ]
    pd.testing.assert_frame_equal(raw, original)
