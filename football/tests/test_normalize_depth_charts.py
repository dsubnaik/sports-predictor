import numpy as np
import pandas as pd
import pytest

from football.data.normalize_depth_charts import (
    CANONICAL_DEPTH_CHART_COLUMNS,
    NFLVERSE_DATED_DEPTH_CHART_COLUMNS,
    normalize_nflverse_depth_charts,
)


def raw_depth(rows):
    result = pd.DataFrame(rows, columns=NFLVERSE_DATED_DEPTH_CHART_COLUMNS)
    for column in NFLVERSE_DATED_DEPTH_CHART_COLUMNS:
        result[column] = result[column].astype("object")
    return result


def test_current_dated_schema_normalizes_to_exact_canonical_contract():
    source = raw_depth(
        [
            ("2026-09-08T12:00:00Z", "KC", "Runner", "rb-1", "RB", "RB", 1),
            ("2026-09-08T12:00:00Z", "KC", None, None, "QB", "QB", 1),
        ]
    )

    result = normalize_nflverse_depth_charts(source)

    assert result.columns.tolist() == CANONICAL_DEPTH_CHART_COLUMNS
    assert result["position"].tolist() == ["QB", "RB"]
    assert result.loc[result["position"].eq("QB"), "player_id"].isna().all()
    assert isinstance(result["snapshot_timestamp"].dtype, pd.DatetimeTZDtype)
    assert str(result["snapshot_timestamp"].dt.tz) == "UTC"
    assert result.loc[result["position"].eq("RB"), "player_id"].item() == "rb-1"
    assert result.loc[result["position"].eq("RB"), "depth_position"].item() == "RB"
    assert result.loc[result["position"].eq("RB"), "depth_rank"].item() == 1


def test_normalizer_keeps_all_positions_and_only_requires_rb_identity():
    source = raw_depth(
        [
            ("2026-09-08", "KC", "Runner", "rb-1", "RB", "RB", 1),
            ("2026-09-08", "KC", None, None, "WR", None, "unknown"),
        ]
    )

    result = normalize_nflverse_depth_charts(source)

    assert result["position"].tolist() == ["RB", "WR"]
    assert pd.isna(result.loc[result["position"].eq("WR"), "player_id"].item())


@pytest.mark.parametrize("column", ["team", "pos_abb"])
@pytest.mark.parametrize("value", [None, "", "   ", 12])
def test_normalizer_requires_global_team_and_position_text(column, value):
    source = raw_depth(
        [("2026-09-08", "KC", "Runner", "rb-1", "RB", "RB", 1)]
    )
    source.loc[0, column] = value

    with pytest.raises(ValueError, match="nonblank strings"):
        normalize_nflverse_depth_charts(source)


@pytest.mark.parametrize("value", [None, "not-a-date"])
def test_normalizer_requires_valid_snapshot_timestamps(value):
    source = raw_depth([(value, "KC", "Runner", "rb-1", "RB", "RB", 1)])

    with pytest.raises(ValueError, match="snapshot timestamps"):
        normalize_nflverse_depth_charts(source)


@pytest.mark.parametrize("column", ["gsis_id", "player_name"])
@pytest.mark.parametrize("value", [None, "", "   ", 12])
def test_normalizer_requires_rb_identity(column, value):
    source = raw_depth(
        [("2026-09-08", "KC", "Runner", "rb-1", "RB", "RB", 1)]
    )
    source.loc[0, column] = value

    with pytest.raises(ValueError, match="nonblank strings"):
        normalize_nflverse_depth_charts(source)


@pytest.mark.parametrize("rank_dtype", ["Int64", "Float64"])
def test_nullable_rb_ranks_are_preserved_and_sorted_last(rank_dtype):
    source = raw_depth(
        [
            ("2026-09-08", "KC", "Unranked", "rb-2", "RB", "RB", pd.NA),
            ("2026-09-08", "KC", "Ranked", "rb-1", "RB", "RB", 1),
        ]
    )
    source["pos_rank"] = source["pos_rank"].astype(rank_dtype)

    result = normalize_nflverse_depth_charts(source)

    assert result["player_id"].tolist() == ["rb-1", "rb-2"]
    assert result.loc[result["player_id"].eq("rb-2"), "depth_rank"].isna().all()
    public_ranks = result["depth_rank"].dropna().astype("string").tolist()
    assert "<missing>" not in public_ranks


def test_categorical_depth_positions_are_preserved_without_numeric_conversion():
    source = raw_depth(
        [
            ("2026-09-08", "KC", "Primary", "rb-1", "RB", "RB", 1),
            ("2026-09-08", "KC", "Passing", "rb-2", "RB", "3DRB", 2),
        ]
    )

    result = normalize_nflverse_depth_charts(source)

    assert result["depth_position"].tolist() == ["RB", "3DRB"]
    assert result["depth_rank"].tolist() == [1, 2]
    assert all(isinstance(value, str) for value in result["depth_position"])


@pytest.mark.parametrize("value", [None, pd.NA, "", "   "])
def test_missing_or_blank_depth_position_normalizes_to_missing(value):
    source = raw_depth(
        [("2026-09-08", "KC", "Runner", "rb-1", "RB", value, 1)]
    )

    result = normalize_nflverse_depth_charts(source)

    assert pd.isna(result["depth_position"].item())


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("pos_rank", 0),
        ("pos_rank", 1.5),
        ("pos_rank", True),
        ("pos_rank", np.inf),
        ("pos_rank", "RB"),
        ("pos_slot", 12),
        ("pos_slot", False),
        ("pos_slot", np.inf),
    ],
)
def test_invalid_rb_depth_metadata_is_rejected(column, value):
    source = raw_depth(
        [("2026-09-08", "KC", "Runner", "rb-1", "RB", "RB", 1)]
    )
    source.loc[0, column] = value

    with pytest.raises(ValueError, match="depth_"):
        normalize_nflverse_depth_charts(source)


def test_exact_duplicates_are_removed_but_conflicts_raise():
    row = ("2026-09-08", "KC", "Runner", "rb-1", "RB", "RB", 1)
    duplicate = raw_depth([row, row])

    assert len(normalize_nflverse_depth_charts(duplicate)) == 1

    conflict = raw_depth([row, (*row[:-1], 2)])
    with pytest.raises(ValueError, match="Conflicting NFL RB"):
        normalize_nflverse_depth_charts(conflict)


def test_same_name_different_id_rb_players_are_preserved():
    source = raw_depth(
        [
            ("2026-09-08", "KC", "Same Name", "rb-1", "RB", "RB", 1),
            ("2026-09-08", "KC", "Same Name", "rb-2", "RB", "RB", 2),
        ]
    )

    result = normalize_nflverse_depth_charts(source)

    assert result["player_id"].tolist() == ["rb-1", "rb-2"]


def test_mixed_current_and_legacy_rows_raise_clearly():
    current = raw_depth(
        [("2026-09-08", "KC", "Runner", "rb-1", "RB", "RB", 1)]
    )
    legacy = pd.DataFrame(
        {
            "season": [2024],
            "week": [1],
            "game_type": ["REG"],
            "club_code": ["KC"],
            "full_name": ["Old Runner"],
            "gsis_id": ["old-rb"],
            "position": ["RB"],
            "depth_team": [1],
        }
    )
    mixed = pd.concat([current, legacy], ignore_index=True)

    with pytest.raises(ValueError, match="Mixed depth-chart schemas"):
        normalize_nflverse_depth_charts(mixed)


def test_mixed_current_and_canonical_rows_raise_clearly():
    current = raw_depth(
        [("2026-09-08", "KC", "Runner", "rb-1", "RB", "RB", 1)]
    )
    canonical = pd.DataFrame(
        [
            ("BUF", "rb-2", "Other", "RB", "2026-09-08", "RB", 1),
        ],
        columns=CANONICAL_DEPTH_CHART_COLUMNS,
    )
    mixed = pd.concat([current, canonical], ignore_index=True)

    with pytest.raises(ValueError, match="Mixed depth-chart schemas"):
        normalize_nflverse_depth_charts(mixed)


def test_legacy_schema_is_explicitly_unsupported():
    legacy = pd.DataFrame(
        {
            "season": [2024],
            "week": [1],
            "game_type": ["REG"],
            "club_code": ["KC"],
            "full_name": ["Old Runner"],
            "gsis_id": ["old-rb"],
            "position": ["RB"],
            "depth_team": [1],
        }
    )

    with pytest.raises(ValueError, match=r"only the 2025\+ nflverse dated schema"):
        normalize_nflverse_depth_charts(legacy)


def test_missing_current_columns_raise_clearly():
    source = pd.DataFrame({"dt": ["2026-09-08"], "team": ["KC"]})

    with pytest.raises(ValueError, match="missing required columns"):
        normalize_nflverse_depth_charts(source)


def test_empty_valid_source_returns_complete_empty_schema():
    source = pd.DataFrame(columns=NFLVERSE_DATED_DEPTH_CHART_COLUMNS)

    result = normalize_nflverse_depth_charts(source)

    assert result.empty
    assert result.columns.tolist() == CANONICAL_DEPTH_CHART_COLUMNS


def test_normalization_is_deterministic_and_does_not_mutate_input():
    source = raw_depth(
        [
            ("2026-09-08T13:00:00Z", "KC", "Second", "rb-2", "RB", "3DRB", 2),
            ("2026-09-08T12:00:00Z", "BUF", "First", "rb-1", "RB", "RB", 1),
            ("2026-09-08T13:00:00Z", "KC", None, None, "WR", None, None),
        ]
    )
    original = source.copy(deep=True)

    forward = normalize_nflverse_depth_charts(source)
    reverse = normalize_nflverse_depth_charts(source.iloc[::-1])

    pd.testing.assert_frame_equal(forward, reverse)
    pd.testing.assert_frame_equal(source, original)


def test_realistic_current_nflverse_row_normalizes_with_categorical_slot():
    source = pd.DataFrame(
        [
            {
                "dt": "2026-09-08T12:00:00Z",
                "team": "KC",
                "player_name": "Kansas City Runner",
                "espn_id": "1001",
                "gsis_id": "00-001001",
                "pos_grp_id": "1",
                "pos_grp": "Offense",
                "pos_id": "2",
                "pos_name": "Running Back",
                "pos_abb": "RB",
                "pos_slot": "RB",
                "pos_rank": 1,
            }
        ]
    )

    result = normalize_nflverse_depth_charts(source)

    assert result.loc[0, "depth_position"] == "RB"
    assert result.loc[0, "depth_rank"] == 1
