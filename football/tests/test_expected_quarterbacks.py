import pandas as pd
import pytest

from football.features.expected_quarterbacks import (
    OUTPUT_COLUMNS,
    resolve_expected_quarterbacks as _resolve_expected_quarterbacks,
)


AS_OF_DATE = "2026-09-09"


def resolve_expected_quarterbacks(*args, **kwargs) -> pd.DataFrame:
    kwargs.setdefault("as_of_date", AS_OF_DATE)
    return _resolve_expected_quarterbacks(*args, **kwargs)


def make_schedule() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_KC_LAC",
                "game_date": "2026-09-10",
                "game_time": "20:20",
                "team": "KC",
                "opponent": "LAC",
                "home_away": "home",
            },
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_KC_LAC",
                "game_date": "2026-09-10",
                "game_time": "20:20",
                "team": "LAC",
                "opponent": "KC",
                "home_away": "away",
            },
        ]
    )


def make_depth_charts() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "game_type": "REG",
                "club_code": "KC",
                "gsis_id": "qb_kc_2",
                "full_name": "Kansas City Backup",
                "position": "QB",
                "depth_team": 2,
                "formation": "Shotgun",
            },
            {
                "season": 2026,
                "week": 1,
                "game_type": "REG",
                "club_code": "KC",
                "gsis_id": "qb_kc_1",
                "full_name": "Kansas City Starter",
                "position": "QB",
                "depth_team": 1,
                "formation": "Singleback",
            },
            {
                "season": 2026,
                "week": 1,
                "game_type": "REG",
                "club_code": "LAC",
                "gsis_id": "qb_lac_1",
                "full_name": "Los Angeles Starter",
                "position": "QB",
                "depth_team": 1,
                "formation": "Shotgun",
            },
        ]
    )


def make_current_depth_charts() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dt": "2026-09-08",
                "team": "KC",
                "player_name": "Kansas City Backup",
                "espn_id": "1002",
                "gsis_id": "qb_kc_2",
                "pos_grp_id": 1,
                "pos_grp": "Offense",
                "pos_id": 1,
                "pos_name": "Quarterback",
                "pos_abb": "QB",
                "pos_slot": "QB",
                "pos_rank": 2,
            },
            {
                "dt": "2026-09-08",
                "team": "KC",
                "player_name": "Kansas City Starter",
                "espn_id": "1001",
                "gsis_id": "qb_kc_1",
                "pos_grp_id": 1,
                "pos_grp": "Offense",
                "pos_id": 1,
                "pos_name": "Quarterback",
                "pos_abb": "QB",
                "pos_slot": "QB",
                "pos_rank": 1,
            },
            {
                "dt": "2026-09-07",
                "team": "LAC",
                "player_name": "Los Angeles Starter",
                "espn_id": "2001",
                "gsis_id": "qb_lac_1",
                "pos_grp_id": 1,
                "pos_grp": "Offense",
                "pos_id": 1,
                "pos_name": "Quarterback",
                "pos_abb": "QB",
                "pos_slot": "QB",
                "pos_rank": 1,
            },
        ]
    )


def get_team(result: pd.DataFrame, team: str) -> pd.Series:
    return result.loc[result["team"] == team].iloc[0]


def test_resolve_expected_quarterbacks_selects_depth_chart_starter():
    result = resolve_expected_quarterbacks(
        make_schedule(),
        make_depth_charts(),
        2026,
        1,
    )

    row = get_team(result, "KC")

    assert row["expected_player_id"] == "qb_kc_1"
    assert row["expected_player_name"] == "Kansas City Starter"
    assert row["selection_source"] == "depth_chart"
    assert row["depth_rank"] == 1
    assert not row["starter_uncertain"]


def test_resolve_expected_quarterbacks_removes_duplicate_formation_rows():
    depth_charts = pd.concat(
        [
            make_depth_charts(),
            pd.DataFrame(
                [
                    {
                        "season": 2026,
                        "week": 1,
                        "game_type": "REG",
                        "club_code": "KC",
                        "gsis_id": "qb_kc_1",
                        "full_name": "Kansas City Starter",
                        "position": "QB",
                        "depth_team": 1,
                        "formation": "I-Form",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    result = resolve_expected_quarterbacks(
        make_schedule(),
        depth_charts,
        2026,
        1,
    )

    assert len(result.loc[result["team"] == "KC"]) == 1
    assert not get_team(result, "KC")["starter_uncertain"]


def test_resolve_expected_quarterbacks_uses_best_numeric_depth_rank():
    depth_charts = make_depth_charts()
    depth_charts.loc[depth_charts["gsis_id"] == "qb_kc_1", "depth_team"] = 3
    depth_charts.loc[depth_charts["gsis_id"] == "qb_kc_2", "depth_team"] = 2

    result = resolve_expected_quarterbacks(
        make_schedule(),
        depth_charts,
        2026,
        1,
    )

    row = get_team(result, "KC")

    assert row["expected_player_id"] == "qb_kc_2"
    assert row["depth_rank"] == 2


def test_resolve_expected_quarterbacks_breaks_tied_depth_ranks_by_player_id():
    depth_charts = pd.concat(
        [
            make_depth_charts(),
            pd.DataFrame(
                [
                    {
                        "season": 2026,
                        "week": 1,
                        "game_type": "REG",
                        "club_code": "KC",
                        "gsis_id": "qb_kc_0",
                        "full_name": "Kansas City Tie",
                        "position": "QB",
                        "depth_team": 1,
                        "formation": "Shotgun",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    result = resolve_expected_quarterbacks(
        make_schedule(),
        depth_charts,
        2026,
        1,
    )

    row = get_team(result, "KC")

    assert row["expected_player_id"] == "qb_kc_0"
    assert row["starter_uncertain"]


def test_resolve_expected_quarterbacks_applies_manual_override_precedence():
    overrides = pd.DataFrame(
        [
            {
                "season": 2026,
                "report_week": 1,
                "team": "KC",
                "player_id": "qb_kc_override",
                "player_name": "Kansas City Override",
                "selection_notes": "Coach announced starter.",
            }
        ]
    )

    result = resolve_expected_quarterbacks(
        make_schedule(),
        make_depth_charts(),
        2026,
        1,
        manual_overrides=overrides,
    )

    row = get_team(result, "KC")

    assert row["expected_player_id"] == "qb_kc_override"
    assert row["expected_player_name"] == "Kansas City Override"
    assert row["selection_source"] == "manual_override"
    assert not row["starter_uncertain"]
    assert row["selection_notes"] == "Coach announced starter."


def test_resolve_expected_quarterbacks_rejects_conflicting_manual_overrides():
    overrides = pd.DataFrame(
        [
            {
                "season": 2026,
                "report_week": 1,
                "team": "KC",
                "player_id": "qb_kc_a",
                "player_name": "Kansas City A",
                "selection_notes": "Source A",
            },
            {
                "season": 2026,
                "report_week": 1,
                "team": "KC",
                "player_id": "qb_kc_b",
                "player_name": "Kansas City B",
                "selection_notes": "Source B",
            },
        ]
    )

    with pytest.raises(ValueError, match="Conflicting manual quarterback"):
        resolve_expected_quarterbacks(
            make_schedule(),
            make_depth_charts(),
            2026,
            1,
            manual_overrides=overrides,
        )


def test_resolve_expected_quarterbacks_keeps_team_missing_from_depth_charts():
    depth_charts = make_depth_charts().loc[
        lambda data: data["club_code"] != "LAC"
    ]

    result = resolve_expected_quarterbacks(
        make_schedule(),
        depth_charts,
        2026,
        1,
    )

    row = get_team(result, "LAC")

    assert pd.isna(row["expected_player_id"])
    assert pd.isna(row["expected_player_name"])
    assert row["selection_source"] == "unresolved"
    assert row["starter_uncertain"]


def test_resolve_expected_quarterbacks_keeps_team_with_no_quarterback():
    depth_charts = make_depth_charts()
    depth_charts.loc[depth_charts["club_code"] == "LAC", "position"] = "WR"

    result = resolve_expected_quarterbacks(
        make_schedule(),
        depth_charts,
        2026,
        1,
    )

    assert get_team(result, "LAC")["selection_source"] == "unresolved"


def test_resolve_expected_quarterbacks_excludes_other_positions():
    depth_charts = pd.concat(
        [
            make_depth_charts(),
            pd.DataFrame(
                [
                    {
                        "season": 2026,
                        "week": 1,
                        "game_type": "REG",
                        "club_code": "KC",
                        "gsis_id": "rb_kc_1",
                        "full_name": "Kansas City Running Back",
                        "position": "RB",
                        "depth_team": 1,
                        "formation": "Singleback",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    result = resolve_expected_quarterbacks(
        make_schedule(),
        depth_charts,
        2026,
        1,
    )

    assert get_team(result, "KC")["expected_player_id"] == "qb_kc_1"


def test_resolve_expected_quarterbacks_excludes_other_seasons_and_weeks():
    depth_charts = pd.concat(
        [
            make_depth_charts(),
            pd.DataFrame(
                [
                    {
                        "season": 2025,
                        "week": 1,
                        "game_type": "REG",
                        "club_code": "KC",
                        "gsis_id": "qb_old_season",
                        "full_name": "Old Season",
                        "position": "QB",
                        "depth_team": 1,
                        "formation": "Shotgun",
                    },
                    {
                        "season": 2026,
                        "week": 2,
                        "game_type": "REG",
                        "club_code": "KC",
                        "gsis_id": "qb_other_week",
                        "full_name": "Other Week",
                        "position": "QB",
                        "depth_team": 1,
                        "formation": "Shotgun",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )

    result = resolve_expected_quarterbacks(
        make_schedule(),
        depth_charts,
        2026,
        1,
    )

    assert get_team(result, "KC")["expected_player_id"] == "qb_kc_1"


def test_resolve_expected_quarterbacks_excludes_postseason_depth_chart_rows():
    depth_charts = pd.concat(
        [
            make_depth_charts(),
            pd.DataFrame(
                [
                    {
                        "season": 2026,
                        "week": 1,
                        "game_type": "POST",
                        "club_code": "KC",
                        "gsis_id": "qb_postseason",
                        "full_name": "Postseason QB",
                        "position": "QB",
                        "depth_team": 1,
                        "formation": "Shotgun",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    result = resolve_expected_quarterbacks(
        make_schedule(),
        depth_charts,
        2026,
        1,
    )

    assert get_team(result, "KC")["expected_player_id"] == "qb_kc_1"


def test_resolve_expected_quarterbacks_returns_both_teams_from_game():
    result = resolve_expected_quarterbacks(
        make_schedule(),
        make_depth_charts(),
        2026,
        1,
    )

    assert result["team"].tolist() == ["KC", "LAC"]
    assert len(result) == 2


def test_resolve_expected_quarterbacks_accepts_current_depth_chart_mapping():
    depth_charts = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "season_type": "REG",
                "team": "KC",
                "gsis_id": "qb_kc_new",
                "player_name": "Kansas City Current",
                "pos_abb": "QB",
                "pos_rank": 1,
            },
            {
                "season": 2026,
                "week": 1,
                "season_type": "REG",
                "team": "LAC",
                "gsis_id": "qb_lac_new",
                "player_name": "Los Angeles Current",
                "pos_abb": "QB",
                "pos_rank": 1,
            },
        ]
    )

    result = resolve_expected_quarterbacks(
        make_schedule(),
        depth_charts,
        2026,
        1,
    )

    assert get_team(result, "KC")["expected_player_id"] == "qb_kc_new"


def test_resolve_expected_quarterbacks_accepts_current_dated_snapshot_schema():
    result = resolve_expected_quarterbacks(
        make_schedule(),
        make_current_depth_charts(),
        2026,
        1,
    )

    row = get_team(result, "KC")

    assert row["expected_player_id"] == "qb_kc_1"
    assert row["expected_player_name"] == "Kansas City Starter"
    assert row["selection_source"] == "depth_chart"
    assert row["depth_rank"] == 1
    assert row["depth_chart_date"] == pd.Timestamp("2026-09-08")


def test_resolve_expected_quarterbacks_uses_latest_snapshot_on_or_before_as_of_date():
    depth_charts = pd.concat(
        [
            make_current_depth_charts(),
            pd.DataFrame(
                [
                    {
                        "dt": "2026-09-09",
                        "team": "KC",
                        "player_name": "Kansas City Updated Starter",
                        "espn_id": "1003",
                        "gsis_id": "qb_kc_3",
                        "pos_grp_id": 1,
                        "pos_grp": "Offense",
                        "pos_id": 1,
                        "pos_name": "Quarterback",
                        "pos_abb": "QB",
                        "pos_slot": "QB",
                        "pos_rank": 1,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    result = resolve_expected_quarterbacks(
        make_schedule(),
        depth_charts,
        2026,
        1,
        as_of_date="2026-09-09",
    )

    row = get_team(result, "KC")

    assert row["expected_player_id"] == "qb_kc_3"
    assert row["depth_chart_date"] == pd.Timestamp("2026-09-09")


def test_resolve_expected_quarterbacks_accepts_utc_aware_depth_dates_with_plain_as_of_date():
    depth_charts = make_current_depth_charts()
    depth_charts["dt"] = pd.to_datetime(depth_charts["dt"], utc=True)

    result = resolve_expected_quarterbacks(
        make_schedule(),
        depth_charts,
        2026,
        1,
        as_of_date="2026-09-08",
    )

    row = get_team(result, "KC")

    assert row["expected_player_id"] == "qb_kc_1"
    assert row["depth_chart_date"] == pd.Timestamp("2026-09-08")
    assert row["depth_chart_date"].tzinfo is None


def test_resolve_expected_quarterbacks_includes_same_utc_calendar_date_snapshot():
    depth_charts = pd.concat(
        [
            make_current_depth_charts(),
            pd.DataFrame(
                [
                    {
                        "dt": "2026-09-09T23:59:59Z",
                        "team": "KC",
                        "player_name": "Kansas City Same Day Starter",
                        "espn_id": "1005",
                        "gsis_id": "qb_kc_same_day",
                        "pos_grp_id": 1,
                        "pos_grp": "Offense",
                        "pos_id": 1,
                        "pos_name": "Quarterback",
                        "pos_abb": "QB",
                        "pos_slot": "QB",
                        "pos_rank": 1,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    result = resolve_expected_quarterbacks(
        make_schedule(),
        depth_charts,
        2026,
        1,
        as_of_date="2026-09-09",
    )

    row = get_team(result, "KC")

    assert row["expected_player_id"] == "qb_kc_same_day"
    assert row["depth_chart_date"] == pd.Timestamp("2026-09-09")


def test_resolve_expected_quarterbacks_excludes_next_utc_calendar_date_snapshot():
    depth_charts = pd.concat(
        [
            make_current_depth_charts(),
            pd.DataFrame(
                [
                    {
                        "dt": "2026-09-10T00:00:00Z",
                        "team": "KC",
                        "player_name": "Kansas City Next Day Starter",
                        "espn_id": "1006",
                        "gsis_id": "qb_kc_next_day",
                        "pos_grp_id": 1,
                        "pos_grp": "Offense",
                        "pos_id": 1,
                        "pos_name": "Quarterback",
                        "pos_abb": "QB",
                        "pos_slot": "QB",
                        "pos_rank": 1,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    result = resolve_expected_quarterbacks(
        make_schedule(),
        depth_charts,
        2026,
        1,
        as_of_date="2026-09-09",
    )

    assert get_team(result, "KC")["expected_player_id"] == "qb_kc_1"


def test_resolve_expected_quarterbacks_accepts_timezone_aware_as_of_date():
    depth_charts = pd.concat(
        [
            make_current_depth_charts(),
            pd.DataFrame(
                [
                    {
                        "dt": "2026-09-09T00:00:00Z",
                        "team": "KC",
                        "player_name": "Kansas City UTC Starter",
                        "espn_id": "1007",
                        "gsis_id": "qb_kc_utc",
                        "pos_grp_id": 1,
                        "pos_grp": "Offense",
                        "pos_id": 1,
                        "pos_name": "Quarterback",
                        "pos_abb": "QB",
                        "pos_slot": "QB",
                        "pos_rank": 1,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    result = resolve_expected_quarterbacks(
        make_schedule(),
        depth_charts,
        2026,
        1,
        as_of_date=pd.Timestamp("2026-09-08 20:00:00", tz="America/Chicago"),
    )

    row = get_team(result, "KC")

    assert row["expected_player_id"] == "qb_kc_utc"
    assert row["depth_chart_date"] == pd.Timestamp("2026-09-09")


def test_resolve_expected_quarterbacks_excludes_future_snapshots():
    depth_charts = pd.concat(
        [
            make_current_depth_charts(),
            pd.DataFrame(
                [
                    {
                        "dt": "2026-09-10",
                        "team": "KC",
                        "player_name": "Kansas City Future Starter",
                        "espn_id": "1004",
                        "gsis_id": "qb_kc_future",
                        "pos_grp_id": 1,
                        "pos_grp": "Offense",
                        "pos_id": 1,
                        "pos_name": "Quarterback",
                        "pos_abb": "QB",
                        "pos_slot": "QB",
                        "pos_rank": 1,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    result = resolve_expected_quarterbacks(
        make_schedule(),
        depth_charts,
        2026,
        1,
        as_of_date="2026-09-09",
    )

    assert get_team(result, "KC")["expected_player_id"] == "qb_kc_1"


def test_resolve_expected_quarterbacks_uses_each_teams_latest_snapshot_date():
    result = resolve_expected_quarterbacks(
        make_schedule(),
        make_current_depth_charts(),
        2026,
        1,
        as_of_date="2026-09-09",
    )

    assert get_team(result, "KC")["depth_chart_date"] == pd.Timestamp("2026-09-08")
    assert get_team(result, "LAC")["depth_chart_date"] == pd.Timestamp("2026-09-07")


def test_resolve_expected_quarterbacks_rejects_invalid_current_snapshot_dates():
    depth_charts = make_current_depth_charts()
    depth_charts.loc[0, "dt"] = "not-a-date"

    with pytest.raises(ValueError, match="invalid dt values"):
        resolve_expected_quarterbacks(
            make_schedule(),
            depth_charts,
            2026,
            1,
        )


@pytest.mark.parametrize("as_of_date", [None, "", "not-a-date"])
def test_resolve_expected_quarterbacks_validates_as_of_date(as_of_date):
    with pytest.raises(ValueError, match="as_of_date"):
        _resolve_expected_quarterbacks(
            make_schedule(),
            make_current_depth_charts(),
            2026,
            1,
            as_of_date=as_of_date,
        )


def test_resolve_expected_quarterbacks_filters_current_schema_to_quarterbacks():
    depth_charts = pd.concat(
        [
            make_current_depth_charts(),
            pd.DataFrame(
                [
                    {
                        "dt": "2026-09-08",
                        "team": "KC",
                        "player_name": "Kansas City Running Back",
                        "espn_id": "3001",
                        "gsis_id": "rb_kc_1",
                        "pos_grp_id": 1,
                        "pos_grp": "Offense",
                        "pos_id": 2,
                        "pos_name": "Running Back",
                        "pos_abb": "RB",
                        "pos_slot": "RB",
                        "pos_rank": 1,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    result = resolve_expected_quarterbacks(
        make_schedule(),
        depth_charts,
        2026,
        1,
    )

    assert get_team(result, "KC")["expected_player_id"] == "qb_kc_1"


def test_resolve_expected_quarterbacks_selects_lowest_current_schema_rank():
    depth_charts = make_current_depth_charts()
    depth_charts.loc[depth_charts["gsis_id"] == "qb_kc_1", "pos_rank"] = 3
    depth_charts.loc[depth_charts["gsis_id"] == "qb_kc_2", "pos_rank"] = 2

    result = resolve_expected_quarterbacks(
        make_schedule(),
        depth_charts,
        2026,
        1,
    )

    row = get_team(result, "KC")

    assert row["expected_player_id"] == "qb_kc_2"
    assert row["depth_rank"] == 2


def test_resolve_expected_quarterbacks_manual_override_clears_depth_chart_date():
    overrides = pd.DataFrame(
        [
            {
                "season": 2026,
                "report_week": 1,
                "team": "KC",
                "player_id": "qb_kc_override",
                "player_name": "Kansas City Override",
                "selection_notes": "Coach announced starter.",
            }
        ]
    )

    result = resolve_expected_quarterbacks(
        make_schedule(),
        make_current_depth_charts(),
        2026,
        1,
        manual_overrides=overrides,
    )

    row = get_team(result, "KC")

    assert row["selection_source"] == "manual_override"
    assert pd.isna(row["depth_chart_date"])


def test_resolve_expected_quarterbacks_manual_override_supports_source_date():
    overrides = pd.DataFrame(
        [
            {
                "season": 2026,
                "report_week": 1,
                "team": "KC",
                "player_id": "qb_kc_override",
                "player_name": "Kansas City Override",
                "selection_notes": "Coach announced starter.",
                "depth_chart_date": "2026-09-09",
            }
        ]
    )

    result = resolve_expected_quarterbacks(
        make_schedule(),
        make_current_depth_charts(),
        2026,
        1,
        manual_overrides=overrides,
    )

    row = get_team(result, "KC")

    assert row["selection_source"] == "manual_override"
    assert row["depth_chart_date"] == pd.Timestamp("2026-09-09")


def test_resolve_expected_quarterbacks_current_schema_keeps_unresolved_teams():
    depth_charts = make_current_depth_charts().loc[lambda data: data["team"] != "LAC"]

    result = resolve_expected_quarterbacks(
        make_schedule(),
        depth_charts,
        2026,
        1,
    )

    row = get_team(result, "LAC")

    assert pd.isna(row["expected_player_id"])
    assert pd.isna(row["depth_chart_date"])
    assert row["selection_source"] == "unresolved"


def test_resolve_expected_quarterbacks_accepts_legacy_weekly_contract_without_source_date():
    depth_charts = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "game_type": "REG",
                "team": "KC",
                "player_id": "qb_kc_legacy",
                "player_name": "Kansas City Legacy",
                "position": "QB",
                "depth_rank": 1,
            },
            {
                "season": 2026,
                "week": 1,
                "game_type": "REG",
                "team": "LAC",
                "player_id": "qb_lac_legacy",
                "player_name": "Los Angeles Legacy",
                "position": "QB",
                "depth_rank": 1,
            },
        ]
    )

    result = resolve_expected_quarterbacks(
        make_schedule(),
        depth_charts,
        2026,
        1,
    )

    row = get_team(result, "KC")

    assert row["expected_player_id"] == "qb_kc_legacy"
    assert pd.isna(row["depth_chart_date"])


def test_resolve_expected_quarterbacks_validates_missing_schedule_columns():
    schedule = make_schedule().drop(columns=["game_time", "opponent"])

    with pytest.raises(ValueError) as error:
        resolve_expected_quarterbacks(
            schedule,
            make_depth_charts(),
            2026,
            1,
        )

    message = str(error.value)

    assert "Schedule data is missing required columns" in message
    assert "game_time" in message
    assert "opponent" in message


def test_resolve_expected_quarterbacks_validates_missing_depth_columns():
    depth_charts = make_depth_charts().drop(columns=["depth_team", "full_name"])

    with pytest.raises(ValueError) as error:
        resolve_expected_quarterbacks(
            make_schedule(),
            depth_charts,
            2026,
            1,
        )

    message = str(error.value)

    assert "NFL depth-chart data is missing required columns" in message
    assert "depth_team" in message
    assert "full_name" in message


def test_resolve_expected_quarterbacks_validates_missing_override_columns():
    overrides = pd.DataFrame(
        [
            {
                "season": 2026,
                "report_week": 1,
                "team": "KC",
                "player_id": "qb_kc_override",
            }
        ]
    )

    with pytest.raises(ValueError) as error:
        resolve_expected_quarterbacks(
            make_schedule(),
            make_depth_charts(),
            2026,
            1,
            manual_overrides=overrides,
        )

    message = str(error.value)

    assert "Manual quarterback override data is missing required columns" in message
    assert "player_name" in message
    assert "selection_notes" in message


@pytest.mark.parametrize(
    ("report_season", "report_week", "expected_message"),
    [
        (True, 1, "report_season"),
        ("2026", 1, "report_season"),
        (0, 1, "report_season"),
        (2026, False, "report_week"),
        (2026, 1.5, "report_week"),
        (2026, 0, "report_week"),
    ],
)
def test_resolve_expected_quarterbacks_validates_report_values(
    report_season,
    report_week,
    expected_message,
):
    with pytest.raises(ValueError, match=expected_message):
        resolve_expected_quarterbacks(
            make_schedule(),
            make_depth_charts(),
            report_season,
            report_week,
        )


def test_resolve_expected_quarterbacks_does_not_mutate_inputs():
    schedule = make_schedule()
    depth_charts = make_depth_charts()
    overrides = pd.DataFrame(
        [
            {
                "season": 2026,
                "report_week": 1,
                "team": "KC",
                "player_id": "qb_kc_override",
                "player_name": "Kansas City Override",
                "selection_notes": "Coach announced starter.",
            }
        ]
    )
    original_schedule = schedule.copy(deep=True)
    original_depth_charts = depth_charts.copy(deep=True)
    original_overrides = overrides.copy(deep=True)

    resolve_expected_quarterbacks(
        schedule,
        depth_charts,
        2026,
        1,
        manual_overrides=overrides,
    )

    pd.testing.assert_frame_equal(schedule, original_schedule)
    pd.testing.assert_frame_equal(depth_charts, original_depth_charts)
    pd.testing.assert_frame_equal(overrides, original_overrides)


def test_resolve_expected_quarterbacks_outputs_expected_columns():
    result = resolve_expected_quarterbacks(
        make_schedule(),
        make_depth_charts(),
        2026,
        1,
    )

    assert result.columns.tolist() == OUTPUT_COLUMNS
