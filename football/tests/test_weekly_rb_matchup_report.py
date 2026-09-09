import numpy as np
import pandas as pd
import pytest

from football.features.expected_running_backs import (
    OUTPUT_COLUMNS as EXPECTED_RB_COLUMNS,
)
from football.features.rb_defense_matchup_metrics import (
    OUTPUT_COLUMNS as DEFENSE_COLUMNS,
)
from football.features.rb_form_metrics import OUTPUT_COLUMNS as RB_FORM_COLUMNS
from football.reports.build_weekly_rb_matchup_report import (
    OUTPUT_COLUMNS,
    build_weekly_rb_matchup_report,
)


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
                "home_away": "away",
            },
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_KC_LAC",
                "game_date": "2026-09-10",
                "game_time": "20:20",
                "team": "LAC",
                "opponent": "KC",
                "home_away": "home",
            },
            {
                "season": 2026,
                "week": 2,
                "game_id": "2026_02_KC_BUF",
                "game_date": "2026-09-17",
                "game_time": "13:00",
                "team": "KC",
                "opponent": "BUF",
                "home_away": "home",
            },
        ]
    )


def expected_row(
    team: str,
    player_id: object,
    player_name: object,
    order: object,
    *,
    source: str = "depth_chart",
    missing: bool = False,
) -> dict:
    return {
        "report_season": 2026,
        "report_week": 1,
        "team": team,
        "player_id": player_id,
        "player_name": player_name,
        "position": "RB",
        "participant_order": order,
        "selection_source": source,
        "depth_chart_date": "2026-09-08",
        "depth_chart_week": pd.NA,
        "depth_position": "RB",
        "depth_rank": order,
        "resolution_missing": missing,
        "selection_notes": "" if not missing else "No eligible depth chart",
    }


def make_expected() -> pd.DataFrame:
    return pd.DataFrame(
        [
            expected_row("KC", "rb_kc_1", "Same Name", 1),
            expected_row("KC", "rb_kc_2", "Same Name", 2),
            expected_row(
                "LAC",
                "rb_lac_manual",
                "Manual Runner",
                1,
                source="manual_override",
            ),
        ],
        columns=EXPECTED_RB_COLUMNS,
    )


def rb_metric_row(
    player_id: str,
    *,
    latest_team: str = "KC",
    season_games: int = 4,
    last3_games: int = 3,
    yards: float = 60.0,
) -> dict:
    return {
        "report_season": 2026,
        "report_week": 1,
        "historical_season": 2025,
        "player_id": player_id,
        "player_name": "Historical Name",
        "latest_team": latest_team,
        "rb_season_rushing_yards_avg": yards,
        "rb_last3_rushing_yards_avg": yards + 1,
        "rb_season_rushing_attempts_avg": 12.0,
        "rb_last3_rushing_attempts_avg": 13.0,
        "rb_season_receptions_avg": 2.0,
        "rb_last3_receptions_avg": 3.0,
        "rb_season_targets_avg": 3.0,
        "rb_last3_targets_avg": 4.0,
        "rb_season_receiving_yards_avg": 20.0,
        "rb_last3_receiving_yards_avg": 25.0,
        "rb_season_opportunities_avg": 15.0,
        "rb_last3_opportunities_avg": 17.0,
        "rb_season_carry_share_avg": 0.5,
        "rb_last3_carry_share_avg": 0.55,
        "rb_season_target_share_avg": 0.1,
        "rb_last3_target_share_avg": 0.12,
        "rb_season_opportunity_share_avg": 0.4,
        "rb_last3_opportunity_share_avg": 0.45,
        "rb_season_yards_per_carry": 5.0,
        "rb_last3_yards_per_carry": 5.2,
        "rb_season_games": season_games,
        "rb_last3_games": last3_games,
    }


def make_rb_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            rb_metric_row("rb_kc_1", latest_team="KC", yards=0.0),
            rb_metric_row("rb_kc_2", latest_team="OLD", season_games=2),
            rb_metric_row("rb_lac_manual", latest_team="LAC"),
        ],
        columns=RB_FORM_COLUMNS,
    )


def defense_metric_row(
    defense: str,
    *,
    rank: int,
    season_games: int = 4,
    yards_allowed: float = 100.0,
) -> dict:
    return {
        "report_season": 2026,
        "report_week": 1,
        "historical_season": 2025,
        "defense": defense,
        "defense_season_rb_rushing_attempts_avg_allowed": 20.0,
        "defense_last3_rb_rushing_attempts_avg_allowed": 21.0,
        "defense_season_rb_rushing_yards_avg_allowed": yards_allowed,
        "defense_last3_rb_rushing_yards_avg_allowed": yards_allowed + 5,
        "defense_season_rb_rushing_touchdowns_avg_allowed": 1.0,
        "defense_last3_rb_rushing_touchdowns_avg_allowed": 1.2,
        "defense_season_rb_receptions_avg_allowed": 5.0,
        "defense_last3_rb_receptions_avg_allowed": 6.0,
        "defense_season_rb_targets_avg_allowed": 6.0,
        "defense_last3_rb_targets_avg_allowed": 7.0,
        "defense_season_rb_receiving_yards_avg_allowed": 35.0,
        "defense_last3_rb_receiving_yards_avg_allowed": 40.0,
        "defense_season_rb_receiving_touchdowns_avg_allowed": 0.3,
        "defense_last3_rb_receiving_touchdowns_avg_allowed": 0.4,
        "defense_season_rb_opportunities_avg_allowed": 26.0,
        "defense_last3_rb_opportunities_avg_allowed": 28.0,
        "defense_season_rb_players_used_avg": 2.0,
        "defense_last3_rb_players_used_avg": 2.3,
        "defense_season_rb_yards_per_carry_allowed": 5.0,
        "defense_last3_rb_yards_per_carry_allowed": 5.1,
        "defense_season_games": season_games,
        "defense_last3_games": min(season_games, 3),
        "matchup_rank": rank,
    }


def make_defense_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            defense_metric_row("LAC", rank=1, yards_allowed=0.0),
            defense_metric_row("KC", rank=8, season_games=2),
        ],
        columns=DEFENSE_COLUMNS,
    )


def build(
    schedule=None,
    expected=None,
    rbs=None,
    defenses=None,
    *,
    report_season=2026,
    report_week=1,
) -> pd.DataFrame:
    return build_weekly_rb_matchup_report(
        make_schedule() if schedule is None else schedule,
        make_expected() if expected is None else expected,
        make_rb_metrics() if rbs is None else rbs,
        make_defense_metrics() if defenses is None else defenses,
        report_season,
        report_week,
    )


def test_one_scheduled_expected_rb_produces_one_summary_row():
    expected = pd.DataFrame([expected_row("KC", "rb_kc_1", "Runner", 1)])

    result = build(expected=expected)

    assert result.columns.tolist() == OUTPUT_COLUMNS
    assert result.loc[0, ["team", "player_id", "opponent"]].tolist() == [
        "KC",
        "rb_kc_1",
        "LAC",
    ]
    assert result.loc[1, ["team", "opponent"]].tolist() == ["LAC", "KC"]
    assert pd.isna(result.loc[1, "player_id"])


def test_multiple_expected_rbs_and_backups_are_preserved():
    result = build()

    kc = result.loc[result["team"].eq("KC")]
    assert kc["player_id"].tolist() == ["rb_kc_1", "rb_kc_2"]
    assert kc["multiple_expected_rbs"].tolist() == [True, True]


def test_both_teams_in_one_game_produce_participant_rows():
    result = build()

    assert result["team"].tolist() == ["KC", "KC", "LAC"]
    assert result["opponent"].tolist() == ["LAC", "LAC", "KC"]


def test_defense_metrics_join_by_schedule_opponent():
    result = build()

    kc = result.loc[result["team"].eq("KC")].iloc[0]
    lac = result.loc[result["team"].eq("LAC")].iloc[0]
    assert kc["defense_season_rb_rushing_yards_avg_allowed"] == 0.0
    assert kc["matchup_rank"] == 1
    assert lac["matchup_rank"] == 8


def test_rb_form_joins_by_player_id_not_name():
    result = build()

    kc = result.loc[
        result["team"].eq("KC") & result["player_id"].eq("rb_kc_1")
    ].iloc[0]
    assert kc["player_name"] == "Same Name"
    assert "Historical Name" not in result.columns
    assert kc["rb_season_rushing_yards_avg"] == 0.0


def test_same_player_names_with_different_ids_remain_separate():
    result = build()

    kc = result.loc[result["team"].eq("KC")]
    assert kc["player_name"].tolist() == ["Same Name", "Same Name"]
    assert kc["rb_season_rushing_yards_avg"].tolist() == [0.0, 60.0]


def test_all_approved_metric_columns_and_participant_context_are_preserved():
    result = build()

    for column in RB_FORM_COLUMNS:
        if column not in ["report_season", "report_week", "historical_season", "player_id", "player_name"]:
            assert column in result.columns
    for column in DEFENSE_COLUMNS:
        if column not in ["report_season", "report_week", "historical_season", "defense"]:
            assert column in result.columns
    assert result.loc[result["team"].eq("LAC"), "selection_source"].item() == "manual_override"
    assert "depth_chart_date" in result.columns
    assert "depth_position" in result.columns


def test_unresolved_participant_row_is_preserved():
    expected = pd.DataFrame(
        [expected_row("KC", pd.NA, pd.NA, pd.NA, missing=True)],
        columns=EXPECTED_RB_COLUMNS,
    )

    result = build(expected=expected)

    kc = result.loc[result["team"].eq("KC")].iloc[0]
    assert pd.isna(kc["player_id"])
    assert bool(kc["participant_resolution_missing"]) is True
    assert bool(kc["rb_history_missing"]) is False


def test_mixed_resolved_and_missing_id_participants_are_preserved():
    expected = pd.DataFrame(
        [
            expected_row("KC", "rb_kc_1", "Runner", 1),
            expected_row("KC", pd.NA, "Unidentified", 2, missing=True),
        ],
        columns=EXPECTED_RB_COLUMNS,
    )
    expected.loc[1, "selection_notes"] = (
        "Selected depth-chart participant lacks a stable player ID"
    )

    result = build(expected=expected)

    assert result.loc[result["team"].eq("KC")].shape[0] == 2
    missing = result.loc[result["player_id"].isna()].iloc[0]
    assert bool(missing["participant_resolution_missing"])
    assert bool(missing["rb_history_missing"])
    assert pd.isna(missing["rb_season_rushing_yards_avg"])
    assert not bool(missing["multiple_expected_rbs"])


def test_resolved_participant_without_rb_history_is_preserved_and_flagged():
    rbs = make_rb_metrics().loc[lambda data: data["player_id"].ne("rb_kc_2")]

    result = build(rbs=rbs)

    row = result.loc[result["player_id"].eq("rb_kc_2")].iloc[0]
    assert bool(row["rb_history_missing"]) is True
    assert pd.isna(row["rb_season_opportunities_avg"])


def test_missing_defense_history_is_preserved_and_flagged():
    defenses = make_defense_metrics().loc[lambda data: data["defense"].ne("KC")]

    result = build(defenses=defenses)

    lac = result.loc[result["team"].eq("LAC")].iloc[0]
    assert bool(lac["defense_history_missing"]) is True
    assert pd.isna(lac["defense_season_rb_rushing_yards_avg_allowed"])


def test_real_zero_metrics_are_not_missing_history():
    result = build()

    kc = result.loc[result["player_id"].eq("rb_kc_1")].iloc[0]
    assert kc["rb_season_rushing_yards_avg"] == 0.0
    assert kc["defense_season_rb_rushing_yards_avg_allowed"] == 0.0
    assert bool(kc["rb_history_missing"]) is False
    assert bool(kc["defense_history_missing"]) is False


def test_team_mismatch_is_warning_only():
    result = build()

    same = result.loc[result["player_id"].eq("rb_kc_1")].iloc[0]
    mismatch = result.loc[result["player_id"].eq("rb_kc_2")].iloc[0]
    assert bool(same["participant_team_mismatch"]) is False
    assert bool(mismatch["participant_team_mismatch"]) is True
    assert mismatch["team"] == "KC"
    assert mismatch["latest_team"] == "OLD"


def test_unresolved_rows_do_not_count_as_multiple_expected_rbs():
    expected = pd.DataFrame(
        [
            expected_row("KC", "rb_kc_1", "Runner", 1),
            expected_row("LAC", pd.NA, pd.NA, pd.NA, missing=True),
        ],
        columns=EXPECTED_RB_COLUMNS,
    )

    result = build(expected=expected)

    assert not bool(
        result.loc[result["team"].eq("KC"), "multiple_expected_rbs"].item()
    )
    assert not bool(
        result.loc[result["team"].eq("LAC"), "multiple_expected_rbs"].item()
    )


def test_limited_sample_warnings_use_fewer_than_three_season_games():
    result = build()

    assert bool(result.loc[result["player_id"].eq("rb_kc_2"), "limited_rb_sample"].item()) is True
    assert bool(result.loc[result["team"].eq("LAC"), "limited_defense_sample"].item()) is True
    assert bool(result.loc[result["player_id"].eq("rb_kc_1"), "limited_rb_sample"].item()) is False


def test_report_and_historical_context_are_preserved():
    result = build()

    assert result["report_season"].eq(2026).all()
    assert result["report_week"].eq(1).all()
    assert result["historical_season"].eq(2025).all()


def test_inconsistent_historical_seasons_raise():
    defenses = make_defense_metrics()
    defenses["historical_season"] = 2024

    with pytest.raises(ValueError, match="same historical_season"):
        build(defenses=defenses)


@pytest.mark.parametrize(
    ("input_name", "column"),
    [
        ("expected", "report_week"),
        ("rbs", "report_week"),
        ("defenses", "report_week"),
    ],
)
def test_mismatched_report_contexts_raise(input_name, column):
    inputs = {
        "expected": make_expected(),
        "rbs": make_rb_metrics(),
        "defenses": make_defense_metrics(),
    }
    inputs[input_name][column] = 2

    with pytest.raises(ValueError, match="outside the requested report context"):
        build(**inputs)


@pytest.mark.parametrize("name", ["report_season", "report_week"])
@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "1"])
def test_invalid_report_values_raise(name, value):
    with pytest.raises(ValueError, match=f"{name} must be an integer"):
        build(**{name: value})


def test_schedule_filtering_excludes_other_weeks_and_seasons():
    schedule = pd.concat(
        [
            make_schedule(),
            make_schedule().assign(season=2025, week=1, game_id="old"),
        ],
        ignore_index=True,
    )

    result = build(schedule=schedule)

    assert result["game_id"].unique().tolist() == ["2026_01_KC_LAC"]


def test_conflicting_duplicate_schedule_rows_raise():
    schedule = pd.concat(
        [
            make_schedule(),
            make_schedule().iloc[[0]].assign(opponent="BUF"),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="conflicting team rows"):
        build(schedule=schedule)


def test_conflicting_expected_rows_raise():
    expected = pd.concat(
        [
            make_expected(),
            pd.DataFrame([expected_row("KC", "rb_kc_1", "Changed", 3)]),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="conflicting participant rows"):
        build(expected=expected)


def test_duplicate_metric_rows_cannot_multiply_output():
    rbs = pd.concat([make_rb_metrics(), make_rb_metrics().iloc[[0]]])
    defenses = pd.concat([make_defense_metrics(), make_defense_metrics().iloc[[0]]])

    result = build(rbs=rbs, defenses=defenses)

    assert len(result) == 3
    assert not result.duplicated(["game_id", "team", "player_id", "participant_order"]).any()


def test_conflicting_duplicate_metric_rows_raise():
    rbs = pd.concat([make_rb_metrics(), make_rb_metrics().iloc[[0]].copy()])
    rbs.iloc[-1, rbs.columns.get_loc("rb_season_games")] = 99
    with pytest.raises(ValueError, match="Conflicting running-back form metrics"):
        build(rbs=rbs)

    defenses = pd.concat(
        [make_defense_metrics(), make_defense_metrics().iloc[[0]].copy()]
    )
    defenses.iloc[-1, defenses.columns.get_loc("matchup_rank")] = 99
    with pytest.raises(ValueError, match="Conflicting RB defense matchup metrics"):
        build(defenses=defenses)


def test_missing_expected_rows_create_diagnostic_rows():
    expected = make_expected().loc[lambda data: data["team"].ne("LAC")]

    result = build(expected=expected)

    lac = result.loc[result["team"].eq("LAC")].iloc[0]
    assert lac["selection_source"] == "unresolved"
    assert bool(lac["participant_resolution_missing"]) is True
    assert "No expected RB participant rows" in lac["selection_notes"]


@pytest.mark.parametrize(
    ("input_name", "column"),
    [
        ("schedule", "opponent"),
        ("expected", "selection_source"),
        ("rbs", "rb_season_games"),
        ("defenses", "matchup_rank"),
    ],
)
def test_missing_required_columns_raise(input_name, column):
    inputs = {
        "schedule": make_schedule(),
        "expected": make_expected(),
        "rbs": make_rb_metrics(),
        "defenses": make_defense_metrics(),
    }
    inputs[input_name] = inputs[input_name].drop(columns=[column])

    with pytest.raises(ValueError, match="missing required columns"):
        build(**inputs)


def test_input_order_does_not_affect_output_and_inputs_are_immutable():
    schedule = make_schedule()
    expected = make_expected()
    rbs = make_rb_metrics()
    defenses = make_defense_metrics()
    originals = [data.copy(deep=True) for data in [schedule, expected, rbs, defenses]]

    forward = build(schedule, expected, rbs, defenses)
    reverse = build(
        schedule.iloc[::-1],
        expected.iloc[::-1],
        rbs.iloc[::-1],
        defenses.iloc[::-1],
    )

    pd.testing.assert_frame_equal(forward, reverse)
    for current, original in zip([schedule, expected, rbs, defenses], originals):
        pd.testing.assert_frame_equal(current, original)


def test_empty_selected_schedule_returns_complete_output_schema():
    result = build(report_week=3)

    assert result.empty
    assert result.columns.tolist() == OUTPUT_COLUMNS
