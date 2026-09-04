import pandas as pd
import pytest

from football.reports.build_weekly_qb_matchup_report import (
    OUTPUT_COLUMNS,
    build_weekly_qb_matchup_report,
)


def make_expected_qbs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2026,
                "report_week": 1,
                "game_id": "2026_01_KC_LAC",
                "game_date": "2026-09-10",
                "game_time": "20:20",
                "team": "KC",
                "opponent": "LAC",
                "home_away": "away",
                "expected_player_id": "qb_mahomes",
                "expected_player_name": "Patrick Mahomes",
                "selection_source": "depth_chart",
                "depth_rank": 1,
                "depth_chart_date": "2026-09-01",
                "starter_uncertain": False,
                "selection_notes": "",
            },
            {
                "season": 2026,
                "report_week": 1,
                "game_id": "2026_01_KC_LAC",
                "game_date": "2026-09-10",
                "game_time": "20:20",
                "team": "LAC",
                "opponent": "KC",
                "home_away": "home",
                "expected_player_id": "qb_herbert",
                "expected_player_name": "Justin Herbert",
                "selection_source": "depth_chart",
                "depth_rank": 1,
                "depth_chart_date": "2026-09-01",
                "starter_uncertain": False,
                "selection_notes": "",
            },
            {
                "season": 2026,
                "report_week": 1,
                "game_id": "2026_01_BUF_MIA",
                "game_date": "2026-09-11",
                "game_time": "13:00",
                "team": "BUF",
                "opponent": "MIA",
                "home_away": "away",
                "expected_player_id": "qb_allen",
                "expected_player_name": "Josh Allen",
                "selection_source": "depth_chart",
                "depth_rank": 1,
                "depth_chart_date": "2026-09-01",
                "starter_uncertain": False,
                "selection_notes": "",
            },
        ]
    )


def make_qb_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2025,
                "report_week": 19,
                "player_id": "qb_mahomes",
                "player_name": "Old overlapping name",
                "latest_team": "KC",
                "qb_season_avg": 270.0,
                "qb_last3_avg": 285.0,
                "qb_season_attempts_avg": 36.0,
                "qb_last3_attempts_avg": 38.0,
                "qb_season_games": 17,
                "qb_last3_games": 3,
                "flagged_games": 1,
            },
            {
                "season": 2025,
                "report_week": 19,
                "player_id": "qb_herbert",
                "player_name": "Old overlapping name",
                "latest_team": "LAC",
                "qb_season_avg": 255.0,
                "qb_last3_avg": 245.0,
                "qb_season_attempts_avg": 34.0,
                "qb_last3_attempts_avg": 32.0,
                "qb_season_games": 16,
                "qb_last3_games": 3,
                "flagged_games": 0,
            },
        ]
    )


def make_defense_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2025,
                "report_week": 19,
                "defense": "LAC",
                "team": "Overlapping team ignored",
                "defense_season_avg_allowed": 240.0,
                "defense_last3_avg_allowed": 230.0,
                "defense_season_games": 17,
                "defense_last3_games": 3,
                "flagged_games": 2,
                "matchup_rank": 2,
            },
            {
                "season": 2025,
                "report_week": 19,
                "defense": "KC",
                "team": "Overlapping team ignored",
                "defense_season_avg_allowed": 220.0,
                "defense_last3_avg_allowed": 210.0,
                "defense_season_games": 17,
                "defense_last3_games": 3,
                "flagged_games": 0,
                "matchup_rank": 8,
            },
        ]
    )


def test_build_weekly_qb_matchup_report_joins_qb_metrics_by_expected_player_id():
    result = build_weekly_qb_matchup_report(
        make_expected_qbs(), make_qb_metrics(), make_defense_metrics()
    )

    kc = result.loc[result["team"] == "KC"].iloc[0]
    assert kc["qb_season_avg"] == 270.0
    assert kc["qb_last3_attempts_avg"] == 38.0
    assert kc["qb_flagged_games"] == 1


def test_build_weekly_qb_matchup_report_joins_defense_metrics_by_opponent():
    result = build_weekly_qb_matchup_report(
        make_expected_qbs(), make_qb_metrics(), make_defense_metrics()
    )

    kc = result.loc[result["team"] == "KC"].iloc[0]
    assert kc["opponent"] == "LAC"
    assert kc["defense_season_avg_allowed"] == 240.0
    assert kc["defense_flagged_games"] == 2


def test_build_weekly_qb_matchup_report_supports_2026_schedule_with_2025_history():
    result = build_weekly_qb_matchup_report(
        make_expected_qbs(), make_qb_metrics(), make_defense_metrics()
    )

    row = result.loc[result["team"] == "KC"].iloc[0]
    assert row["season"] == 2026
    assert row["report_week"] == 1
    assert row["qb_history_season"] == 2025
    assert row["defense_history_season"] == 2025


def test_build_weekly_qb_matchup_report_preserves_history_cutoff_context():
    result = build_weekly_qb_matchup_report(
        make_expected_qbs(), make_qb_metrics(), make_defense_metrics()
    )

    row = result.loc[result["team"] == "LAC"].iloc[0]
    assert row["qb_history_cutoff_week"] == 19
    assert row["defense_history_cutoff_week"] == 19


def test_build_weekly_qb_matchup_report_retains_all_scheduled_teams():
    result = build_weekly_qb_matchup_report(
        make_expected_qbs(), make_qb_metrics(), make_defense_metrics()
    )

    assert result["team"].tolist() == ["KC", "LAC", "BUF"]


def test_build_weekly_qb_matchup_report_keeps_unresolved_expected_qb():
    expected = make_expected_qbs()
    expected.loc[expected["team"] == "BUF", "expected_player_id"] = pd.NA
    expected.loc[expected["team"] == "BUF", "expected_player_name"] = pd.NA
    result = build_weekly_qb_matchup_report(
        expected, make_qb_metrics(), make_defense_metrics()
    )

    buf = result.loc[result["team"] == "BUF"].iloc[0]
    assert pd.isna(buf["expected_player_id"])
    assert bool(buf["missing_qb_history"]) is True


def test_build_weekly_qb_matchup_report_flags_missing_qb_history():
    result = build_weekly_qb_matchup_report(
        make_expected_qbs(), make_qb_metrics(), make_defense_metrics()
    )

    buf = result.loc[result["team"] == "BUF"].iloc[0]
    assert bool(buf["missing_qb_history"]) is True
    assert pd.isna(buf["qb_season_avg"])


def test_build_weekly_qb_matchup_report_flags_missing_defense_history():
    result = build_weekly_qb_matchup_report(
        make_expected_qbs(), make_qb_metrics(), make_defense_metrics()
    )

    buf = result.loc[result["team"] == "BUF"].iloc[0]
    assert bool(buf["missing_defense_history"]) is True
    assert pd.isna(buf["defense_season_avg_allowed"])


def test_build_weekly_qb_matchup_report_allows_identical_metric_duplicates():
    qb_metrics = pd.concat([make_qb_metrics(), make_qb_metrics().iloc[[0]]])
    defense_metrics = pd.concat(
        [make_defense_metrics(), make_defense_metrics().iloc[[0]]]
    )

    result = build_weekly_qb_matchup_report(
        make_expected_qbs(), qb_metrics, defense_metrics
    )

    assert len(result) == 3
    assert result.loc[result["team"] == "KC", "qb_season_avg"].iloc[0] == 270.0


def test_build_weekly_qb_matchup_report_rejects_conflicting_qb_metric_duplicates():
    qb_metrics = pd.concat([make_qb_metrics(), make_qb_metrics().iloc[[0]].copy()])
    qb_metrics.iloc[-1, qb_metrics.columns.get_loc("qb_season_avg")] = 999.0

    with pytest.raises(ValueError, match="Conflicting quarterback form metrics"):
        build_weekly_qb_matchup_report(
            make_expected_qbs(), qb_metrics, make_defense_metrics()
        )


def test_build_weekly_qb_matchup_report_rejects_conflicting_defense_duplicates():
    defense_metrics = pd.concat(
        [make_defense_metrics(), make_defense_metrics().iloc[[0]].copy()]
    )
    defense_metrics.iloc[
        -1, defense_metrics.columns.get_loc("defense_season_avg_allowed")
    ] = 999.0

    with pytest.raises(ValueError, match="Conflicting defense matchup metrics"):
        build_weekly_qb_matchup_report(
            make_expected_qbs(), make_qb_metrics(), defense_metrics
        )


def test_build_weekly_qb_matchup_report_rejects_duplicate_expected_team_game():
    expected = pd.concat([make_expected_qbs(), make_expected_qbs().iloc[[0]]])

    with pytest.raises(ValueError, match="duplicate scheduled team-game"):
        build_weekly_qb_matchup_report(
            expected, make_qb_metrics(), make_defense_metrics()
        )


def test_build_weekly_qb_matchup_report_ignores_overlapping_metric_columns():
    result = build_weekly_qb_matchup_report(
        make_expected_qbs(), make_qb_metrics(), make_defense_metrics()
    )

    assert "player_name" not in result.columns
    assert "latest_team" not in result.columns
    assert result.loc[result["team"] == "KC", "expected_player_name"].iloc[
        0
    ] == "Patrick Mahomes"


def test_build_weekly_qb_matchup_report_sorts_by_rank_missing_last_then_game_team():
    result = build_weekly_qb_matchup_report(
        make_expected_qbs(), make_qb_metrics(), make_defense_metrics()
    )

    assert result["team"].tolist() == ["KC", "LAC", "BUF"]
    assert result["matchup_rank"].iloc[0:2].tolist() == [2.0, 8.0]
    assert pd.isna(result["matchup_rank"].iloc[2])


def test_build_weekly_qb_matchup_report_empty_expected_input_returns_schema():
    result = build_weekly_qb_matchup_report(
        make_expected_qbs().iloc[0:0], make_qb_metrics(), make_defense_metrics()
    )

    assert result.empty
    assert result.columns.tolist() == OUTPUT_COLUMNS


def test_build_weekly_qb_matchup_report_does_not_mutate_inputs():
    expected = make_expected_qbs()
    qbs = make_qb_metrics()
    defenses = make_defense_metrics()
    expected_before = expected.copy(deep=True)
    qbs_before = qbs.copy(deep=True)
    defenses_before = defenses.copy(deep=True)

    build_weekly_qb_matchup_report(expected, qbs, defenses)

    pd.testing.assert_frame_equal(expected, expected_before)
    pd.testing.assert_frame_equal(qbs, qbs_before)
    pd.testing.assert_frame_equal(defenses, defenses_before)


def test_build_weekly_qb_matchup_report_validates_required_columns():
    expected = make_expected_qbs().drop(columns=["opponent"])

    with pytest.raises(ValueError, match="Expected quarterback data"):
        build_weekly_qb_matchup_report(
            expected, make_qb_metrics(), make_defense_metrics()
        )
