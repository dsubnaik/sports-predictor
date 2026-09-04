import pandas as pd
import pytest

from football.pipeline import WeeklyQBResearchResult, build_weekly_qb_research


def make_player_stats(season: int) -> pd.DataFrame:
    rows = []
    for week in [1, 2, 3]:
        rows.extend(
            [
                player_row(season, week, "KC", "LAC", "qb_kc", "Kansas QB", 30 + week, 250 + week),
                player_row(season, week, "LAC", "KC", "qb_lac", "Los Angeles QB", 25 + week, 220 + week),
                player_row(season, week, "BUF", "MIA", "qb_buf", "Buffalo QB", 28 + week, 230 + week),
                player_row(season, week, "MIA", "BUF", "qb_mia", "Miami QB", 20 + week, 180 + week),
                player_row(season, week, "LV", "DEN", "qb_irrelevant", "Irrelevant QB", 35, 300),
            ]
        )
    rows.append(
        player_row(season, 4, "KC", "LAC", "qb_kc", "Kansas QB", 40, 404)
    )
    rows.append(
        player_row(season, 1, "KC", "LAC", "rb_kc", "Kansas RB", 1, 12, position="RB")
    )
    rows.append(
        {
            **player_row(season, 5, "KC", "LAC", "qb_post", "Postseason QB", 50, 500),
            "season_type": "POST",
        }
    )
    return pd.DataFrame(rows)


def player_row(
    season: int,
    week: int,
    team: str,
    opponent: str,
    player_id: str,
    player_name: str,
    attempts: int,
    yards: int,
    position: str = "QB",
) -> dict[str, object]:
    return {
        "season": season,
        "week": week,
        "season_type": "REG",
        "game_id": f"{season}_{week:02d}_{team}_{opponent}",
        "player_id": player_id,
        "player_display_name": player_name,
        "position": position,
        "team": team,
        "opponent_team": opponent,
        "attempts": attempts,
        "completions": attempts - 5,
        "passing_yards": yards,
        "passing_tds": 2,
        "passing_interceptions": 1,
    }


def make_schedule_source(season: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": season,
                "week": 1,
                "game_id": f"{season}_01_KC_LAC",
                "game_type": "REG",
                "gameday": f"{season}-09-10",
                "gametime": "20:20",
                "home_team": "KC",
                "away_team": "LAC",
            },
            {
                "season": season,
                "week": 1,
                "game_id": f"{season}_01_BUF_MIA",
                "game_type": "REG",
                "gameday": f"{season}-09-11",
                "gametime": "13:00",
                "home_team": "BUF",
                "away_team": "MIA",
            },
            {
                "season": season,
                "week": 2,
                "game_id": f"{season}_02_KC_LAC",
                "game_type": "REG",
                "gameday": f"{season}-09-17",
                "gametime": "20:20",
                "home_team": "KC",
                "away_team": "LAC",
            },
        ]
    )


def make_depth_charts(season: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            depth_row(season, 1, "KC", "qb_kc", "Kansas QB"),
            depth_row(season, 1, "LAC", "qb_lac", "Los Angeles QB"),
            depth_row(season, 1, "BUF", "qb_buf", "Buffalo QB"),
            depth_row(season, 2, "KC", "qb_kc", "Kansas QB"),
            depth_row(season, 2, "LAC", "qb_lac", "Los Angeles QB"),
        ]
    )


def depth_row(
    season: int,
    week: int,
    team: str,
    player_id: str,
    player_name: str,
) -> dict[str, object]:
    return {
        "season": season,
        "week": week,
        "game_type": "REG",
        "club_code": team,
        "gsis_id": player_id,
        "full_name": player_name,
        "position": "QB",
        "depth_team": 1,
    }


class RecordingLoaders:
    def __init__(self) -> None:
        self.player_stats_seasons: list[int] = []
        self.schedule_seasons: list[int] = []
        self.depth_chart_seasons: list[int] = []
        self.player_stats = {2025: make_player_stats(2025), 2026: make_player_stats(2026)}
        self.schedules = {2026: make_schedule_source(2026)}
        self.depth_charts = {2026: make_depth_charts(2026)}

    def load_player_stats(self, **kwargs) -> pd.DataFrame:
        self.player_stats_seasons.append(kwargs["seasons"])
        return self.player_stats[kwargs["seasons"]]

    def load_schedules(self, **kwargs) -> pd.DataFrame:
        self.schedule_seasons.append(kwargs["seasons"])
        return self.schedules[kwargs["seasons"]]

    def load_depth_charts(self, **kwargs) -> pd.DataFrame:
        self.depth_chart_seasons.append(kwargs["seasons"])
        return self.depth_charts[kwargs["seasons"]]


def build_with(loaders: RecordingLoaders, **kwargs) -> WeeklyQBResearchResult:
    return build_weekly_qb_research(
        as_of_date="2026-09-09",
        player_stats_loader=loaders.load_player_stats,
        schedule_loader=loaders.load_schedules,
        depth_chart_loader=loaders.load_depth_charts,
        **kwargs,
    )


def test_week_1_defaults_to_previous_season_and_requests_expected_seasons():
    loaders = RecordingLoaders()

    result = build_with(loaders, report_season=2026, report_week=1)

    assert loaders.player_stats_seasons == [2025]
    assert loaders.schedule_seasons == [2026]
    assert loaders.depth_chart_seasons == [2026]
    assert set(result.summary["qb_history_season"]) == {2025}
    assert set(result.summary["qb_history_cutoff_week"]) == {5}


def test_week_2_defaults_to_current_season_and_applies_same_season_cutoff():
    loaders = RecordingLoaders()

    result = build_with(loaders, report_season=2026, report_week=2)

    assert loaders.player_stats_seasons == [2026]
    assert set(result.summary["qb_history_season"].dropna()) == {2026}
    assert set(result.summary["qb_history_cutoff_week"].dropna()) == {2}
    assert result.qb_game_logs["week"].max() == 1
    assert result.defense_game_logs["week"].max() == 1


def test_explicit_history_season_override_uses_complete_previous_history():
    loaders = RecordingLoaders()

    result = build_with(
        loaders,
        report_season=2026,
        report_week=2,
        history_season=2025,
    )

    assert loaders.player_stats_seasons == [2025]
    assert set(result.summary["qb_history_cutoff_week"]) == {5}
    assert 4 in result.qb_game_logs["week"].tolist()


def test_future_history_season_is_rejected_before_loading():
    loaders = RecordingLoaders()

    with pytest.raises(ValueError, match="history_season"):
        build_with(
            loaders,
            report_season=2026,
            report_week=1,
            history_season=2027,
        )

    assert loaders.player_stats_seasons == []


def test_pipeline_returns_all_three_result_tables_with_relevant_logs_only():
    loaders = RecordingLoaders()

    result = build_with(loaders, report_season=2026, report_week=1)

    assert isinstance(result.summary, pd.DataFrame)
    assert isinstance(result.qb_game_logs, pd.DataFrame)
    assert isinstance(result.defense_game_logs, pd.DataFrame)
    assert set(result.qb_game_logs["player_id"]) == {"qb_buf", "qb_kc", "qb_lac"}
    assert "qb_irrelevant" not in set(result.qb_game_logs["player_id"])
    assert set(result.defense_game_logs["defense"]) == {"BUF", "KC", "LAC", "MIA"}
    assert "DEN" not in set(result.defense_game_logs["defense"])


def test_unresolved_qb_is_retained_in_summary_without_qb_logs():
    loaders = RecordingLoaders()

    result = build_with(loaders, report_season=2026, report_week=1)
    mia = result.summary.loc[result.summary["team"] == "MIA"].iloc[0]

    assert mia["selection_source"] == "unresolved"
    assert bool(mia["missing_qb_history"]) is True
    assert "qb_mia" not in set(result.qb_game_logs["player_id"])


def test_manual_override_is_forwarded_to_expected_qb_resolution():
    loaders = RecordingLoaders()
    overrides = pd.DataFrame(
        [
            {
                "season": 2026,
                "report_week": 1,
                "team": "MIA",
                "player_id": "qb_mia",
                "player_name": "Miami QB",
                "selection_notes": "Manual starter.",
            }
        ]
    )

    result = build_with(
        loaders,
        report_season=2026,
        report_week=1,
        manual_qb_overrides=overrides,
    )
    mia = result.summary.loc[result.summary["team"] == "MIA"].iloc[0]

    assert mia["selection_source"] == "manual_override"
    assert mia["expected_player_id"] == "qb_mia"
    assert "qb_mia" in set(result.qb_game_logs["player_id"])


def test_outputs_are_deterministic():
    first_loaders = RecordingLoaders()
    second_loaders = RecordingLoaders()

    first = build_with(first_loaders, report_season=2026, report_week=1)
    second = build_with(second_loaders, report_season=2026, report_week=1)

    pd.testing.assert_frame_equal(first.summary, second.summary)
    pd.testing.assert_frame_equal(first.qb_game_logs, second.qb_game_logs)
    pd.testing.assert_frame_equal(first.defense_game_logs, second.defense_game_logs)


def test_loader_outputs_and_manual_overrides_are_not_mutated():
    loaders = RecordingLoaders()
    original_stats = loaders.player_stats[2025].copy(deep=True)
    original_schedule = loaders.schedules[2026].copy(deep=True)
    original_depth = loaders.depth_charts[2026].copy(deep=True)
    overrides = pd.DataFrame(
        [
            {
                "season": 2026,
                "report_week": 1,
                "team": "KC",
                "player_id": "qb_kc",
                "player_name": "Kansas QB",
                "selection_notes": "No change.",
            }
        ]
    )
    original_overrides = overrides.copy(deep=True)

    build_with(
        loaders,
        report_season=2026,
        report_week=1,
        manual_qb_overrides=overrides,
    )

    pd.testing.assert_frame_equal(loaders.player_stats[2025], original_stats)
    pd.testing.assert_frame_equal(loaders.schedules[2026], original_schedule)
    pd.testing.assert_frame_equal(loaders.depth_charts[2026], original_depth)
    pd.testing.assert_frame_equal(overrides, original_overrides)


@pytest.mark.parametrize(
    ("report_season", "report_week", "match"),
    [
        ("2026", 1, "report_season"),
        (2026, 0, "report_week"),
        (2026, False, "report_week"),
    ],
)
def test_report_values_are_validated(report_season, report_week, match):
    loaders = RecordingLoaders()

    with pytest.raises(ValueError, match=match):
        build_with(loaders, report_season=report_season, report_week=report_week)
