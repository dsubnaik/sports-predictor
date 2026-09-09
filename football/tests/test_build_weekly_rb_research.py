import pandas as pd
import pytest

from football.data.normalize_depth_charts import NFLVERSE_DATED_DEPTH_CHART_COLUMNS
from football.features.defense_rb_game_logs import DEFENSE_RB_GAME_LOG_COLUMNS
from football.features.running_back_usage import RUNNING_BACK_USAGE_COLUMNS
from football.pipeline import WeeklyRBResearchResult, build_weekly_rb_research
from football.reports.build_weekly_rb_matchup_report import OUTPUT_COLUMNS as REPORT_COLUMNS


def player_row(season, week, team, opponent, player_id, player_name, carries, yards):
    return {
        "season": season,
        "week": week,
        "season_type": "REG",
        "game_id": f"{season}_{week:02d}_{team}_{opponent}",
        "player_id": player_id,
        "player_display_name": player_name,
        "position": "RB",
        "team": team,
        "opponent_team": opponent,
        "carries": carries,
        "rushing_yards": yards,
        "rushing_tds": 0,
        "receptions": 0,
        "targets": 0,
        "receiving_yards": 0,
        "receiving_tds": 0,
    }


def make_player_stats(season):
    rows = []
    for week in [1, 2, 3]:
        rows.extend(
            [
                player_row(season, week, "KC", "LAC", "rb_kc_1", "Kansas One", 12, 60),
                player_row(season, week, "KC", "LAC", "rb_kc_2", "Kansas Two", 5, 20),
                player_row(season, week, "LAC", "KC", "rb_lac", "Los Angeles", 10, 40),
                player_row(season, week, "BUF", "MIA", "rb_irrelevant", "Irrelevant", 8, 30),
            ]
        )
    return pd.DataFrame(rows)


def make_schedule(season):
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


def depth_row(timestamp, team, player_id, player_name, rank):
    return (timestamp, team, player_name, player_id, "RB", "RB", rank)


def make_depth_charts(_season):
    return pd.DataFrame(
        [
            depth_row("2026-09-14T10:00:00Z", "KC", "rb_kc_1", "Kansas One", 1),
            depth_row("2026-09-14T10:00:00Z", "KC", "rb_kc_2", "Kansas Two", 2),
            depth_row("2026-09-14T10:00:00Z", "LAC", "rb_lac", "Los Angeles", 1),
            depth_row("2026-09-20T10:00:00Z", "KC", "rb_future", "Future", 1),
        ],
        columns=NFLVERSE_DATED_DEPTH_CHART_COLUMNS,
    )


class RecordingLoaders:
    def __init__(self):
        self.player_stats_seasons = []
        self.schedule_seasons = []
        self.depth_chart_seasons = []
        self.player_stats = {2025: make_player_stats(2025), 2026: make_player_stats(2026)}
        self.schedules = {2026: make_schedule(2026)}
        self.depth_charts = {2026: make_depth_charts(2026)}

    def load_player_stats(self, **kwargs):
        self.player_stats_seasons.append(kwargs["seasons"])
        return self.player_stats[kwargs["seasons"]]

    def load_schedules(self, **kwargs):
        self.schedule_seasons.append(kwargs["seasons"])
        return self.schedules[kwargs["seasons"]]

    def load_depth_charts(self, **kwargs):
        self.depth_chart_seasons.append(kwargs["seasons"])
        return self.depth_charts[kwargs["seasons"]]


def build(loaders, **kwargs):
    return build_weekly_rb_research(
        as_of_date="2026-09-15",
        player_stats_loader=loaders.load_player_stats,
        schedule_loader=loaders.load_schedules,
        depth_chart_loader=loaders.load_depth_charts,
        **kwargs,
    )


def test_week_one_uses_previous_season_and_week_two_uses_prior_current_games():
    week_one = RecordingLoaders()
    one = build(week_one, report_season=2026, report_week=1)
    week_two = RecordingLoaders()
    two = build(week_two, report_season=2026, report_week=2)

    assert week_one.player_stats_seasons == [2025]
    assert set(one.summary["historical_season"]) == {2025}
    assert week_two.player_stats_seasons == [2026]
    assert set(two.summary["historical_season"]) == {2026}
    assert two.rb_game_logs["week"].max() == 1
    assert two.defense_game_logs["week"].max() == 1


def test_explicit_earlier_history_uses_all_available_rows_and_empty_history_is_safe():
    loaders = RecordingLoaders()
    result = build(loaders, report_season=2026, report_week=2, history_season=2025)

    assert result.rb_game_logs["week"].max() == 3
    assert result.defense_game_logs["week"].max() == 3

    loaders.player_stats[2025] = make_player_stats(2025).iloc[0:0].copy()
    empty = build(loaders, report_season=2026, report_week=2, history_season=2025)
    assert empty.rb_game_logs.columns.tolist() == RUNNING_BACK_USAGE_COLUMNS
    assert empty.defense_game_logs.columns.tolist() == DEFENSE_RB_GAME_LOG_COLUMNS
    assert empty.summary["historical_season"].tolist() == [2025, 2025, 2025]


def test_multiple_rbs_future_depth_charts_and_relevant_logs_are_handled():
    result = build(RecordingLoaders(), report_season=2026, report_week=2)

    assert result.summary.loc[result.summary["team"].eq("KC"), "player_id"].tolist() == ["rb_kc_1", "rb_kc_2"]
    assert "rb_future" not in set(result.summary["player_id"].dropna())
    assert set(result.rb_game_logs["player_id"]) == {"rb_kc_1", "rb_kc_2", "rb_lac"}
    assert set(result.defense_game_logs["defense"]) == {"KC", "LAC"}
    assert "rb_irrelevant" not in set(result.rb_game_logs["player_id"])


def test_manual_overrides_replace_only_their_team_and_preserve_multiple_players():
    overrides = pd.DataFrame(
        [
            {"season": 2026, "report_week": 2, "team": "KC", "player_id": "manual_1", "player_name": "Manual One", "participant_order": 1},
            {"season": 2026, "report_week": 2, "team": "KC", "player_id": "manual_2", "player_name": "Manual Two", "participant_order": 2},
        ]
    )
    result = build(RecordingLoaders(), report_season=2026, report_week=2, manual_rb_overrides=overrides)

    assert result.summary.loc[result.summary["team"].eq("KC"), "player_id"].tolist() == ["manual_1", "manual_2"]
    assert set(result.summary.loc[result.summary["team"].eq("KC"), "selection_source"]) == {"manual_override"}
    assert result.summary.loc[result.summary["team"].eq("LAC"), "player_id"].tolist() == ["rb_lac"]


def test_empty_selected_schedule_short_circuits_player_and_depth_loaders():
    loaders = RecordingLoaders()
    result = build(loaders, report_season=2026, report_week=9)

    assert result.summary.columns.tolist() == REPORT_COLUMNS
    assert result.rb_game_logs.columns.tolist() == RUNNING_BACK_USAGE_COLUMNS
    assert result.defense_game_logs.columns.tolist() == DEFENSE_RB_GAME_LOG_COLUMNS
    assert loaders.schedule_seasons == [2026]
    assert loaders.player_stats_seasons == []
    assert loaders.depth_chart_seasons == []


@pytest.mark.parametrize(
    ("history_season", "match"),
    [
        (2027, "history_season"),
        (0, "history_season"),
        (-1, "history_season"),
        (True, "history_season"),
        (2025.5, "history_season"),
    ],
)
def test_empty_selected_schedule_validates_explicit_history_season(
    history_season, match
):
    loaders = RecordingLoaders()

    with pytest.raises(ValueError, match=match):
        build(
            loaders,
            report_season=2026,
            report_week=9,
            history_season=history_season,
        )

    assert loaders.schedule_seasons == [2026]
    assert loaders.player_stats_seasons == []
    assert loaders.depth_chart_seasons == []


@pytest.mark.parametrize(
    ("report_week", "history_season"),
    [(9, 2025), (1, None), (9, None)],
)
def test_empty_selected_schedule_valid_history_never_loads_expensive_sources(
    report_week, history_season
):
    loaders = RecordingLoaders()
    if report_week == 1:
        loaders.schedules[2026] = loaders.schedules[2026].loc[
            loaders.schedules[2026]["week"].ne(1)
        ].copy()
    result = build(
        loaders,
        report_season=2026,
        report_week=report_week,
        history_season=history_season,
    )

    assert result.summary.columns.tolist() == REPORT_COLUMNS
    assert loaders.schedule_seasons == [2026]
    assert loaders.player_stats_seasons == []
    assert loaders.depth_chart_seasons == []


def test_unresolved_and_missing_history_are_preserved_in_summary():
    loaders = RecordingLoaders()
    loaders.depth_charts[2026] = pd.DataFrame(
        [depth_row("2026-09-14", "KC", "rb_kc_1", "Kansas One", 1)],
        columns=NFLVERSE_DATED_DEPTH_CHART_COLUMNS,
    )
    result = build(loaders, report_season=2026, report_week=2)

    lac = result.summary.loc[result.summary["team"].eq("LAC")].iloc[0]
    assert bool(lac["participant_resolution_missing"])
    assert bool(lac["rb_history_missing"]) is False


def test_loaders_are_used_once_and_inputs_are_not_mutated():
    loaders = RecordingLoaders()
    stats = loaders.player_stats[2026].copy(deep=True)
    schedule = loaders.schedules[2026].copy(deep=True)
    depth = loaders.depth_charts[2026].copy(deep=True)
    overrides = pd.DataFrame(
        [{"season": 2026, "report_week": 2, "team": "KC", "player_id": "manual", "player_name": "Manual", "participant_order": 1}]
    )
    original_overrides = overrides.copy(deep=True)

    build(loaders, report_season=2026, report_week=2, manual_rb_overrides=overrides)

    assert loaders.player_stats_seasons == [2026]
    assert loaders.schedule_seasons == [2026]
    assert loaders.depth_chart_seasons == [2026]
    pd.testing.assert_frame_equal(loaders.player_stats[2026], stats)
    pd.testing.assert_frame_equal(loaders.schedules[2026], schedule)
    pd.testing.assert_frame_equal(loaders.depth_charts[2026], depth)
    pd.testing.assert_frame_equal(overrides, original_overrides)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"report_season": True, "report_week": 2}, "report_season"),
        ({"report_season": 2026, "report_week": False}, "report_week"),
        ({"report_season": 2026, "report_week": 2, "history_season": 2027}, "history_season"),
    ],
)
def test_invalid_context_values_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        build(RecordingLoaders(), **kwargs)


def test_result_contract_and_determinism():
    first = build(RecordingLoaders(), report_season=2026, report_week=2)
    second = build(RecordingLoaders(), report_season=2026, report_week=2)

    assert isinstance(first, WeeklyRBResearchResult)
    assert set(first.__dict__) == {"summary", "rb_game_logs", "defense_game_logs"}
    pd.testing.assert_frame_equal(first.summary, second.summary)
    pd.testing.assert_frame_equal(first.rb_game_logs, second.rb_game_logs)
    pd.testing.assert_frame_equal(first.defense_game_logs, second.defense_game_logs)
