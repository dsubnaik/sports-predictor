from copy import deepcopy

import pandas as pd
import pytest

from football.data.normalize_depth_charts import NFLVERSE_DATED_DEPTH_CHART_COLUMNS
from football.features.defense_rb_game_logs import DEFENSE_RB_GAME_LOG_COLUMNS
from football.features.running_back_usage import RUNNING_BACK_USAGE_COLUMNS
from football.pipeline import (
    WeeklyPlayerPropOddsResult,
    WeeklyRBResearchResult,
    build_weekly_rb_research,
)
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


def test_pipeline_keeps_missing_id_participants_out_of_rb_logs():
    loaders = RecordingLoaders()
    loaders.depth_charts[2026] = pd.DataFrame(
        [
            depth_row("2026-09-14", "KC", "rb_kc_1", "Kansas One", 1),
            depth_row("2026-09-14", "KC", None, "Unknown Kansas", 2),
            depth_row("2026-09-14", "LAC", "rb_lac", "Los Angeles", 1),
        ],
        columns=NFLVERSE_DATED_DEPTH_CHART_COLUMNS,
    )

    result = build(loaders, report_season=2026, report_week=2)

    unknown = result.summary.loc[
        result.summary["player_name"].eq("Unknown Kansas")
    ].iloc[0]
    assert pd.isna(unknown["player_id"])
    assert bool(unknown["participant_resolution_missing"])
    assert bool(unknown["rb_history_missing"])
    assert "Unknown Kansas" not in set(result.rb_game_logs["player_name"])
    assert result.rb_game_logs["player_id"].notna().all()


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
    assert set(first.__dict__) == {"summary", "rb_game_logs", "defense_game_logs", "player_prop_odds"}
    assert first.player_prop_odds is None
    pd.testing.assert_frame_equal(first.summary, second.summary)
    pd.testing.assert_frame_equal(first.rb_game_logs, second.rb_game_logs)
    pd.testing.assert_frame_equal(first.defense_game_logs, second.defense_game_logs)


def _rush_event_payload():
    return [{
        "id": "odds-kc-lac",
        "commence_time": "2026-09-18T00:20:00Z",
        "home_team": "Kansas City Chiefs",
        "away_team": "Los Angeles Chargers",
    }]


def _rush_prop_payload(player_names=("Kansas One", "Kansas Two"), bookmakers=True):
    markets = []
    for player_name in player_names:
        markets.append({
            "key": "player_rush_yds",
            "last_update": "2026-09-17T12:01:00Z",
            "outcomes": [
                {"name": "Over", "description": player_name, "price": -110, "point": 55.5},
                {"name": "Under", "description": player_name, "price": -115, "point": 55.5},
            ],
        })
    return {
        "id": "odds-kc-lac",
        "commence_time": "2026-09-18T00:20:00Z",
        "home_team": "Kansas City Chiefs",
        "away_team": "Los Angeles Chargers",
        "bookmakers": ([{
            "key": "draftkings", "title": "DraftKings",
            "last_update": "2026-09-17T12:00:00Z", "markets": markets,
        }] if bookmakers else []),
    }


def test_rushing_props_are_disabled_by_default_without_odds_calls():
    loaders = RecordingLoaders()
    calls = []
    result = build(
        loaders,
        report_season=2026,
        report_week=2,
        odds_event_fetcher=lambda **_kwargs: calls.append("events"),
        odds_event_props_fetcher=lambda *_args, **_kwargs: calls.append("props"),
    )
    baseline = build(RecordingLoaders(), report_season=2026, report_week=2)

    assert result.player_prop_odds is None
    assert calls == []
    pd.testing.assert_frame_equal(result.summary, baseline.summary)
    pd.testing.assert_frame_equal(result.rb_game_logs, baseline.rb_game_logs)
    pd.testing.assert_frame_equal(result.defense_game_logs, baseline.defense_game_logs)


def test_enabled_rushing_props_include_starter_and_backup_without_summary_multiplication():
    loaders = RecordingLoaders()
    event_calls, prop_calls = [], []
    event_payload = _rush_event_payload()
    prop_payload = _rush_prop_payload()
    event_before, prop_before = deepcopy(event_payload), deepcopy(prop_payload)

    def fetch_events(*, api_key):
        event_calls.append(api_key)
        return event_payload

    def fetch_props(event_id, market_keys, *, api_key):
        prop_calls.append((event_id, market_keys, api_key))
        return prop_payload

    result = build(
        loaders,
        report_season=2026,
        report_week=2,
        include_player_props=True,
        odds_api_key="test-key",
        odds_event_fetcher=fetch_events,
        odds_event_props_fetcher=fetch_props,
    )

    assert isinstance(result.player_prop_odds, WeeklyPlayerPropOddsResult)
    assert loaders.schedule_seasons == [2026]
    assert loaders.depth_chart_seasons == [2026]
    assert loaders.player_stats_seasons == [2026]
    assert event_calls == ["test-key"]
    assert prop_calls == [("odds-kc-lac", ("player_rush_yds",), "test-key")]
    odds = result.player_prop_odds.player_matched_odds
    assert odds["market_key"].tolist() == ["player_rush_yds"] * 4
    assert odds["player_id"].tolist() == ["rb_kc_1", "rb_kc_1", "rb_kc_2", "rb_kc_2"]
    assert odds["event_match_status"].tolist() == ["matched"] * 4
    assert odds["price"].tolist() == [-110, -115, -110, -115]
    assert odds["bookmaker_last_update"].notna().all()
    assert odds["market_last_update"].notna().all()
    assert len(result.summary) == 3
    assert event_payload == event_before
    assert prop_payload == prop_before


def test_enabled_rushing_props_keep_unresolved_rows_safe_and_no_bookmakers_empty():
    loaders = RecordingLoaders()
    loaders.depth_charts[2026] = pd.DataFrame(
        [
            depth_row("2026-09-14", "KC", "rb_kc_1", "Kansas One", 1),
            depth_row("2026-09-14", "KC", None, "Unknown Kansas", 2),
            depth_row("2026-09-14", "LAC", "rb_lac", "Los Angeles", 1),
        ],
        columns=NFLVERSE_DATED_DEPTH_CHART_COLUMNS,
    )
    unresolved = build(
        loaders,
        report_season=2026,
        report_week=2,
        include_player_props=True,
        odds_event_fetcher=lambda **_kwargs: _rush_event_payload(),
        odds_event_props_fetcher=lambda *_args, **_kwargs: _rush_prop_payload(("Unknown Kansas",)),
    )
    odds = unresolved.player_prop_odds.player_matched_odds
    assert odds["match_status"].tolist() == ["unmatched", "unmatched"]
    assert odds["player_id"].isna().all()
    empty = build(
        RecordingLoaders(),
        report_season=2026,
        report_week=2,
        include_player_props=True,
        odds_event_fetcher=lambda **_kwargs: _rush_event_payload(),
        odds_event_props_fetcher=lambda *_args, **_kwargs: _rush_prop_payload(bookmakers=False),
    )
    assert empty.player_prop_odds.normalized_odds.empty
    assert empty.player_prop_odds.player_matched_odds.empty
    assert len(empty.summary) == 3


def test_empty_schedule_never_fetches_odds_and_non_boolean_opt_in_fails_before_loaders():
    loaders = RecordingLoaders()
    calls = []
    result = build(
        loaders,
        report_season=2026,
        report_week=9,
        include_player_props=True,
        odds_event_fetcher=lambda **_kwargs: calls.append("events"),
        odds_event_props_fetcher=lambda *_args, **_kwargs: calls.append("props"),
    )
    assert result.player_prop_odds is None
    assert calls == []
    assert loaders.player_stats_seasons == []
    assert loaders.depth_chart_seasons == []

    invalid = RecordingLoaders()
    with pytest.raises(TypeError, match="include_player_props"):
        build(invalid, report_season=2026, report_week=2, include_player_props=1)
    assert invalid.schedule_seasons == []
    assert invalid.player_stats_seasons == []
    assert invalid.depth_chart_seasons == []


def test_enabled_odds_failure_propagates_without_partial_result():
    with pytest.raises(RuntimeError, match="event failure"):
        build(
            RecordingLoaders(),
            report_season=2026,
            report_week=2,
            include_player_props=True,
            odds_event_fetcher=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("event failure")),
        )
