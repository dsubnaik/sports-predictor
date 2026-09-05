import pandas as pd

from football.ui.qb_research_view import (
    DEFENSIVE_MATCHUP_RANK_HELP,
    build_matchup_options,
    default_history_season,
    filter_defense_game_log,
    filter_qb_game_log,
    find_matchup,
    prepare_defense_log_display,
    prepare_qb_log_display,
    prepare_summary_display,
)


def make_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2026,
                "report_week": 1,
                "game_id": "2026_01_KC_LAC",
                "team": "KC",
                "opponent": "LAC",
                "expected_player_id": "same_name_kc",
                "expected_player_name": "Alex Smith",
                "selection_source": "depth_chart",
                "depth_chart_date": "2026-09-01",
                "starter_uncertain": False,
                "qb_history_season": 2025,
                "qb_history_cutoff_week": 19,
                "qb_season_avg": 247.26,
                "qb_last3_avg": 255.04,
                "qb_season_attempts_avg": 34.66,
                "qb_season_games": 17,
                "qb_flagged_games": 0,
                "defense_history_season": 2025,
                "defense_history_cutoff_week": 19,
                "defense_season_avg_allowed": 238.88,
                "defense_last3_avg_allowed": 221.15,
                "defense_season_games": 17,
                "defense_flagged_games": 1,
                "matchup_rank": 1,
                "missing_qb_history": False,
                "missing_defense_history": False,
            },
            {
                "season": 2026,
                "report_week": 1,
                "game_id": "2026_01_BUF_MIA",
                "team": "BUF",
                "opponent": "MIA",
                "expected_player_id": "same_name_buf",
                "expected_player_name": "Alex Smith",
                "selection_source": "manual_override",
                "depth_chart_date": "2026-09-02",
                "starter_uncertain": True,
                "qb_history_season": pd.NA,
                "qb_history_cutoff_week": 19,
                "qb_season_avg": pd.NA,
                "qb_last3_avg": pd.NA,
                "qb_season_attempts_avg": pd.NA,
                "qb_season_games": pd.NA,
                "qb_flagged_games": pd.NA,
                "defense_history_season": pd.NA,
                "defense_history_cutoff_week": 19,
                "defense_season_avg_allowed": pd.NA,
                "defense_last3_avg_allowed": pd.NA,
                "defense_season_games": pd.NA,
                "defense_flagged_games": pd.NA,
                "matchup_rank": pd.NA,
                "missing_qb_history": True,
                "missing_defense_history": True,
            },
        ]
    )


def make_qb_logs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            qb_log_row("same_name_kc", "Alex Smith", "KC", "LAC", 2, "2025_02_KC_LAC"),
            qb_log_row("same_name_buf", "Alex Smith", "BUF", "MIA", 1, "2025_01_BUF_MIA"),
            qb_log_row("same_name_kc", "Alex Smith", "KC", "DEN", 1, "2025_01_KC_DEN"),
        ]
    )


def qb_log_row(
    player_id: str,
    player_name: str,
    team: str,
    opponent: str,
    week: int,
    game_id: str,
) -> dict[str, object]:
    return {
        "season": 2025,
        "week": week,
        "game_id": game_id,
        "player_id": player_id,
        "player_name": player_name,
        "team": team,
        "opponent": opponent,
        "passing_attempts": 30 + week,
        "completions": 20 + week,
        "passing_yards": 240 + week,
        "passing_touchdowns": 2,
        "interceptions": 1,
        "low_attempt_primary_qb": False,
        "similar_attempt_split": week == 1,
        "exact_attempt_tie": False,
    }


def make_defense_logs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            defense_log_row("LAC", "KC", "same_name_kc", "Alex Smith", 2, "2025_02_KC_LAC"),
            defense_log_row("MIA", "BUF", "same_name_buf", "Alex Smith", 1, "2025_01_BUF_MIA"),
            defense_log_row("LAC", "DEN", "den_qb", "Denver QB", 1, "2025_01_DEN_LAC"),
        ]
    )


def defense_log_row(
    defense: str,
    offense: str,
    qb_id: str,
    qb_name: str,
    week: int,
    game_id: str,
) -> dict[str, object]:
    return {
        "season": 2025,
        "week": week,
        "game_id": game_id,
        "defense": defense,
        "offense_team": offense,
        "opposing_qb_id": qb_id,
        "opposing_qb_name": qb_name,
        "passing_attempts_allowed": 30 + week,
        "completions_allowed": 20 + week,
        "passing_yards_allowed": 240 + week,
        "passing_touchdowns_allowed": 2,
        "opposing_interceptions": 1,
        "low_attempt_primary_qb": False,
        "similar_attempt_split": week == 1,
        "exact_attempt_tie": False,
    }


def test_summary_display_column_mapping():
    display = prepare_summary_display(make_summary())

    assert display.columns.tolist() == [
        "Defensive Matchup Rank",
        "Expected QB",
        "Team",
        "Opponent",
        "QB Season Avg",
        "QB Last 3 Avg",
        "QB Season Att Avg",
        "Defense Season Avg Allowed",
        "Defense Last 3 Avg Allowed",
        "QB Sample",
        "Defense Sample",
        "Missing QB History",
        "Missing Defense History",
    ]


def test_rounding_without_mutating_backend_data():
    summary = make_summary()
    original = summary.copy(deep=True)

    display = prepare_summary_display(summary)

    assert display.loc[0, "QB Season Avg"] == 247.3
    assert display.loc[0, "QB Last 3 Avg"] == 255.0
    assert display.loc[0, "QB Season Att Avg"] == 34.7
    pd.testing.assert_frame_equal(summary, original)


def test_stable_matchup_option_creation():
    options = build_matchup_options(make_summary())

    assert options[0].option_id == "2026|1|2026_01_KC_LAC|KC"
    assert "KC vs LAC" in options[0].label
    assert "Alex Smith" in options[0].label


def test_correct_selected_qb_log_filtering():
    summary = make_summary()
    matchup = summary.loc[summary["team"] == "KC"].iloc[0]

    result = filter_qb_game_log(make_qb_logs(), matchup)

    assert result["player_id"].tolist() == ["same_name_kc", "same_name_kc"]
    assert result["week"].tolist() == [1, 2]


def test_correct_selected_defense_log_filtering():
    summary = make_summary()
    matchup = summary.loc[summary["team"] == "KC"].iloc[0]

    result = filter_defense_game_log(make_defense_logs(), matchup)

    assert result["defense"].tolist() == ["LAC", "LAC"]
    assert result["week"].tolist() == [1, 2]


def test_duplicate_player_names_use_distinct_matchup_identifiers():
    summary = make_summary()
    options = build_matchup_options(summary)

    assert options[0].label.count("Alex Smith") == 1
    assert options[1].label.count("Alex Smith") == 1
    assert options[0].option_id != options[1].option_id
    assert find_matchup(summary, options[0].option_id)["team"] == "KC"
    assert find_matchup(summary, options[1].option_id)["team"] == "BUF"


def test_missing_histories_are_displayed():
    display = prepare_summary_display(make_summary())

    assert bool(display.loc[1, "Missing QB History"]) is True
    assert bool(display.loc[1, "Missing Defense History"]) is True


def test_empty_detailed_logs_return_empty_display_tables():
    empty_qb_log = make_qb_logs().iloc[0:0]
    empty_defense_log = make_defense_logs().iloc[0:0]

    assert prepare_qb_log_display(empty_qb_log).empty
    assert prepare_defense_log_display(empty_defense_log).empty


def test_input_immutability():
    summary = make_summary()
    qb_logs = make_qb_logs()
    defense_logs = make_defense_logs()
    original_summary = summary.copy(deep=True)
    original_qb_logs = qb_logs.copy(deep=True)
    original_defense_logs = defense_logs.copy(deep=True)
    matchup = summary.iloc[0]

    prepare_summary_display(summary)
    build_matchup_options(summary)
    filter_qb_game_log(qb_logs, matchup)
    filter_defense_game_log(defense_logs, matchup)

    pd.testing.assert_frame_equal(summary, original_summary)
    pd.testing.assert_frame_equal(qb_logs, original_qb_logs)
    pd.testing.assert_frame_equal(defense_logs, original_defense_logs)


def test_default_history_season_matches_pipeline_rule():
    assert default_history_season(2026, 1) == 2025
    assert default_history_season(2026, 2) == 2026


def test_defensive_matchup_rank_help_explains_ranking_scope():
    assert "Rank 1" in DEFENSIVE_MATCHUP_RANK_HELP
    assert "most passing yards per game" in DEFENSIVE_MATCHUP_RANK_HELP
    assert "not a quarterback talent ranking or model prediction" in (
        DEFENSIVE_MATCHUP_RANK_HELP
    )
