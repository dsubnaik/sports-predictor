import numpy as np
import pandas as pd
import pytest

from football.features.defense_rb_game_logs import (
    DEFENSE_RB_GAME_LOG_COLUMNS,
)
from football.features.rb_defense_matchup_metrics import (
    OUTPUT_COLUMNS,
    build_rb_defense_matchup_metrics,
)


def make_log(
    *,
    season: int = 2026,
    week: int = 1,
    game_id: str = "2026_01_KC_LAC",
    defense: str = "LAC",
    offense_team: str = "KC",
    rushing_attempts: float = 10,
    rushing_yards: float = 100,
    rushing_touchdowns: float = 0,
    receptions: float = 2,
    targets: float = 3,
    receiving_yards: float = 20,
    receiving_touchdowns: float = 0,
    opportunities: float | None = None,
    players_used: float = 1,
) -> dict[str, object]:
    return {
        "season": season,
        "week": week,
        "game_id": game_id,
        "defense": defense,
        "offense_team": offense_team,
        "rb_rushing_attempts_allowed": rushing_attempts,
        "rb_rushing_yards_allowed": rushing_yards,
        "rb_rushing_touchdowns_allowed": rushing_touchdowns,
        "rb_receptions_allowed": receptions,
        "rb_targets_allowed": targets,
        "rb_receiving_yards_allowed": receiving_yards,
        "rb_receiving_touchdowns_allowed": receiving_touchdowns,
        "rb_opportunities_allowed": (
            rushing_attempts + targets
            if opportunities is None
            else opportunities
        ),
        "rb_players_used": players_used,
    }


def make_logs(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=DEFENSE_RB_GAME_LOG_COLUMNS)


def make_metric_logs() -> pd.DataFrame:
    return make_logs(
        make_log(),
        make_log(
            week=2,
            game_id="2026_02_DEN_LAC",
            offense_team="DEN",
            rushing_attempts=20,
            rushing_yards=200,
            rushing_touchdowns=1,
            receptions=4,
            targets=5,
            receiving_yards=40,
            receiving_touchdowns=1,
            opportunities=25,
            players_used=2,
        ),
        make_log(
            week=3,
            game_id="2026_03_BUF_LAC",
            offense_team="BUF",
            rushing_attempts=30,
            rushing_yards=300,
            rushing_touchdowns=2,
            receptions=6,
            targets=7,
            receiving_yards=60,
            receiving_touchdowns=0,
            opportunities=37,
            players_used=3,
        ),
        make_log(
            week=4,
            game_id="2026_04_MIA_LAC",
            offense_team="MIA",
            rushing_attempts=40,
            rushing_yards=400,
            rushing_touchdowns=3,
            receptions=8,
            targets=9,
            receiving_yards=80,
            receiving_touchdowns=1,
            opportunities=49,
            players_used=4,
        ),
        make_log(
            week=5,
            game_id="2026_05_NYJ_LAC",
            offense_team="NYJ",
            rushing_attempts=99,
            rushing_yards=999,
            rushing_touchdowns=9,
            receptions=9,
            targets=10,
            receiving_yards=999,
            receiving_touchdowns=9,
            opportunities=109,
            players_used=9,
        ),
        make_log(
            week=6,
            game_id="2026_06_NE_LAC",
            offense_team="NE",
            rushing_attempts=100,
            rushing_yards=1000,
            rushing_touchdowns=10,
            receptions=10,
            targets=11,
            receiving_yards=1000,
            receiving_touchdowns=10,
            opportunities=111,
            players_used=10,
        ),
        make_log(
            season=2025,
            week=18,
            game_id="2025_18_KC_LAC",
            offense_team="KC",
            rushing_attempts=50,
            rushing_yards=500,
            rushing_touchdowns=5,
            receptions=10,
            targets=12,
            receiving_yards=100,
            receiving_touchdowns=2,
            opportunities=62,
            players_used=5,
        ),
    )


def get_defense(result: pd.DataFrame, defense: str) -> pd.Series:
    return result.loc[result["defense"] == defense].iloc[0]


def test_build_rb_defense_metrics_calculates_all_season_and_last3_averages():
    result = build_rb_defense_matchup_metrics(make_metric_logs(), 2026, 5)
    defense = get_defense(result, "LAC")

    assert defense["defense_season_rb_rushing_attempts_avg_allowed"] == 25
    assert defense["defense_last3_rb_rushing_attempts_avg_allowed"] == 30
    assert defense["defense_season_rb_rushing_yards_avg_allowed"] == 250
    assert defense["defense_last3_rb_rushing_yards_avg_allowed"] == 300
    assert defense["defense_season_rb_rushing_touchdowns_avg_allowed"] == 1.5
    assert defense["defense_last3_rb_rushing_touchdowns_avg_allowed"] == 2
    assert defense["defense_season_rb_receptions_avg_allowed"] == 5
    assert defense["defense_last3_rb_receptions_avg_allowed"] == 6
    assert defense["defense_season_rb_targets_avg_allowed"] == 6
    assert defense["defense_last3_rb_targets_avg_allowed"] == 7
    assert defense["defense_season_rb_receiving_yards_avg_allowed"] == 50
    assert defense["defense_last3_rb_receiving_yards_avg_allowed"] == 60
    assert (
        defense["defense_season_rb_receiving_touchdowns_avg_allowed"]
        == 0.5
    )
    assert defense["defense_last3_rb_receiving_touchdowns_avg_allowed"] == 2 / 3
    assert defense["defense_season_rb_opportunities_avg_allowed"] == 31
    assert defense["defense_last3_rb_opportunities_avg_allowed"] == 37
    assert defense["defense_season_rb_players_used_avg"] == 2.5
    assert defense["defense_last3_rb_players_used_avg"] == 3


def test_yards_per_carry_uses_weighted_totals_for_season_and_last3():
    data = make_logs(
        make_log(rushing_attempts=1, rushing_yards=10),
        make_log(
            week=2,
            game_id="2026_02_DEN_LAC",
            offense_team="DEN",
            rushing_attempts=9,
            rushing_yards=9,
        ),
        make_log(
            week=3,
            game_id="2026_03_BUF_LAC",
            offense_team="BUF",
            rushing_attempts=10,
            rushing_yards=50,
        ),
        make_log(
            week=4,
            game_id="2026_04_MIA_LAC",
            offense_team="MIA",
            rushing_attempts=30,
            rushing_yards=60,
        ),
    )

    defense = get_defense(
        build_rb_defense_matchup_metrics(data, 2026, 5),
        "LAC",
    )

    assert defense["defense_season_rb_yards_per_carry_allowed"] == pytest.approx(
        129 / 50
    )
    assert defense["defense_last3_rb_yards_per_carry_allowed"] == pytest.approx(
        119 / 49
    )


def test_zero_rushing_attempts_produce_finite_zero_yards_per_carry():
    data = make_logs(
        make_log(rushing_attempts=0, rushing_yards=-5),
        make_log(
            week=2,
            game_id="2026_02_DEN_LAC",
            offense_team="DEN",
            rushing_attempts=0,
            rushing_yards=0,
        ),
    )

    defense = get_defense(
        build_rb_defense_matchup_metrics(data, 2026, 3),
        "LAC",
    )

    assert defense["defense_season_rb_yards_per_carry_allowed"] == 0.0
    assert defense["defense_last3_rb_yards_per_carry_allowed"] == 0.0
    assert np.isfinite(defense["defense_season_rb_yards_per_carry_allowed"])
    assert np.isfinite(defense["defense_last3_rb_yards_per_carry_allowed"])


def test_negative_rushing_and_receiving_yardage_remain_valid():
    data = make_logs(
        make_log(
            rushing_attempts=5,
            rushing_yards=-10,
            receiving_yards=-6,
        )
    )

    defense = get_defense(
        build_rb_defense_matchup_metrics(data, 2026, 2),
        "LAC",
    )

    assert defense["defense_season_rb_rushing_yards_avg_allowed"] == -10
    assert defense["defense_last3_rb_rushing_yards_avg_allowed"] == -10
    assert defense["defense_season_rb_receiving_yards_avg_allowed"] == -6
    assert defense["defense_last3_rb_receiving_yards_avg_allowed"] == -6
    assert defense["defense_season_rb_yards_per_carry_allowed"] == -2


def test_current_season_excludes_report_week_future_and_other_seasons():
    defense = get_defense(
        build_rb_defense_matchup_metrics(make_metric_logs(), 2026, 5),
        "LAC",
    )

    assert defense["defense_season_games"] == 4
    assert defense["defense_season_rb_rushing_yards_avg_allowed"] == 250
    assert defense["report_season"] == 2026
    assert defense["report_week"] == 5
    assert defense["historical_season"] == 2026


def test_earlier_historical_season_uses_complete_history_and_report_context():
    data = make_logs(
        make_log(
            season=2025,
            week=1,
            game_id="2025_01_KC_LAC",
            rushing_yards=100,
        ),
        make_log(
            season=2025,
            week=18,
            game_id="2025_18_DEN_LAC",
            offense_team="DEN",
            rushing_yards=300,
        ),
        make_log(
            season=2026,
            week=1,
            game_id="2026_01_BUF_LAC",
            offense_team="BUF",
            rushing_yards=999,
        ),
    )

    defense = get_defense(
        build_rb_defense_matchup_metrics(
            data,
            report_season=2026,
            report_week=1,
            historical_season=2025,
        ),
        "LAC",
    )

    assert defense["report_season"] == 2026
    assert defense["report_week"] == 1
    assert defense["historical_season"] == 2025
    assert defense["defense_season_games"] == 2
    assert defense["defense_season_rb_rushing_yards_avg_allowed"] == 200


def test_historical_season_later_than_report_season_is_rejected():
    with pytest.raises(
        ValueError,
        match="historical_season must be less than or equal to report_season",
    ):
        build_rb_defense_matchup_metrics(
            make_metric_logs(),
            report_season=2026,
            report_week=1,
            historical_season=2027,
        )


@pytest.mark.parametrize(
    ("defense", "season_games", "last3_games"),
    [
        ("ONE", 1, 1),
        ("TWO", 2, 2),
        ("FOUR", 4, 3),
    ],
)
def test_each_defense_has_an_independent_last3_window(
    defense,
    season_games,
    last3_games,
):
    rows = []
    game_counts = {"ONE": 1, "TWO": 2, "FOUR": 4}
    for current_defense, game_count in game_counts.items():
        for week in range(1, game_count + 1):
            rows.append(
                make_log(
                    week=week,
                    game_id=f"2026_{week:02d}_OFF_{current_defense}",
                    defense=current_defense,
                    offense_team=f"O{week}",
                    rushing_yards=week * 10,
                )
            )

    result = build_rb_defense_matchup_metrics(make_logs(*rows), 2026, 6)
    row = get_defense(result, defense)

    assert row["defense_season_games"] == season_games
    assert row["defense_last3_games"] == last3_games
    if defense == "FOUR":
        assert row["defense_last3_rb_rushing_yards_avg_allowed"] == 30


def test_rank_one_has_highest_eligible_season_rushing_yards_average():
    data = make_logs(
        make_log(defense="HIGH", rushing_yards=300),
        make_log(
            game_id="2026_01_DEN_MID",
            defense="MID",
            rushing_yards=200,
        ),
        make_log(
            game_id="2026_01_BUF_LOW",
            defense="LOW",
            rushing_yards=100,
        ),
    )

    result = build_rb_defense_matchup_metrics(data, 2026, 2)

    assert result["defense"].tolist() == ["HIGH", "MID", "LOW"]
    assert result["matchup_rank"].tolist() == [1, 2, 3]


def test_ranking_uses_season_average_instead_of_last3_average():
    rows = []
    for week, yards in enumerate([1000, 0, 0, 0], start=1):
        rows.append(
            make_log(
                week=week,
                game_id=f"2026_{week:02d}_OFF_SEASON",
                defense="SEASON",
                offense_team=f"S{week}",
                rushing_yards=yards,
            )
        )
    for week, yards in enumerate([200, 200, 200, 200], start=1):
        rows.append(
            make_log(
                week=week,
                game_id=f"2026_{week:02d}_OFF_RECENT",
                defense="RECENT",
                offense_team=f"R{week}",
                rushing_yards=yards,
            )
        )

    result = build_rb_defense_matchup_metrics(make_logs(*rows), 2026, 5)

    assert get_defense(result, "SEASON")["matchup_rank"] == 1
    assert get_defense(result, "SEASON")[
        "defense_last3_rb_rushing_yards_avg_allowed"
    ] == 0
    assert get_defense(result, "RECENT")[
        "defense_last3_rb_rushing_yards_avg_allowed"
    ] == 200


def test_tied_season_averages_share_minimum_rank():
    data = make_logs(
        make_log(defense="AAA", rushing_yards=300),
        make_log(
            game_id="2026_01_DEN_BBB",
            defense="BBB",
            rushing_yards=300,
        ),
        make_log(
            game_id="2026_01_BUF_CCC",
            defense="CCC",
            rushing_yards=200,
        ),
    )

    result = build_rb_defense_matchup_metrics(data, 2026, 2)

    assert get_defense(result, "AAA")["matchup_rank"] == 1
    assert get_defense(result, "BBB")["matchup_rank"] == 1
    assert get_defense(result, "CCC")["matchup_rank"] == 3


def test_ranking_uses_only_eligible_historical_games():
    data = make_logs(
        make_log(defense="AAA", rushing_yards=300),
        make_log(
            game_id="2026_01_DEN_BBB",
            defense="BBB",
            rushing_yards=200,
        ),
        make_log(
            week=2,
            game_id="2026_02_BUF_BBB",
            defense="BBB",
            rushing_yards=999,
        ),
    )

    result = build_rb_defense_matchup_metrics(data, 2026, 2)

    assert result["defense"].tolist() == ["AAA", "BBB"]
    assert result["matchup_rank"].tolist() == [1, 2]


def test_exact_duplicate_defense_game_does_not_change_metrics():
    row = make_log()

    result = build_rb_defense_matchup_metrics(make_logs(row, row), 2026, 2)

    assert result.iloc[0]["defense_season_games"] == 1
    assert result.iloc[0][
        "defense_season_rb_rushing_yards_avg_allowed"
    ] == 100


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("offense_team", "NE"),
        ("rb_rushing_yards_allowed", 999),
        ("rb_players_used", 9),
    ],
)
def test_conflicting_duplicate_defense_games_raise_before_metrics(
    column,
    value,
):
    row = make_log()
    conflict = {**row, column: value}

    with pytest.raises(ValueError) as error:
        build_rb_defense_matchup_metrics(
            make_logs(row, conflict),
            2026,
            2,
        )

    message = str(error.value)
    assert "Conflicting RB defense-game records found for keys" in message
    assert "2026_01_KC_LAC" in message
    assert "LAC" in message


def test_missing_complete_defense_log_schema_raises_clear_error():
    data = make_logs(make_log()).drop(
        columns=["offense_team", "rb_targets_allowed"]
    )

    with pytest.raises(ValueError) as error:
        build_rb_defense_matchup_metrics(data, 2026, 2)

    message = str(error.value)
    assert "RB defense matchup data is missing required columns" in message
    assert "offense_team" in message
    assert "rb_targets_allowed" in message


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("rb_rushing_attempts_allowed", "invalid", "must be numeric"),
        ("rb_rushing_yards_allowed", np.nan, "cannot contain missing values"),
        ("rb_receiving_yards_allowed", np.inf, "must contain finite values"),
        ("rb_receiving_yards_allowed", -np.inf, "must contain finite values"),
        ("rb_targets_allowed", True, "must be numeric"),
    ],
)
def test_invalid_production_values_raise_clear_errors(column, value, message):
    row = make_log()
    row[column] = value

    with pytest.raises(ValueError, match=message):
        build_rb_defense_matchup_metrics(make_logs(row), 2026, 2)


@pytest.mark.parametrize(
    "column",
    [
        "rb_rushing_attempts_allowed",
        "rb_rushing_touchdowns_allowed",
        "rb_receptions_allowed",
        "rb_targets_allowed",
        "rb_receiving_touchdowns_allowed",
        "rb_opportunities_allowed",
        "rb_players_used",
    ],
)
def test_negative_count_or_volume_values_are_rejected(column):
    row = make_log()
    row[column] = -1

    with pytest.raises(ValueError, match="cannot contain negative values"):
        build_rb_defense_matchup_metrics(make_logs(row), 2026, 2)


@pytest.mark.parametrize(
    ("report_season", "report_week", "historical_season", "expected_name"),
    [
        (True, 2, None, "report_season"),
        (2026, False, None, "report_week"),
        (2026, 2, True, "historical_season"),
        ("2026", 2, None, "report_season"),
        (2026, 2.5, None, "report_week"),
        (2026, 2, "2025", "historical_season"),
        (0, 2, None, "report_season"),
        (2026, 0, None, "report_week"),
        (2026, 2, 0, "historical_season"),
    ],
)
def test_invalid_report_and_historical_parameters_are_rejected(
    report_season,
    report_week,
    historical_season,
    expected_name,
):
    with pytest.raises(ValueError, match=expected_name):
        build_rb_defense_matchup_metrics(
            make_logs(make_log()),
            report_season,
            report_week,
            historical_season,
        )


def test_numpy_integer_report_parameters_are_accepted():
    result = build_rb_defense_matchup_metrics(
        make_logs(make_log()),
        np.int64(2026),
        np.int64(2),
        np.int64(2026),
    )

    assert result.iloc[0]["defense_season_games"] == 1


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("defense", None, "cannot contain missing values"),
        ("offense_team", "   ", "cannot contain blank values"),
        ("season", True, "must be numeric"),
        ("week", 0, "must contain positive integers"),
        ("week", 1.5, "must contain positive integers"),
    ],
)
def test_invalid_identity_values_are_rejected(column, value, message):
    row = make_log()
    row[column] = value

    with pytest.raises(ValueError, match=message):
        build_rb_defense_matchup_metrics(make_logs(row), 2026, 2)


def test_input_order_does_not_affect_ranked_output_or_recent_windows():
    data = make_logs(
        make_log(),
        make_log(
            week=2,
            game_id="2026_02_DEN_LAC",
            offense_team="DEN",
            rushing_yards=200,
        ),
        make_log(
            week=3,
            game_id="2026_03_BUF_LAC",
            offense_team="BUF",
            rushing_yards=300,
        ),
        make_log(
            game_id="2026_01_MIA_BUF",
            defense="BUF",
            offense_team="MIA",
            rushing_yards=150,
        ),
    )
    expected = build_rb_defense_matchup_metrics(data, 2026, 4)
    shuffled = build_rb_defense_matchup_metrics(
        data.sample(frac=1, random_state=42).reset_index(drop=True),
        2026,
        4,
    )

    pd.testing.assert_frame_equal(shuffled, expected)


def test_input_dataframe_remains_unchanged():
    data = make_metric_logs()
    original = data.copy(deep=True)

    build_rb_defense_matchup_metrics(data, 2026, 5)

    pd.testing.assert_frame_equal(data, original)


def test_valid_empty_input_returns_complete_output_schema():
    data = pd.DataFrame(columns=DEFENSE_RB_GAME_LOG_COLUMNS)

    result = build_rb_defense_matchup_metrics(data, 2026, 5)

    assert result.empty
    assert result.columns.tolist() == OUTPUT_COLUMNS


def test_no_eligible_history_returns_complete_output_schema_without_fallback():
    result = build_rb_defense_matchup_metrics(make_metric_logs(), 2026, 1)

    assert result.empty
    assert result.columns.tolist() == OUTPUT_COLUMNS


def test_different_defenses_remain_one_row_each():
    data = make_logs(
        make_log(defense="LAC", rushing_yards=200),
        make_log(
            game_id="2026_01_DEN_BUF",
            defense="BUF",
            offense_team="DEN",
            rushing_yards=100,
        ),
        make_log(
            week=2,
            game_id="2026_02_MIA_BUF",
            defense="BUF",
            offense_team="MIA",
            rushing_yards=300,
        ),
    )

    result = build_rb_defense_matchup_metrics(data, 2026, 3)

    assert result["defense"].tolist() == ["BUF", "LAC"]
    assert not result.duplicated(subset=["defense"]).any()
    assert get_defense(result, "BUF")["defense_season_games"] == 2
    assert get_defense(result, "LAC")["defense_season_games"] == 1


def test_output_uses_complete_documented_column_order():
    result = build_rb_defense_matchup_metrics(
        make_logs(make_log()),
        2026,
        2,
    )

    assert result.columns.tolist() == OUTPUT_COLUMNS
