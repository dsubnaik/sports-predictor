import numpy as np
import pandas as pd
import pytest

from football.features.rb_form_metrics import (
    OUTPUT_COLUMNS,
    build_rb_form_metrics,
)
from football.features.running_back_usage import RUNNING_BACK_USAGE_COLUMNS


def make_row(
    *,
    season: int,
    week: int,
    player_id: str,
    player_name: str,
    team: str,
    rushing_attempts: float,
    rushing_yards: float,
    receptions: float,
    targets: float,
    receiving_yards: float,
    carry_share: float,
    target_share: float,
    opportunity_share: float,
    game_id: str | None = None,
) -> dict[str, object]:
    opportunities = rushing_attempts + targets
    return {
        "season": season,
        "week": week,
        "game_id": game_id or f"{season}_{week:02d}_{team}_OPP",
        "player_id": player_id,
        "player_name": player_name,
        "team": team,
        "opponent": "OPP",
        "position": "RB",
        "rushing_attempts": rushing_attempts,
        "rushing_yards": rushing_yards,
        "rushing_touchdowns": 0,
        "receptions": receptions,
        "targets": targets,
        "receiving_yards": receiving_yards,
        "receiving_touchdowns": 0,
        "opportunities": opportunities,
        "team_rb_rushing_attempts": rushing_attempts,
        "team_rb_targets": targets,
        "team_rb_opportunities": opportunities,
        "carry_share": carry_share,
        "target_share": target_share,
        "opportunity_share": opportunity_share,
        "lead_team_rusher": False,
        "lead_targeted_rb": False,
        "backfield_opportunity_leader": False,
        "tied_backfield_opportunity_leader": False,
        "low_volume_rb": opportunities <= 3,
        "shared_backfield": False,
    }


def make_annotated_rb_rows() -> pd.DataFrame:
    rows = [
        make_row(
            season=2026,
            week=5,
            player_id="rb_1",
            player_name="Alex Runner",
            team="LEAK",
            rushing_attempts=99,
            rushing_yards=999,
            receptions=9,
            targets=10,
            receiving_yards=999,
            carry_share=0.99,
            target_share=0.99,
            opportunity_share=0.99,
        ),
        make_row(
            season=2026,
            week=3,
            player_id="rb_1",
            player_name="Alex Runner",
            team="BBB",
            rushing_attempts=0,
            rushing_yards=0,
            receptions=1,
            targets=2,
            receiving_yards=12,
            carry_share=0.0,
            target_share=0.2,
            opportunity_share=0.1,
        ),
        make_row(
            season=2026,
            week=1,
            player_id="rb_2",
            player_name="Blake Backup",
            team="AAA",
            rushing_attempts=1,
            rushing_yards=2,
            receptions=1,
            targets=1,
            receiving_yards=3,
            carry_share=0.05,
            target_share=0.1,
            opportunity_share=0.07,
            game_id="2026_01_AAA_OPP",
        ),
        make_row(
            season=2026,
            week=1,
            player_id="rb_1",
            player_name="Alex Runner",
            team="AAA",
            rushing_attempts=10,
            rushing_yards=40,
            receptions=2,
            targets=3,
            receiving_yards=15,
            carry_share=0.5,
            target_share=0.3,
            opportunity_share=0.4,
            game_id="2026_01_AAA_OPP",
        ),
        make_row(
            season=2025,
            week=4,
            player_id="rb_1",
            player_name="Alex Runner",
            team="OLD",
            rushing_attempts=10,
            rushing_yards=80,
            receptions=2,
            targets=3,
            receiving_yards=-4,
            carry_share=0.5,
            target_share=0.3,
            opportunity_share=0.4,
        ),
        make_row(
            season=2026,
            week=4,
            player_id="rb_1",
            player_name="Alex Runner",
            team="CCC",
            rushing_attempts=5,
            rushing_yards=25,
            receptions=0,
            targets=1,
            receiving_yards=-2,
            carry_share=0.25,
            target_share=0.1,
            opportunity_share=0.2,
        ),
        make_row(
            season=2026,
            week=2,
            player_id="rb_1",
            player_name="Alex Runner",
            team="AAA",
            rushing_attempts=20,
            rushing_yards=100,
            receptions=3,
            targets=4,
            receiving_yards=30,
            carry_share=0.8,
            target_share=0.4,
            opportunity_share=0.6,
        ),
        make_row(
            season=2026,
            week=3,
            player_id="rb_2",
            player_name="Blake Backup",
            team="AAA",
            rushing_attempts=2,
            rushing_yards=8,
            receptions=0,
            targets=1,
            receiving_yards=0,
            carry_share=0.1,
            target_share=0.1,
            opportunity_share=0.1,
            game_id="2026_03_AAA_OPP",
        ),
        make_row(
            season=2026,
            week=6,
            player_id="rb_1",
            player_name="Alex Runner",
            team="FUT",
            rushing_attempts=100,
            rushing_yards=1000,
            receptions=10,
            targets=10,
            receiving_yards=1000,
            carry_share=1.0,
            target_share=1.0,
            opportunity_share=1.0,
        ),
        make_row(
            season=2026,
            week=2,
            player_id="rb_3",
            player_name="Alex Runner",
            team="DDD",
            rushing_attempts=3,
            rushing_yards=9,
            receptions=0,
            targets=0,
            receiving_yards=0,
            carry_share=0.2,
            target_share=0.0,
            opportunity_share=0.2,
            game_id="2026_02_DDD_OPP",
        ),
    ]
    return pd.DataFrame(rows, columns=RUNNING_BACK_USAGE_COLUMNS)


def get_rb(result: pd.DataFrame, player_id: str) -> pd.Series:
    return result.loc[result["player_id"] == player_id].iloc[0]


def test_build_rb_form_metrics_calculates_season_and_last3_rushing_metrics():
    rb = get_rb(build_rb_form_metrics(make_annotated_rb_rows(), 2026, 5), "rb_1")

    assert rb["rb_season_rushing_yards_avg"] == pytest.approx(165 / 4)
    assert rb["rb_last3_rushing_yards_avg"] == pytest.approx(125 / 3)
    assert rb["rb_season_rushing_attempts_avg"] == pytest.approx(35 / 4)
    assert rb["rb_last3_rushing_attempts_avg"] == pytest.approx(25 / 3)


def test_build_rb_form_metrics_calculates_receiving_averages():
    rb = get_rb(build_rb_form_metrics(make_annotated_rb_rows(), 2026, 5), "rb_1")

    assert rb["rb_season_receptions_avg"] == pytest.approx(6 / 4)
    assert rb["rb_last3_receptions_avg"] == pytest.approx(4 / 3)
    assert rb["rb_season_targets_avg"] == pytest.approx(10 / 4)
    assert rb["rb_last3_targets_avg"] == pytest.approx(7 / 3)
    assert rb["rb_season_receiving_yards_avg"] == pytest.approx(55 / 4)
    assert rb["rb_last3_receiving_yards_avg"] == pytest.approx(40 / 3)


def test_build_rb_form_metrics_calculates_opportunity_averages():
    rb = get_rb(build_rb_form_metrics(make_annotated_rb_rows(), 2026, 5), "rb_1")

    assert rb["rb_season_opportunities_avg"] == pytest.approx(45 / 4)
    assert rb["rb_last3_opportunities_avg"] == pytest.approx(32 / 3)


def test_build_rb_form_metrics_calculates_usage_share_averages():
    rb = get_rb(build_rb_form_metrics(make_annotated_rb_rows(), 2026, 5), "rb_1")

    assert rb["rb_season_carry_share_avg"] == pytest.approx(1.55 / 4)
    assert rb["rb_last3_carry_share_avg"] == pytest.approx(1.05 / 3)
    assert rb["rb_season_target_share_avg"] == pytest.approx(1.0 / 4)
    assert rb["rb_last3_target_share_avg"] == pytest.approx(0.7 / 3)
    assert rb["rb_season_opportunity_share_avg"] == pytest.approx(1.3 / 4)
    assert rb["rb_last3_opportunity_share_avg"] == pytest.approx(0.9 / 3)


def test_build_rb_form_metrics_uses_weighted_yards_per_carry():
    rb = get_rb(build_rb_form_metrics(make_annotated_rb_rows(), 2026, 5), "rb_1")

    assert rb["rb_season_yards_per_carry"] == pytest.approx(165 / 35)
    assert rb["rb_last3_yards_per_carry"] == pytest.approx(125 / 25)


def test_build_rb_form_metrics_zero_attempts_produce_finite_zero_yards_per_carry():
    data = pd.DataFrame(
        [
            make_row(
                season=2026,
                week=1,
                player_id="rb_zero",
                player_name="Zero Carry",
                team="AAA",
                rushing_attempts=0,
                rushing_yards=0,
                receptions=1,
                targets=2,
                receiving_yards=5,
                carry_share=0,
                target_share=1,
                opportunity_share=1,
            )
        ],
        columns=RUNNING_BACK_USAGE_COLUMNS,
    )

    rb = get_rb(build_rb_form_metrics(data, 2026, 2), "rb_zero")

    assert rb["rb_season_yards_per_carry"] == 0.0
    assert rb["rb_last3_yards_per_carry"] == 0.0
    assert np.isfinite(rb["rb_season_yards_per_carry"])
    assert np.isfinite(rb["rb_last3_yards_per_carry"])


def test_build_rb_form_metrics_accepts_and_calculates_negative_yardage():
    data = pd.DataFrame(
        [
            make_row(
                season=2026,
                week=1,
                player_id="rb_negative",
                player_name="Negative Yardage",
                team="AAA",
                rushing_attempts=5,
                rushing_yards=-10,
                receptions=1,
                targets=2,
                receiving_yards=-6,
                carry_share=0.5,
                target_share=0.5,
                opportunity_share=0.5,
            )
        ],
        columns=RUNNING_BACK_USAGE_COLUMNS,
    )

    rb = get_rb(build_rb_form_metrics(data, 2026, 2), "rb_negative")

    assert rb["rb_season_rushing_yards_avg"] == -10
    assert rb["rb_last3_rushing_yards_avg"] == -10
    assert rb["rb_season_receiving_yards_avg"] == -6
    assert rb["rb_last3_receiving_yards_avg"] == -6
    assert rb["rb_season_yards_per_carry"] == -2
    assert rb["rb_last3_yards_per_carry"] == -2


def test_build_rb_form_metrics_excludes_report_week_and_future_values():
    rb = get_rb(build_rb_form_metrics(make_annotated_rb_rows(), 2026, 5), "rb_1")

    assert rb["rb_season_games"] == 4
    assert rb["rb_season_rushing_yards_avg"] == pytest.approx(165 / 4)
    assert rb["rb_season_carry_share_avg"] == pytest.approx(1.55 / 4)
    assert rb["latest_team"] == "CCC"


def test_build_rb_form_metrics_current_season_uses_only_prior_weeks():
    rb = get_rb(build_rb_form_metrics(make_annotated_rb_rows(), 2026, 3), "rb_1")

    assert rb["rb_season_games"] == 2
    assert rb["rb_season_rushing_yards_avg"] == 70
    assert rb["latest_team"] == "AAA"


def test_build_rb_form_metrics_earlier_history_uses_full_season_and_report_context():
    result = build_rb_form_metrics(
        make_annotated_rb_rows(),
        report_season=2026,
        report_week=1,
        historical_season=2025,
    )
    rb = get_rb(result, "rb_1")

    assert rb["report_season"] == 2026
    assert rb["report_week"] == 1
    assert rb["historical_season"] == 2025
    assert rb["rb_season_games"] == 1
    assert rb["rb_season_rushing_yards_avg"] == 80
    assert rb["latest_team"] == "OLD"


def test_build_rb_form_metrics_rejects_history_later_than_report_season():
    with pytest.raises(
        ValueError,
        match="historical_season must be less than or equal to report_season",
    ):
        build_rb_form_metrics(
            make_annotated_rb_rows(),
            report_season=2026,
            report_week=1,
            historical_season=2027,
        )


@pytest.mark.parametrize("historical_season", [True, "2025", 0, 2025.5])
def test_build_rb_form_metrics_validates_explicit_historical_season(
    historical_season,
):
    with pytest.raises(
        ValueError,
        match="historical_season must be an integer greater than or equal to 1",
    ):
        build_rb_form_metrics(
            make_annotated_rb_rows(),
            report_season=2026,
            report_week=1,
            historical_season=historical_season,
        )


def test_build_rb_form_metrics_gives_each_player_an_independent_last3_window():
    result = build_rb_form_metrics(make_annotated_rb_rows(), 2026, 5)

    assert get_rb(result, "rb_1")["rb_last3_games"] == 3
    assert get_rb(result, "rb_2")["rb_last3_games"] == 2
    assert get_rb(result, "rb_2")["rb_last3_rushing_yards_avg"] == 5


def test_build_rb_form_metrics_preserves_multiple_backfield_players_and_backups():
    result = build_rb_form_metrics(make_annotated_rb_rows(), 2026, 2)

    assert result["player_id"].tolist() == ["rb_1", "rb_2"]
    assert get_rb(result, "rb_2")["rb_season_games"] == 1
    assert get_rb(result, "rb_2")["rb_season_opportunities_avg"] == 2


def test_build_rb_form_metrics_retains_zero_carry_games_in_game_averages():
    rb = get_rb(build_rb_form_metrics(make_annotated_rb_rows(), 2026, 5), "rb_1")

    assert rb["rb_season_games"] == 4
    assert rb["rb_season_rushing_attempts_avg"] == pytest.approx(35 / 4)


def test_build_rb_form_metrics_groups_by_player_id_not_name():
    result = build_rb_form_metrics(make_annotated_rb_rows(), 2026, 5)

    same_name = result.loc[result["player_name"] == "Alex Runner"]

    assert same_name["player_id"].tolist() == ["rb_1", "rb_3"]
    assert get_rb(result, "rb_1")["rb_season_games"] == 4
    assert get_rb(result, "rb_3")["rb_season_games"] == 1


def test_build_rb_form_metrics_uses_latest_eligible_team_after_team_change():
    result = build_rb_form_metrics(make_annotated_rb_rows(), 2026, 5)

    assert get_rb(result, "rb_1")["latest_team"] == "CCC"


@pytest.mark.parametrize(
    ("report_week", "season_games", "last3_games", "last3_yards_avg"),
    [
        (2, 1, 1, 40),
        (3, 2, 2, 70),
        (5, 4, 3, 125 / 3),
    ],
)
def test_build_rb_form_metrics_uses_available_games_up_to_three(
    report_week,
    season_games,
    last3_games,
    last3_yards_avg,
):
    rb = get_rb(
        build_rb_form_metrics(make_annotated_rb_rows(), 2026, report_week),
        "rb_1",
    )

    assert rb["rb_season_games"] == season_games
    assert rb["rb_last3_games"] == last3_games
    assert rb["rb_last3_rushing_yards_avg"] == pytest.approx(last3_yards_avg)


def test_build_rb_form_metrics_is_deterministic_for_input_order():
    data = make_annotated_rb_rows()
    expected = build_rb_form_metrics(data, 2026, 5)
    shuffled = build_rb_form_metrics(
        data.sample(frac=1, random_state=42).reset_index(drop=True),
        2026,
        5,
    )

    pd.testing.assert_frame_equal(shuffled, expected)
    assert expected["player_id"].tolist() == ["rb_1", "rb_3", "rb_2"]


def test_build_rb_form_metrics_does_not_mutate_input():
    data = make_annotated_rb_rows()
    original = data.copy(deep=True)

    build_rb_form_metrics(data, 2026, 5)

    pd.testing.assert_frame_equal(data, original)


def test_build_rb_form_metrics_returns_empty_schema_without_eligible_history():
    result = build_rb_form_metrics(make_annotated_rb_rows(), 2024, 8)

    assert result.empty
    assert result.columns.tolist() == OUTPUT_COLUMNS


def test_build_rb_form_metrics_returns_empty_schema_for_valid_empty_input():
    data = pd.DataFrame(columns=RUNNING_BACK_USAGE_COLUMNS)

    result = build_rb_form_metrics(data, 2026, 5)

    assert result.empty
    assert result.columns.tolist() == OUTPUT_COLUMNS


def test_build_rb_form_metrics_validates_complete_annotated_schema():
    data = make_annotated_rb_rows().drop(
        columns=["rushing_yards", "shared_backfield"]
    )

    with pytest.raises(ValueError) as error:
        build_rb_form_metrics(data, 2026, 5)

    message = str(error.value)
    assert "Running back form data is missing required columns" in message
    assert "rushing_yards" in message
    assert "shared_backfield" in message


def test_build_rb_form_metrics_ignores_exact_duplicate_games():
    row = make_annotated_rb_rows().iloc[3].to_dict()
    data = pd.DataFrame([row, row], columns=RUNNING_BACK_USAGE_COLUMNS)

    rb = get_rb(build_rb_form_metrics(data, 2026, 2), "rb_1")

    assert rb["rb_season_games"] == 1
    assert rb["rb_season_rushing_yards_avg"] == 40


@pytest.mark.parametrize(
    ("column", "new_value"),
    [
        ("player_name", "Different Name"),
        ("team", "ZZZ"),
        ("rushing_yards", 999),
        ("carry_share", 0.99),
        ("low_volume_rb", True),
    ],
)
def test_build_rb_form_metrics_rejects_conflicting_duplicate_games(
    column,
    new_value,
):
    row = make_annotated_rb_rows().iloc[3].to_dict()
    conflict = {**row, column: new_value}
    data = pd.DataFrame([row, conflict], columns=RUNNING_BACK_USAGE_COLUMNS)

    with pytest.raises(ValueError) as error:
        build_rb_form_metrics(data, 2026, 2)

    message = str(error.value)
    assert "Conflicting running-back form records found for keys" in message
    assert "2026_01_AAA_OPP" in message
    assert "rb_1" in message


@pytest.mark.parametrize(
    ("report_season", "report_week", "expected_name"),
    [
        ("2026", 5, "report_season"),
        (0, 5, "report_season"),
        (2026, 4.5, "report_week"),
        (2026, 0, "report_week"),
    ],
)
def test_build_rb_form_metrics_validates_report_value_types(
    report_season,
    report_week,
    expected_name,
):
    with pytest.raises(
        ValueError,
        match=f"{expected_name} must be an integer greater than or equal to 1",
    ):
        build_rb_form_metrics(
            make_annotated_rb_rows(),
            report_season,
            report_week,
        )


@pytest.mark.parametrize(
    ("report_season", "report_week", "expected_name"),
    [
        (True, 5, "report_season"),
        (2026, False, "report_week"),
    ],
)
def test_build_rb_form_metrics_rejects_boolean_report_values(
    report_season,
    report_week,
    expected_name,
):
    with pytest.raises(ValueError, match=expected_name):
        build_rb_form_metrics(
            make_annotated_rb_rows(),
            report_season,
            report_week,
        )


def test_build_rb_form_metrics_accepts_integer_scalar_report_values():
    result = build_rb_form_metrics(
        make_annotated_rb_rows(),
        np.int64(2026),
        np.int64(5),
    )

    assert get_rb(result, "rb_1")["rb_season_games"] == 4


@pytest.mark.parametrize("column", ["rushing_yards", "receiving_yards"])
def test_build_rb_form_metrics_rejects_non_numeric_yardage(column):
    data = make_annotated_rb_rows()
    data[column] = data[column].astype(object)
    data.loc[data.index[0], column] = "invalid"

    with pytest.raises(ValueError, match="must be numeric"):
        build_rb_form_metrics(data, 2026, 5)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_build_rb_form_metrics_rejects_missing_or_non_finite_metrics(value):
    data = make_annotated_rb_rows()
    data["receiving_yards"] = data["receiving_yards"].astype(float)
    data.loc[data.index[0], "receiving_yards"] = value
    expected = "cannot contain missing values" if np.isnan(value) else "finite values"

    with pytest.raises(ValueError, match=expected):
        build_rb_form_metrics(data, 2026, 5)


@pytest.mark.parametrize(
    "column",
    [
        "rushing_attempts",
        "receptions",
        "targets",
        "opportunities",
        "carry_share",
        "target_share",
        "opportunity_share",
    ],
)
def test_build_rb_form_metrics_rejects_negative_workload_values(column):
    data = make_annotated_rb_rows()
    data.loc[data.index[0], column] = -1

    with pytest.raises(ValueError, match="cannot contain negative values"):
        build_rb_form_metrics(data, 2026, 5)
