import numpy as np
import pandas as pd
import pytest

from football.features.primary_quarterbacks import PRIMARY_QUARTERBACK_COLUMNS
from football.features.qb_form_metrics import (
    OUTPUT_COLUMNS,
    build_qb_form_metrics,
)


def make_primary_qb_rows() -> pd.DataFrame:
    """Create unsorted primary-quarterback rows."""

    rows = [
        make_row(
            season=2026,
            week=5,
            game_id="2026_05_AAA_BBB",
            player_id="qb_1",
            player_name="Alex Smith",
            team="DDD",
            passing_yards=999,
            primary_attempts=60,
        ),
        make_row(
            season=2026,
            week=3,
            game_id="2026_03_AAA_BBB",
            player_id="qb_1",
            player_name="Alex Smith",
            team="BBB",
            passing_yards=300,
            primary_attempts=40,
            similar_attempt_split=True,
            exact_attempt_tie=True,
        ),
        make_row(
            season=2026,
            week=1,
            game_id="2026_01_CCC_DDD",
            player_id="qb_2",
            player_name="Blake Young",
            team="EEE",
            passing_yards=150,
            primary_attempts=15,
        ),
        make_row(
            season=2026,
            week=1,
            game_id="2026_01_AAA_BBB",
            player_id="qb_1",
            player_name="Alex Smith",
            team="AAA",
            passing_yards=100,
            primary_attempts=20,
        ),
        make_row(
            season=2025,
            week=4,
            game_id="2025_04_AAA_BBB",
            player_id="qb_1",
            player_name="Alex Smith",
            team="OLD",
            passing_yards=800,
            primary_attempts=80,
            low_attempt_primary_qb=True,
        ),
        make_row(
            season=2026,
            week=4,
            game_id="2026_04_AAA_BBB",
            player_id="qb_1",
            player_name="Alex Smith",
            team="CCC",
            passing_yards=400,
            primary_attempts=50,
        ),
        make_row(
            season=2026,
            week=2,
            game_id="2026_02_AAA_BBB",
            player_id="qb_1",
            player_name="Alex Smith",
            team="BBB",
            passing_yards=200,
            primary_attempts=30,
            low_attempt_primary_qb=True,
        ),
        make_row(
            season=2026,
            week=3,
            game_id="2026_03_CCC_DDD",
            player_id="qb_2",
            player_name="Blake Young",
            team="EEE",
            passing_yards=210,
            primary_attempts=21,
        ),
        make_row(
            season=2026,
            week=6,
            game_id="2026_06_AAA_BBB",
            player_id="qb_1",
            player_name="Alex Smith",
            team="FUT",
            passing_yards=1000,
            primary_attempts=70,
            exact_attempt_tie=True,
        ),
        make_row(
            season=2026,
            week=2,
            game_id="2026_02_EEE_FFF",
            player_id="qb_3",
            player_name="Alex Smith",
            team="GGG",
            passing_yards=50,
            primary_attempts=10,
            exact_attempt_tie=True,
        ),
    ]

    return pd.DataFrame(rows, columns=PRIMARY_QUARTERBACK_COLUMNS)


def make_row(
    *,
    season: int,
    week: int,
    game_id: str,
    player_id: str,
    player_name: str,
    team: str,
    passing_yards: int,
    primary_attempts: int,
    low_attempt_primary_qb: bool = False,
    similar_attempt_split: bool = False,
    exact_attempt_tie: bool = False,
) -> dict[str, object]:
    return {
        "season": season,
        "week": week,
        "game_id": game_id,
        "player_id": player_id,
        "player_name": player_name,
        "team": team,
        "opponent": "OPP",
        "passing_attempts": primary_attempts,
        "completions": primary_attempts // 2,
        "passing_yards": passing_yards,
        "passing_touchdowns": 1,
        "interceptions": 0,
        "primary_attempts": primary_attempts,
        "secondary_attempts": 0,
        "quarterbacks_with_attempts": 1,
        "low_attempt_primary_qb": low_attempt_primary_qb,
        "similar_attempt_split": similar_attempt_split,
        "exact_attempt_tie": exact_attempt_tie,
    }


def get_qb(result: pd.DataFrame, player_id: str) -> pd.Series:
    return result.loc[result["player_id"] == player_id].iloc[0]


def test_build_qb_form_metrics_excludes_report_week_records():
    result = build_qb_form_metrics(make_primary_qb_rows(), 2026, 5)

    qb = get_qb(result, "qb_1")

    assert qb["qb_season_avg"] == 250
    assert qb["qb_season_games"] == 4
    assert qb["latest_team"] == "CCC"


def test_build_qb_form_metrics_excludes_future_records():
    result = build_qb_form_metrics(make_primary_qb_rows(), 2026, 5)

    qb = get_qb(result, "qb_1")

    assert qb["qb_season_attempts_avg"] == 35
    assert qb["flagged_games"] == 2


def test_build_qb_form_metrics_excludes_other_seasons():
    result = build_qb_form_metrics(make_primary_qb_rows(), 2026, 5)

    qb = get_qb(result, "qb_1")

    assert qb["qb_season_avg"] == 250
    assert qb["flagged_games"] == 2


def test_build_qb_form_metrics_groups_by_player_id_not_name():
    result = build_qb_form_metrics(make_primary_qb_rows(), 2026, 5)

    alex_rows = result.loc[result["player_name"] == "Alex Smith"]

    assert alex_rows["player_id"].tolist() == ["qb_1", "qb_3"]
    assert get_qb(result, "qb_1")["qb_season_avg"] == 250
    assert get_qb(result, "qb_3")["qb_season_avg"] == 50


def test_build_qb_form_metrics_calculates_season_passing_yards_average():
    result = build_qb_form_metrics(make_primary_qb_rows(), 2026, 5)

    assert get_qb(result, "qb_1")["qb_season_avg"] == 250


def test_build_qb_form_metrics_uses_three_most_recent_prior_games():
    result = build_qb_form_metrics(make_primary_qb_rows(), 2026, 5)

    qb = get_qb(result, "qb_1")

    assert qb["qb_last3_avg"] == 300
    assert qb["qb_last3_games"] == 3


def test_build_qb_form_metrics_calculates_attempt_averages():
    result = build_qb_form_metrics(make_primary_qb_rows(), 2026, 5)

    qb = get_qb(result, "qb_1")

    assert qb["qb_season_attempts_avg"] == 35
    assert qb["qb_last3_attempts_avg"] == 40


def test_build_qb_form_metrics_uses_all_when_fewer_than_three_prior_games():
    result = build_qb_form_metrics(make_primary_qb_rows(), 2026, 5)

    qb = get_qb(result, "qb_2")

    assert qb["qb_season_avg"] == 180
    assert qb["qb_last3_avg"] == 180
    assert qb["qb_season_attempts_avg"] == 18
    assert qb["qb_last3_attempts_avg"] == 18


def test_build_qb_form_metrics_selects_latest_team_from_latest_prior_game():
    result = build_qb_form_metrics(make_primary_qb_rows(), 2026, 5)

    assert get_qb(result, "qb_1")["latest_team"] == "CCC"


def test_build_qb_form_metrics_handles_player_changing_teams():
    result = build_qb_form_metrics(make_primary_qb_rows(), 2026, 4)

    assert get_qb(result, "qb_1")["latest_team"] == "BBB"


def test_build_qb_form_metrics_counts_season_and_last3_samples():
    result = build_qb_form_metrics(make_primary_qb_rows(), 2026, 5)

    qb_1 = get_qb(result, "qb_1")
    qb_2 = get_qb(result, "qb_2")

    assert qb_1["qb_season_games"] == 4
    assert qb_1["qb_last3_games"] == 3
    assert qb_2["qb_season_games"] == 2
    assert qb_2["qb_last3_games"] == 2


def test_build_qb_form_metrics_counts_unique_flagged_games_once():
    data = pd.DataFrame(
        [
            make_row(
                season=2026,
                week=1,
                game_id="2026_01_AAA_BBB",
                player_id="qb_1",
                player_name="Alex Smith",
                team="AAA",
                passing_yards=100,
                primary_attempts=20,
                low_attempt_primary_qb=True,
            ),
            make_row(
                season=2026,
                week=2,
                game_id="2026_02_AAA_BBB",
                player_id="qb_1",
                player_name="Alex Smith",
                team="AAA",
                passing_yards=200,
                primary_attempts=30,
                similar_attempt_split=True,
            ),
            make_row(
                season=2026,
                week=3,
                game_id="2026_03_AAA_BBB",
                player_id="qb_1",
                player_name="Alex Smith",
                team="AAA",
                passing_yards=300,
                primary_attempts=40,
            ),
        ],
        columns=PRIMARY_QUARTERBACK_COLUMNS,
    )

    result = build_qb_form_metrics(data, 2026, 4)

    assert get_qb(result, "qb_1")["flagged_games"] == 2


def test_build_qb_form_metrics_ignores_identical_duplicate_rows():
    rows = [
        make_row(
            season=2026,
            week=1,
            game_id="2026_01_AAA_BBB",
            player_id="qb_1",
            player_name="Alex Smith",
            team="AAA",
            passing_yards=100,
            primary_attempts=20,
            low_attempt_primary_qb=True,
        ),
        make_row(
            season=2026,
            week=1,
            game_id="2026_01_AAA_BBB",
            player_id="qb_1",
            player_name="Alex Smith",
            team="AAA",
            passing_yards=100,
            primary_attempts=20,
            low_attempt_primary_qb=True,
        ),
        make_row(
            season=2026,
            week=2,
            game_id="2026_02_AAA_BBB",
            player_id="qb_1",
            player_name="Alex Smith",
            team="AAA",
            passing_yards=300,
            primary_attempts=40,
        ),
    ]
    data = pd.DataFrame(rows, columns=PRIMARY_QUARTERBACK_COLUMNS)

    result = build_qb_form_metrics(data, 2026, 3)
    qb = get_qb(result, "qb_1")

    assert qb["qb_season_avg"] == 200
    assert qb["qb_season_attempts_avg"] == 30
    assert qb["qb_season_games"] == 2
    assert qb["qb_last3_games"] == 2
    assert qb["flagged_games"] == 1


@pytest.mark.parametrize(
    ("column", "new_value"),
    [
        ("player_name", "Different Name"),
        ("passing_yards", 999),
        ("passing_attempts", 99),
        ("primary_attempts", 99),
        ("team", "ZZZ"),
        ("low_attempt_primary_qb", True),
        ("similar_attempt_split", True),
        ("exact_attempt_tie", True),
    ],
)
def test_build_qb_form_metrics_rejects_conflicting_duplicate_games(
    column,
    new_value,
):
    row = make_row(
        season=2026,
        week=1,
        game_id="2026_01_AAA_BBB",
        player_id="qb_1",
        player_name="Alex Smith",
        team="AAA",
        passing_yards=100,
        primary_attempts=20,
    )
    conflicting_row = {**row, column: new_value}
    data = pd.DataFrame([row, conflicting_row], columns=PRIMARY_QUARTERBACK_COLUMNS)

    with pytest.raises(ValueError) as error:
        build_qb_form_metrics(data, 2026, 2)

    message = str(error.value)

    assert "Conflicting primary quarterback-game records found for keys" in message
    assert "2026_01_AAA_BBB" in message
    assert "qb_1" in message


def test_build_qb_form_metrics_duplicate_records_do_not_create_duplicate_output_rows():
    row = make_row(
        season=2026,
        week=1,
        game_id="2026_01_AAA_BBB",
        player_id="qb_1",
        player_name="Alex Smith",
        team="AAA",
        passing_yards=100,
        primary_attempts=20,
    )
    data = pd.DataFrame([row, row], columns=PRIMARY_QUARTERBACK_COLUMNS)

    result = build_qb_form_metrics(data, 2026, 2)

    assert result["player_id"].tolist() == ["qb_1"]


def test_build_qb_form_metrics_last3_uses_three_unique_quarterback_games():
    duplicate_week_4 = make_row(
        season=2026,
        week=4,
        game_id="2026_04_AAA_BBB",
        player_id="qb_1",
        player_name="Alex Smith",
        team="AAA",
        passing_yards=400,
        primary_attempts=40,
    )
    rows = [
        make_row(
            season=2026,
            week=1,
            game_id="2026_01_AAA_BBB",
            player_id="qb_1",
            player_name="Alex Smith",
            team="AAA",
            passing_yards=100,
            primary_attempts=10,
        ),
        make_row(
            season=2026,
            week=2,
            game_id="2026_02_AAA_BBB",
            player_id="qb_1",
            player_name="Alex Smith",
            team="AAA",
            passing_yards=200,
            primary_attempts=20,
        ),
        make_row(
            season=2026,
            week=3,
            game_id="2026_03_AAA_BBB",
            player_id="qb_1",
            player_name="Alex Smith",
            team="AAA",
            passing_yards=300,
            primary_attempts=30,
        ),
        duplicate_week_4,
        duplicate_week_4,
    ]
    data = pd.DataFrame(rows, columns=PRIMARY_QUARTERBACK_COLUMNS)

    result = build_qb_form_metrics(data, 2026, 5)
    qb = get_qb(result, "qb_1")

    assert qb["qb_last3_avg"] == 300
    assert qb["qb_last3_attempts_avg"] == 30
    assert qb["qb_last3_games"] == 3


def test_build_qb_form_metrics_is_chronological_with_unsorted_input():
    result = build_qb_form_metrics(make_primary_qb_rows(), 2026, 5)

    qb = get_qb(result, "qb_1")

    assert qb["qb_last3_avg"] == 300
    assert qb["latest_team"] == "CCC"


def test_build_qb_form_metrics_week_1_returns_empty_schema():
    result = build_qb_form_metrics(make_primary_qb_rows(), 2026, 1)

    assert result.empty
    assert result.columns.tolist() == OUTPUT_COLUMNS


def test_build_qb_form_metrics_validates_missing_columns():
    data = make_primary_qb_rows().drop(
        columns=["passing_yards", "similar_attempt_split"]
    )

    with pytest.raises(ValueError) as error:
        build_qb_form_metrics(data, 2026, 5)

    message = str(error.value)

    assert "Primary quarterback form data is missing required columns" in message
    assert "passing_yards" in message
    assert "similar_attempt_split" in message


@pytest.mark.parametrize(
    ("report_season", "report_week", "expected_message"),
    [
        (True, 5, "report_season"),
        ("2026", 5, "report_season"),
        (0, 5, "report_season"),
        (2026, False, "report_week"),
        (2026, 4.5, "report_week"),
        (2026, 0, "report_week"),
    ],
)
def test_build_qb_form_metrics_validates_report_values(
    report_season,
    report_week,
    expected_message,
):
    with pytest.raises(ValueError, match=expected_message):
        build_qb_form_metrics(make_primary_qb_rows(), report_season, report_week)


def test_build_qb_form_metrics_accepts_integer_scalar_report_values():
    result = build_qb_form_metrics(make_primary_qb_rows(), np.int64(2026), np.int64(5))

    assert get_qb(result, "qb_1")["qb_season_games"] == 4


def test_build_qb_form_metrics_does_not_mutate_input():
    data = make_primary_qb_rows()
    original = data.copy(deep=True)

    build_qb_form_metrics(data, 2026, 5)

    pd.testing.assert_frame_equal(data, original)


def test_build_qb_form_metrics_sorts_by_player_name_and_player_id():
    result = build_qb_form_metrics(make_primary_qb_rows(), 2026, 5)

    assert result["player_id"].tolist() == ["qb_1", "qb_3", "qb_2"]
