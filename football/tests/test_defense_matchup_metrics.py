import pandas as pd
import pytest

from football.features.defense_matchup_metrics import (
    OUTPUT_COLUMNS,
    build_defense_matchup_metrics,
)


def make_defense_logs() -> pd.DataFrame:
    """Create unsorted defense-versus-QB game logs."""

    return pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 5,
                "game_id": "2026_05_LAC_LV",
                "defense": "LV",
                "passing_yards_allowed": 999,
                "low_attempt_primary_qb": False,
                "similar_attempt_split": False,
                "exact_attempt_tie": False,
            },
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_DEN_KC",
                "defense": "DEN",
                "passing_yards_allowed": 100,
                "low_attempt_primary_qb": False,
                "similar_attempt_split": False,
                "exact_attempt_tie": False,
            },
            {
                "season": 2025,
                "week": 3,
                "game_id": "2025_03_DEN_LV",
                "defense": "DEN",
                "passing_yards_allowed": 500,
                "low_attempt_primary_qb": True,
                "similar_attempt_split": False,
                "exact_attempt_tie": False,
            },
            {
                "season": 2026,
                "week": 3,
                "game_id": "2026_03_DEN_LAC",
                "defense": "DEN",
                "passing_yards_allowed": 300,
                "low_attempt_primary_qb": False,
                "similar_attempt_split": True,
                "exact_attempt_tie": True,
            },
            {
                "season": 2026,
                "week": 2,
                "game_id": "2026_02_LV_DEN",
                "defense": "LV",
                "passing_yards_allowed": 260,
                "low_attempt_primary_qb": False,
                "similar_attempt_split": False,
                "exact_attempt_tie": False,
            },
            {
                "season": 2026,
                "week": 3,
                "game_id": "2026_03_LV_KC",
                "defense": "LV",
                "passing_yards_allowed": 260,
                "low_attempt_primary_qb": True,
                "similar_attempt_split": True,
                "exact_attempt_tie": False,
            },
            {
                "season": 2026,
                "week": 2,
                "game_id": "2026_02_DEN_LV",
                "defense": "DEN",
                "passing_yards_allowed": 200,
                "low_attempt_primary_qb": True,
                "similar_attempt_split": False,
                "exact_attempt_tie": False,
            },
            {
                "season": 2026,
                "week": 4,
                "game_id": "2026_04_DEN_BUF",
                "defense": "DEN",
                "passing_yards_allowed": 400,
                "low_attempt_primary_qb": False,
                "similar_attempt_split": False,
                "exact_attempt_tie": False,
            },
            {
                "season": 2026,
                "week": 4,
                "game_id": "2026_04_LV_MIA",
                "defense": "LV",
                "passing_yards_allowed": 260,
                "low_attempt_primary_qb": False,
                "similar_attempt_split": False,
                "exact_attempt_tie": False,
            },
            {
                "season": 2026,
                "week": 3,
                "game_id": "2026_03_CHI_MIN",
                "defense": "CHI",
                "passing_yards_allowed": 150,
                "low_attempt_primary_qb": False,
                "similar_attempt_split": False,
                "exact_attempt_tie": False,
            },
            {
                "season": 2026,
                "week": 2,
                "game_id": "2026_02_CHI_GB",
                "defense": "CHI",
                "passing_yards_allowed": 170,
                "low_attempt_primary_qb": False,
                "similar_attempt_split": False,
                "exact_attempt_tie": False,
            },
        ]
    )


def get_defense(result: pd.DataFrame, defense: str) -> pd.Series:
    return result.loc[result["defense"] == defense].iloc[0]


def test_build_defense_matchup_metrics_excludes_report_week_records():
    result = build_defense_matchup_metrics(make_defense_logs(), 2026, 4)

    defenses = result["defense"].tolist()
    den = get_defense(result, "DEN")

    assert "BUF" not in defenses
    assert den["defense_season_avg_allowed"] == 200


def test_build_defense_matchup_metrics_excludes_future_records():
    result = build_defense_matchup_metrics(make_defense_logs(), 2026, 4)

    lv = get_defense(result, "LV")

    assert lv["defense_season_avg_allowed"] == 260
    assert lv["defense_season_games"] == 2


def test_build_defense_matchup_metrics_excludes_other_seasons():
    result = build_defense_matchup_metrics(make_defense_logs(), 2026, 4)

    den = get_defense(result, "DEN")

    assert den["defense_season_games"] == 3
    assert den["flagged_games"] == 2


def test_build_defense_matchup_metrics_calculates_season_average():
    result = build_defense_matchup_metrics(make_defense_logs(), 2026, 5)

    den = get_defense(result, "DEN")

    assert den["defense_season_avg_allowed"] == 250


def test_build_defense_matchup_metrics_uses_three_most_recent_prior_games():
    result = build_defense_matchup_metrics(make_defense_logs(), 2026, 5)

    den = get_defense(result, "DEN")

    assert den["defense_last3_avg_allowed"] == 300
    assert den["defense_last3_games"] == 3


def test_build_defense_matchup_metrics_uses_all_when_fewer_than_three_games():
    result = build_defense_matchup_metrics(make_defense_logs(), 2026, 5)

    chi = get_defense(result, "CHI")

    assert chi["defense_last3_avg_allowed"] == 160
    assert chi["defense_last3_games"] == 2


def test_build_defense_matchup_metrics_counts_games_and_last3_samples():
    result = build_defense_matchup_metrics(make_defense_logs(), 2026, 5)

    den = get_defense(result, "DEN")
    lv = get_defense(result, "LV")

    assert den["defense_season_games"] == 4
    assert den["defense_last3_games"] == 3
    assert lv["defense_season_games"] == 3
    assert lv["defense_last3_games"] == 3


def test_build_defense_matchup_metrics_counts_flagged_games_once():
    result = build_defense_matchup_metrics(make_defense_logs(), 2026, 5)

    den = get_defense(result, "DEN")
    lv = get_defense(result, "LV")

    assert den["flagged_games"] == 2
    assert lv["flagged_games"] == 1


def test_build_defense_matchup_metrics_rank_1_is_most_yards_allowed():
    result = build_defense_matchup_metrics(make_defense_logs(), 2026, 5)

    top = result.iloc[0]

    assert top["defense"] == "LV"
    assert top["matchup_rank"] == 1


def test_build_defense_matchup_metrics_ties_receive_same_min_rank():
    data = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_AAA",
                "defense": "AAA",
                "passing_yards_allowed": 300,
                "low_attempt_primary_qb": False,
                "similar_attempt_split": False,
                "exact_attempt_tie": False,
            },
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_BBB",
                "defense": "BBB",
                "passing_yards_allowed": 300,
                "low_attempt_primary_qb": False,
                "similar_attempt_split": False,
                "exact_attempt_tie": False,
            },
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_CCC",
                "defense": "CCC",
                "passing_yards_allowed": 200,
                "low_attempt_primary_qb": False,
                "similar_attempt_split": False,
                "exact_attempt_tie": False,
            },
        ]
    )

    result = build_defense_matchup_metrics(data, 2026, 2)

    assert get_defense(result, "AAA")["matchup_rank"] == 1
    assert get_defense(result, "BBB")["matchup_rank"] == 1
    assert get_defense(result, "CCC")["matchup_rank"] == 3


def test_build_defense_matchup_metrics_week_1_returns_empty_schema():
    result = build_defense_matchup_metrics(make_defense_logs(), 2026, 1)

    assert result.empty
    assert result.columns.tolist() == OUTPUT_COLUMNS


def test_build_defense_matchup_metrics_validates_missing_columns():
    data = make_defense_logs().drop(
        columns=["passing_yards_allowed", "exact_attempt_tie"]
    )

    with pytest.raises(ValueError) as error:
        build_defense_matchup_metrics(data, 2026, 5)

    message = str(error.value)

    assert "Defense matchup data is missing required columns" in message
    assert "passing_yards_allowed" in message
    assert "exact_attempt_tie" in message


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
def test_build_defense_matchup_metrics_validates_report_values(
    report_season,
    report_week,
    expected_message,
):
    with pytest.raises(ValueError, match=expected_message):
        build_defense_matchup_metrics(
            make_defense_logs(),
            report_season,
            report_week,
        )


def test_build_defense_matchup_metrics_is_chronological_with_unsorted_input():
    result = build_defense_matchup_metrics(make_defense_logs(), 2026, 5)

    den = get_defense(result, "DEN")

    assert den["defense_last3_avg_allowed"] == 300


def test_build_defense_matchup_metrics_does_not_mutate_input():
    data = make_defense_logs()
    original = data.copy(deep=True)

    build_defense_matchup_metrics(data, 2026, 5)

    pd.testing.assert_frame_equal(data, original)


def test_build_defense_matchup_metrics_sorts_by_rank_then_defense():
    result = build_defense_matchup_metrics(make_defense_logs(), 2026, 5)

    assert result["defense"].tolist() == ["LV", "DEN", "CHI"]
