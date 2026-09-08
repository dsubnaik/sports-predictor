import numpy as np
import pandas as pd
import pytest

from football.features.defense_rb_game_logs import (
    DEFENSE_RB_GAME_KEYS,
    DEFENSE_RB_GAME_LOG_COLUMNS,
    build_defense_rb_game_logs,
)
from football.features.running_back_usage import RUNNING_BACK_USAGE_COLUMNS


def make_row(
    *,
    season: int = 2026,
    week: int = 1,
    game_id: str = "2026_01_KC_LAC",
    player_id: str = "rb_kc_lead",
    player_name: str = "Kansas City Lead",
    team: str = "KC",
    opponent: str = "LAC",
    rushing_attempts: float = 12,
    rushing_yards: float = 60,
    rushing_touchdowns: float = 1,
    receptions: float = 3,
    targets: float = 4,
    receiving_yards: float = 24,
    receiving_touchdowns: float = 0,
    team_rb_rushing_attempts: float | None = None,
    team_rb_targets: float | None = None,
    team_rb_opportunities: float | None = None,
) -> dict[str, object]:
    opportunities = rushing_attempts + targets
    return {
        "season": season,
        "week": week,
        "game_id": game_id,
        "player_id": player_id,
        "player_name": player_name,
        "team": team,
        "opponent": opponent,
        "position": "RB",
        "rushing_attempts": rushing_attempts,
        "rushing_yards": rushing_yards,
        "rushing_touchdowns": rushing_touchdowns,
        "receptions": receptions,
        "targets": targets,
        "receiving_yards": receiving_yards,
        "receiving_touchdowns": receiving_touchdowns,
        "opportunities": opportunities,
        "team_rb_rushing_attempts": (
            rushing_attempts
            if team_rb_rushing_attempts is None
            else team_rb_rushing_attempts
        ),
        "team_rb_targets": (
            targets if team_rb_targets is None else team_rb_targets
        ),
        "team_rb_opportunities": (
            opportunities
            if team_rb_opportunities is None
            else team_rb_opportunities
        ),
        "carry_share": 0.5,
        "target_share": 0.5,
        "opportunity_share": 0.5,
        "lead_team_rusher": True,
        "lead_targeted_rb": True,
        "backfield_opportunity_leader": True,
        "tied_backfield_opportunity_leader": False,
        "low_volume_rb": opportunities <= 3,
        "shared_backfield": False,
    }


def make_data(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=RUNNING_BACK_USAGE_COLUMNS)


def test_single_rb_maps_context_and_production_to_defense_game():
    result = build_defense_rb_game_logs(make_data(make_row()))

    assert result.columns.tolist() == DEFENSE_RB_GAME_LOG_COLUMNS
    assert result.iloc[0].to_dict() == {
        "season": 2026,
        "week": 1,
        "game_id": "2026_01_KC_LAC",
        "defense": "LAC",
        "offense_team": "KC",
        "rb_rushing_attempts_allowed": 12,
        "rb_rushing_yards_allowed": 60,
        "rb_rushing_touchdowns_allowed": 1,
        "rb_receptions_allowed": 3,
        "rb_targets_allowed": 4,
        "rb_receiving_yards_allowed": 24,
        "rb_receiving_touchdowns_allowed": 0,
        "rb_opportunities_allowed": 16,
        "rb_players_used": 1,
    }


def test_multiple_rbs_aggregate_all_individual_production_once():
    lead = make_row(
        player_name="Shared Name",
        rushing_attempts=12,
        rushing_yards=60,
        rushing_touchdowns=1,
        receptions=3,
        targets=4,
        receiving_yards=24,
        receiving_touchdowns=0,
        team_rb_rushing_attempts=999,
        team_rb_targets=999,
        team_rb_opportunities=1998,
    )
    backup = make_row(
        player_id="rb_kc_backup",
        player_name="Shared Name",
        rushing_attempts=3,
        rushing_yards=11,
        rushing_touchdowns=1,
        receptions=1,
        targets=2,
        receiving_yards=9,
        receiving_touchdowns=1,
        team_rb_rushing_attempts=999,
        team_rb_targets=999,
        team_rb_opportunities=1998,
    )

    result = build_defense_rb_game_logs(make_data(lead, backup))
    row = result.iloc[0]

    assert len(result) == 1
    assert row["rb_rushing_attempts_allowed"] == 15
    assert row["rb_rushing_yards_allowed"] == 71
    assert row["rb_rushing_touchdowns_allowed"] == 2
    assert row["rb_receptions_allowed"] == 4
    assert row["rb_targets_allowed"] == 6
    assert row["rb_receiving_yards_allowed"] == 33
    assert row["rb_receiving_touchdowns_allowed"] == 1
    assert row["rb_opportunities_allowed"] == 21
    assert row["rb_players_used"] == 2


def test_zero_carry_receiver_zero_usage_rb_and_backup_remain_represented():
    receiver = make_row(
        player_id="rb_receiver",
        rushing_attempts=0,
        rushing_yards=0,
        rushing_touchdowns=0,
        receptions=4,
        targets=5,
        receiving_yards=35,
    )
    zero_usage = make_row(
        player_id="rb_zero",
        rushing_attempts=0,
        rushing_yards=0,
        rushing_touchdowns=0,
        receptions=0,
        targets=0,
        receiving_yards=0,
    )
    backup = make_row(
        player_id="rb_backup",
        rushing_attempts=1,
        rushing_yards=2,
        rushing_touchdowns=0,
        receptions=0,
        targets=1,
        receiving_yards=0,
    )

    row = build_defense_rb_game_logs(
        make_data(receiver, zero_usage, backup)
    ).iloc[0]

    assert row["rb_rushing_attempts_allowed"] == 1
    assert row["rb_rushing_yards_allowed"] == 2
    assert row["rb_receptions_allowed"] == 4
    assert row["rb_targets_allowed"] == 6
    assert row["rb_receiving_yards_allowed"] == 35
    assert row["rb_opportunities_allowed"] == 7
    assert row["rb_players_used"] == 3


def test_negative_rushing_and_receiving_yardage_is_preserved():
    row = make_row(
        rushing_attempts=2,
        rushing_yards=-7,
        rushing_touchdowns=0,
        receptions=1,
        targets=2,
        receiving_yards=-3,
    )

    result = build_defense_rb_game_logs(make_data(row)).iloc[0]

    assert result["rb_rushing_yards_allowed"] == -7
    assert result["rb_receiving_yards_allowed"] == -3


def test_exact_duplicate_player_game_does_not_double_count():
    row = make_row()

    result = build_defense_rb_game_logs(make_data(row, row))

    assert result.iloc[0]["rb_rushing_attempts_allowed"] == 12
    assert result.iloc[0]["rb_players_used"] == 1


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("rushing_yards", 999),
        ("team", "NE"),
        ("team_rb_rushing_attempts", 999),
        ("shared_backfield", True),
    ],
)
def test_conflicting_duplicate_player_games_raise_before_aggregation(
    column,
    value,
):
    row = make_row()
    conflict = {**row, column: value}

    with pytest.raises(ValueError) as error:
        build_defense_rb_game_logs(make_data(row, conflict))

    message = str(error.value)
    assert "Conflicting running-back game records found for keys" in message
    assert "2026_01_KC_LAC" in message
    assert "rb_kc_lead" in message


@pytest.mark.parametrize(
    "column",
    [
        "rushing_attempts",
        "rushing_touchdowns",
        "receptions",
        "targets",
        "receiving_touchdowns",
        "opportunities",
    ],
)
def test_negative_count_or_volume_values_are_rejected(column):
    row = make_row()
    row[column] = -1

    with pytest.raises(ValueError, match="cannot contain negative values"):
        build_defense_rb_game_logs(make_data(row))


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("rushing_attempts", "invalid", "must be numeric"),
        ("rushing_yards", np.nan, "cannot contain missing values"),
        ("receiving_yards", np.inf, "must contain finite values"),
        ("receiving_yards", -np.inf, "must contain finite values"),
        ("targets", True, "must be numeric"),
    ],
)
def test_invalid_required_production_values_are_rejected(
    column,
    value,
    message,
):
    row = make_row()
    row[column] = value

    with pytest.raises(ValueError, match=message):
        build_defense_rb_game_logs(make_data(row))


def test_inconsistent_individual_opportunities_are_rejected():
    row = make_row()
    row["opportunities"] += 1

    with pytest.raises(ValueError, match="opportunities must equal"):
        build_defense_rb_game_logs(make_data(row))


def test_both_offenses_in_one_nfl_game_produce_separate_defense_rows():
    kc = make_row()
    lac = make_row(
        player_id="rb_lac",
        player_name="Los Angeles RB",
        team="LAC",
        opponent="KC",
        rushing_attempts=8,
        rushing_yards=30,
        rushing_touchdowns=0,
        receptions=2,
        targets=3,
        receiving_yards=12,
    )

    result = build_defense_rb_game_logs(make_data(lac, kc))

    assert result[["defense", "offense_team"]].to_dict(orient="records") == [
        {"defense": "KC", "offense_team": "LAC"},
        {"defense": "LAC", "offense_team": "KC"},
    ]


def test_games_weeks_seasons_and_historical_trade_teams_remain_separate():
    rows = [
        make_row(
            season=2025,
            week=1,
            game_id="2025_01_BUF_MIA",
            player_id="rb_traded",
            team="BUF",
            opponent="MIA",
        ),
        make_row(
            season=2026,
            week=1,
            game_id="2026_01_NYJ_MIA",
            player_id="rb_traded",
            team="NYJ",
            opponent="MIA",
        ),
        make_row(
            season=2026,
            week=2,
            game_id="2026_02_NE_MIA",
            player_id="rb_ne",
            team="NE",
            opponent="MIA",
        ),
    ]

    result = build_defense_rb_game_logs(make_data(*rows))

    assert result[["season", "week", "game_id", "offense_team"]].to_dict(
        orient="records"
    ) == [
        {
            "season": 2025,
            "week": 1,
            "game_id": "2025_01_BUF_MIA",
            "offense_team": "BUF",
        },
        {
            "season": 2026,
            "week": 1,
            "game_id": "2026_01_NYJ_MIA",
            "offense_team": "NYJ",
        },
        {
            "season": 2026,
            "week": 2,
            "game_id": "2026_02_NE_MIA",
            "offense_team": "NE",
        },
    ]


def test_input_order_does_not_change_stable_chronological_output():
    rows = [
        make_row(),
        make_row(
            week=2,
            game_id="2026_02_BUF_MIA",
            player_id="rb_buf",
            team="BUF",
            opponent="MIA",
        ),
        make_row(
            player_id="rb_lac",
            team="LAC",
            opponent="KC",
        ),
    ]
    data = make_data(*rows)
    expected = build_defense_rb_game_logs(data)
    shuffled = build_defense_rb_game_logs(
        data.sample(frac=1, random_state=42).reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(shuffled, expected)
    assert expected[DEFENSE_RB_GAME_KEYS].to_dict(orient="records") == [
        {
            "season": 2026,
            "week": 1,
            "game_id": "2026_01_KC_LAC",
            "defense": "KC",
        },
        {
            "season": 2026,
            "week": 1,
            "game_id": "2026_01_KC_LAC",
            "defense": "LAC",
        },
        {
            "season": 2026,
            "week": 2,
            "game_id": "2026_02_BUF_MIA",
            "defense": "MIA",
        },
    ]


def test_input_dataframe_is_not_mutated():
    data = make_data(make_row())
    original = data.copy(deep=True)

    build_defense_rb_game_logs(data)

    pd.testing.assert_frame_equal(data, original)


def test_missing_complete_annotated_schema_columns_raise_clear_error():
    data = make_data(make_row()).drop(
        columns=["rushing_yards", "shared_backfield"]
    )

    with pytest.raises(ValueError) as error:
        build_defense_rb_game_logs(data)

    message = str(error.value)
    assert "Annotated running back data is missing required columns" in message
    assert "rushing_yards" in message
    assert "shared_backfield" in message


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("player_id", None, "cannot contain missing values"),
        ("game_id", "   ", "cannot contain blank values"),
        ("season", True, "must be numeric"),
        ("week", 0, "must contain positive integers"),
        ("week", 1.5, "must contain positive integers"),
    ],
)
def test_invalid_grouping_identities_are_rejected(column, value, message):
    row = make_row()
    row[column] = value

    with pytest.raises(ValueError, match=message):
        build_defense_rb_game_logs(make_data(row))


@pytest.mark.parametrize("position", ["WR", None, pd.NA])
def test_invalid_rb_position_is_rejected_instead_of_aggregated(position):
    row = make_row()
    row["position"] = position

    with pytest.raises(ValueError, match="only position == 'RB'"):
        build_defense_rb_game_logs(make_data(row))


def test_conflicting_offenses_for_one_defense_game_are_rejected():
    first = make_row()
    conflict = make_row(
        player_id="rb_other",
        team="NE",
        opponent="LAC",
    )

    with pytest.raises(ValueError) as error:
        build_defense_rb_game_logs(make_data(first, conflict))

    message = str(error.value)
    assert "Conflicting defense-game records found for keys" in message
    assert "2026_01_KC_LAC" in message
    assert "LAC" in message


def test_inconsistent_opponents_for_one_offense_game_raise_clear_error():
    first = make_row()
    conflict = make_row(
        player_id="rb_kc_backup",
        opponent="BUF",
    )

    with pytest.raises(ValueError) as error:
        build_defense_rb_game_logs(make_data(first, conflict))

    message = str(error.value)
    assert "Conflicting offense-game records found for keys" in message
    assert "2026_01_KC_LAC" in message
    assert "'offense_team': 'KC'" in message


def test_inconsistent_opponents_are_rejected_before_partial_totals_return():
    first = make_row(
        rushing_attempts=12,
        rushing_yards=60,
    )
    conflict = make_row(
        player_id="rb_kc_backup",
        opponent="BUF",
        rushing_attempts=5,
        rushing_yards=25,
    )

    with pytest.raises(ValueError, match="Conflicting offense-game records"):
        build_defense_rb_game_logs(make_data(first, conflict))


def test_multiple_valid_games_for_same_offense_team_remain_separate():
    week_one = make_row()
    week_two = make_row(
        week=2,
        game_id="2026_02_BUF_KC",
        player_id="rb_kc_week_two",
        opponent="BUF",
        rushing_attempts=8,
        rushing_yards=44,
    )

    result = build_defense_rb_game_logs(make_data(week_two, week_one))

    assert result[["week", "defense", "offense_team"]].to_dict(
        orient="records"
    ) == [
        {"week": 1, "defense": "LAC", "offense_team": "KC"},
        {"week": 2, "defense": "BUF", "offense_team": "KC"},
    ]
    assert result["rb_rushing_yards_allowed"].tolist() == [60, 44]


def test_offense_game_conflict_error_is_deterministic_for_input_order():
    rows = [
        make_row(),
        make_row(player_id="rb_kc_backup", opponent="BUF"),
        make_row(
            week=2,
            game_id="2026_02_DEN_LV",
            player_id="rb_den_lead",
            team="DEN",
            opponent="LV",
        ),
        make_row(
            week=2,
            game_id="2026_02_DEN_LV",
            player_id="rb_den_backup",
            team="DEN",
            opponent="KC",
        ),
    ]
    forward = make_data(*rows)
    reversed_input = make_data(*reversed(rows))

    with pytest.raises(ValueError) as forward_error:
        build_defense_rb_game_logs(forward)
    with pytest.raises(ValueError) as reversed_error:
        build_defense_rb_game_logs(reversed_input)

    forward_message = str(forward_error.value)
    assert forward_message == str(reversed_error.value)
    assert forward_message.index("2026_01_KC_LAC") < forward_message.index(
        "2026_02_DEN_LV"
    )


def test_output_has_exactly_one_row_per_defense_game_identity():
    data = make_data(
        make_row(),
        make_row(player_id="rb_kc_backup"),
        make_row(player_id="rb_lac", team="LAC", opponent="KC"),
    )

    result = build_defense_rb_game_logs(data)

    assert not result.duplicated(subset=DEFENSE_RB_GAME_KEYS).any()
    assert len(result) == len(result.loc[:, DEFENSE_RB_GAME_KEYS].drop_duplicates())


def test_valid_empty_annotated_input_returns_complete_output_schema():
    data = pd.DataFrame(columns=RUNNING_BACK_USAGE_COLUMNS)

    result = build_defense_rb_game_logs(data)

    assert result.empty
    assert result.columns.tolist() == DEFENSE_RB_GAME_LOG_COLUMNS
