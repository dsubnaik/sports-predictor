import numpy as np
import pandas as pd
import pytest

from football.data.build_running_back_dataset import OUTPUT_COLUMNS
from football.features.running_back_usage import (
    LOW_VOLUME_RB_OPPORTUNITY_THRESHOLD,
    RUNNING_BACK_USAGE_COLUMNS,
    annotate_running_back_usage,
)


def make_running_back_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            make_row(
                season=2026,
                week=1,
                game_id="2026_01_KC_LAC",
                player_id="rb_lac_zero",
                player_name="Los Angeles Zero",
                team="LAC",
                opponent="KC",
                rushing_attempts=0,
                targets=0,
            ),
            make_row(
                season=2026,
                week=2,
                game_id="2026_02_BUF_MIA",
                player_id="rb_mia_receiver",
                player_name="Miami Receiver",
                team="MIA",
                opponent="BUF",
                rushing_attempts=0,
                targets=5,
            ),
            make_row(
                season=2026,
                week=1,
                game_id="2026_01_KC_LAC",
                player_id="rb_kc_lead",
                player_name="Kansas City Lead",
                team="KC",
                opponent="LAC",
                rushing_attempts=12,
                targets=3,
            ),
            make_row(
                season=2026,
                week=1,
                game_id="2026_01_KC_LAC",
                player_id="rb_kc_backup",
                player_name="Kansas City Backup",
                team="KC",
                opponent="LAC",
                rushing_attempts=4,
                targets=1,
            ),
            make_row(
                season=2026,
                week=2,
                game_id="2026_02_BUF_MIA",
                player_id="rb_buf_low",
                player_name="Buffalo Low",
                team="BUF",
                opponent="MIA",
                rushing_attempts=2,
                targets=1,
            ),
            make_row(
                season=2026,
                week=3,
                game_id="2026_03_BAL_CIN",
                player_id="rb_bal_a",
                player_name="Shared Name",
                team="BAL",
                opponent="CIN",
                rushing_attempts=6,
                targets=2,
            ),
            make_row(
                season=2026,
                week=3,
                game_id="2026_03_BAL_CIN",
                player_id="rb_bal_b",
                player_name="Shared Name",
                team="BAL",
                opponent="CIN",
                rushing_attempts=6,
                targets=2,
            ),
        ],
        columns=OUTPUT_COLUMNS,
    )


def make_row(
    *,
    season: int = 2026,
    week: int = 1,
    game_id: str = "2026_01_AAA_BBB",
    player_id: str,
    player_name: str,
    team: str = "AAA",
    opponent: str = "BBB",
    rushing_attempts: int = 0,
    targets: int = 0,
) -> dict[str, object]:
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
        "rushing_yards": rushing_attempts * 4,
        "rushing_touchdowns": 0,
        "receptions": min(targets, 2),
        "targets": targets,
        "receiving_yards": targets * 7,
        "receiving_touchdowns": 0,
    }


def get_player(result: pd.DataFrame, player_id: str) -> pd.Series:
    return result.loc[result["player_id"] == player_id].iloc[0]


def test_annotate_running_back_usage_single_rb_backfield_gets_totals_shares_and_flags():
    data = pd.DataFrame(
        [
            make_row(
                player_id="rb_only",
                player_name="Only RB",
                rushing_attempts=10,
                targets=4,
            )
        ],
        columns=OUTPUT_COLUMNS,
    )

    result = annotate_running_back_usage(data)
    row = get_player(result, "rb_only")

    assert row["team_rb_rushing_attempts"] == 10
    assert row["team_rb_targets"] == 4
    assert row["team_rb_opportunities"] == 14
    assert row["opportunities"] == 14
    assert row["carry_share"] == 1.0
    assert row["target_share"] == 1.0
    assert row["opportunity_share"] == 1.0
    assert row["lead_team_rusher"]
    assert row["lead_targeted_rb"]
    assert row["backfield_opportunity_leader"]
    assert not row["tied_backfield_opportunity_leader"]
    assert not row["low_volume_rb"]
    assert not row["shared_backfield"]


def test_annotate_running_back_usage_preserves_multiple_rbs_from_same_team_game():
    result = annotate_running_back_usage(make_running_back_rows())

    player_ids = result.loc[
        (result["game_id"] == "2026_01_KC_LAC") & (result["team"] == "KC"),
        "player_id",
    ].tolist()

    assert player_ids == ["rb_kc_backup", "rb_kc_lead"]


def test_annotate_running_back_usage_team_rushing_totals_use_all_team_rbs():
    result = annotate_running_back_usage(make_running_back_rows())

    lead = get_player(result, "rb_kc_lead")
    backup = get_player(result, "rb_kc_backup")

    assert lead["team_rb_rushing_attempts"] == 16
    assert backup["team_rb_rushing_attempts"] == 16


def test_annotate_running_back_usage_team_target_totals_use_all_team_rbs():
    result = annotate_running_back_usage(make_running_back_rows())

    lead = get_player(result, "rb_kc_lead")
    backup = get_player(result, "rb_kc_backup")

    assert lead["team_rb_targets"] == 4
    assert backup["team_rb_targets"] == 4


def test_annotate_running_back_usage_opportunities_equal_carries_plus_targets():
    result = annotate_running_back_usage(make_running_back_rows())

    row = get_player(result, "rb_kc_lead")

    assert row["opportunities"] == 15


def test_annotate_running_back_usage_calculates_shares():
    result = annotate_running_back_usage(make_running_back_rows())

    lead = get_player(result, "rb_kc_lead")
    backup = get_player(result, "rb_kc_backup")

    assert lead["carry_share"] == 12 / 16
    assert lead["target_share"] == 3 / 4
    assert lead["opportunity_share"] == 15 / 20
    assert backup["carry_share"] == 4 / 16
    assert backup["target_share"] == 1 / 4
    assert backup["opportunity_share"] == 5 / 20


def test_annotate_running_back_usage_calculates_different_teams_independently():
    result = annotate_running_back_usage(make_running_back_rows())

    kc = get_player(result, "rb_kc_lead")
    lac = get_player(result, "rb_lac_zero")

    assert kc["team_rb_opportunities"] == 20
    assert lac["team_rb_opportunities"] == 0


def test_annotate_running_back_usage_calculates_different_games_independently():
    result = annotate_running_back_usage(make_running_back_rows())

    kc = get_player(result, "rb_kc_lead")
    mia = get_player(result, "rb_mia_receiver")

    assert kc["team_rb_targets"] == 4
    assert mia["team_rb_targets"] == 5


def test_annotate_running_back_usage_low_volume_rb_remains_in_output():
    result = annotate_running_back_usage(make_running_back_rows())

    row = get_player(result, "rb_buf_low")

    assert row["opportunities"] == LOW_VOLUME_RB_OPPORTUNITY_THRESHOLD
    assert row["low_volume_rb"]


def test_annotate_running_back_usage_rb_above_low_volume_threshold_is_not_flagged():
    result = annotate_running_back_usage(make_running_back_rows())

    row = get_player(result, "rb_kc_backup")

    assert row["opportunities"] > LOW_VOLUME_RB_OPPORTUNITY_THRESHOLD
    assert not row["low_volume_rb"]


def test_annotate_running_back_usage_preserves_zero_carry_rb_with_receiving_usage():
    result = annotate_running_back_usage(make_running_back_rows())

    row = get_player(result, "rb_mia_receiver")

    assert row["rushing_attempts"] == 0
    assert row["targets"] == 5
    assert row["opportunities"] == 5
    assert row["carry_share"] == 0.0
    assert row["target_share"] == 1.0
    assert row["opportunity_share"] == 1.0


def test_annotate_running_back_usage_preserves_zero_usage_rb():
    result = annotate_running_back_usage(make_running_back_rows())

    row = get_player(result, "rb_lac_zero")

    assert row["rushing_attempts"] == 0
    assert row["targets"] == 0
    assert row["opportunities"] == 0


def test_annotate_running_back_usage_zero_denominator_shares_are_finite_zeroes():
    result = annotate_running_back_usage(make_running_back_rows())

    row = get_player(result, "rb_lac_zero")

    assert row["carry_share"] == 0.0
    assert row["target_share"] == 0.0
    assert row["opportunity_share"] == 0.0
    assert np.isfinite(
        result[["carry_share", "target_share", "opportunity_share"]].to_numpy()
    ).all()


def test_annotate_running_back_usage_flags_clear_carry_leaders():
    result = annotate_running_back_usage(make_running_back_rows())

    assert get_player(result, "rb_kc_lead")["lead_team_rusher"]
    assert not get_player(result, "rb_kc_backup")["lead_team_rusher"]


def test_annotate_running_back_usage_flags_clear_target_leaders():
    result = annotate_running_back_usage(make_running_back_rows())

    assert get_player(result, "rb_kc_lead")["lead_targeted_rb"]
    assert not get_player(result, "rb_kc_backup")["lead_targeted_rb"]


def test_annotate_running_back_usage_flags_clear_opportunity_leaders():
    result = annotate_running_back_usage(make_running_back_rows())

    assert get_player(result, "rb_kc_lead")["backfield_opportunity_leader"]
    assert not get_player(result, "rb_kc_backup")["backfield_opportunity_leader"]


def test_annotate_running_back_usage_marks_positive_ties_as_leaders():
    result = annotate_running_back_usage(make_running_back_rows())

    first = get_player(result, "rb_bal_a")
    second = get_player(result, "rb_bal_b")

    assert first["lead_team_rusher"]
    assert second["lead_team_rusher"]
    assert first["lead_targeted_rb"]
    assert second["lead_targeted_rb"]
    assert first["backfield_opportunity_leader"]
    assert second["backfield_opportunity_leader"]
    assert first["tied_backfield_opportunity_leader"]
    assert second["tied_backfield_opportunity_leader"]


def test_annotate_running_back_usage_all_zero_backfield_has_no_leaders():
    data = pd.DataFrame(
        [
            make_row(player_id="rb_zero_a", player_name="Zero A"),
            make_row(player_id="rb_zero_b", player_name="Zero B"),
        ],
        columns=OUTPUT_COLUMNS,
    )

    result = annotate_running_back_usage(data)

    for player_id in ["rb_zero_a", "rb_zero_b"]:
        row = get_player(result, player_id)
        assert not row["lead_team_rusher"]
        assert not row["lead_targeted_rb"]
        assert not row["backfield_opportunity_leader"]
        assert not row["tied_backfield_opportunity_leader"]
        assert not row["shared_backfield"]


def test_annotate_running_back_usage_shared_backfield_means_two_positive_opportunity_rbs():
    result = annotate_running_back_usage(make_running_back_rows())

    assert get_player(result, "rb_kc_lead")["shared_backfield"]
    assert get_player(result, "rb_kc_backup")["shared_backfield"]
    assert not get_player(result, "rb_lac_zero")["shared_backfield"]
    assert not get_player(result, "rb_mia_receiver")["shared_backfield"]


def test_annotate_running_back_usage_is_deterministic_with_unsorted_input():
    data = make_running_back_rows().sample(frac=1, random_state=42)

    result = annotate_running_back_usage(data)

    sort_keys = ["season", "week", "game_id", "team", "player_id"]

    assert result[sort_keys].values.tolist() == [
        [2026, 1, "2026_01_KC_LAC", "KC", "rb_kc_backup"],
        [2026, 1, "2026_01_KC_LAC", "KC", "rb_kc_lead"],
        [2026, 1, "2026_01_KC_LAC", "LAC", "rb_lac_zero"],
        [2026, 2, "2026_02_BUF_MIA", "BUF", "rb_buf_low"],
        [2026, 2, "2026_02_BUF_MIA", "MIA", "rb_mia_receiver"],
        [2026, 3, "2026_03_BAL_CIN", "BAL", "rb_bal_a"],
        [2026, 3, "2026_03_BAL_CIN", "BAL", "rb_bal_b"],
    ]


def test_annotate_running_back_usage_does_not_mutate_input():
    data = make_running_back_rows()
    original = data.copy(deep=True)

    annotate_running_back_usage(data)

    pd.testing.assert_frame_equal(data, original)


def test_annotate_running_back_usage_valid_empty_input_returns_complete_schema():
    data = pd.DataFrame(columns=OUTPUT_COLUMNS)

    result = annotate_running_back_usage(data)

    assert result.empty
    assert result.columns.tolist() == RUNNING_BACK_USAGE_COLUMNS


def test_annotate_running_back_usage_validates_missing_columns():
    data = make_running_back_rows().drop(columns=["rushing_attempts", "team"])

    with pytest.raises(ValueError) as error:
        annotate_running_back_usage(data)

    message = str(error.value)

    assert "Running back data is missing required columns" in message
    assert "rushing_attempts" in message
    assert "team" in message


def test_annotate_running_back_usage_rejects_missing_usage_values():
    data = make_running_back_rows()
    data.loc[0, "targets"] = np.nan

    with pytest.raises(ValueError) as error:
        annotate_running_back_usage(data)

    message = str(error.value)

    assert "Running back usage columns cannot contain missing values" in message
    assert "targets" in message


def test_annotate_running_back_usage_ignores_identical_duplicates():
    row = make_row(
        player_id="rb_only",
        player_name="Only RB",
        rushing_attempts=5,
        targets=1,
    )
    data = pd.DataFrame([row, row], columns=OUTPUT_COLUMNS)

    result = annotate_running_back_usage(data)

    assert result["player_id"].tolist() == ["rb_only"]


def test_annotate_running_back_usage_rejects_conflicting_duplicate_player_games():
    row = make_row(
        player_id="rb_only",
        player_name="Only RB",
        rushing_attempts=5,
        targets=1,
    )
    conflict = {**row, "targets": 2}
    data = pd.DataFrame([row, conflict], columns=OUTPUT_COLUMNS)

    with pytest.raises(ValueError) as error:
        annotate_running_back_usage(data)

    message = str(error.value)

    assert "Conflicting running-back game records found" in message
    assert "2026_01_AAA_BBB" in message
    assert "rb_only" in message


def test_annotate_running_back_usage_keeps_same_name_players_with_different_ids():
    result = annotate_running_back_usage(make_running_back_rows())

    same_name_rows = result.loc[result["player_name"] == "Shared Name"]

    assert same_name_rows["player_id"].tolist() == ["rb_bal_a", "rb_bal_b"]


def test_annotate_running_back_usage_operates_only_on_normalized_rows():
    data = pd.DataFrame(
        [
            make_row(
                player_id="rb_zero",
                player_name="Zero RB",
                rushing_attempts=0,
                targets=0,
            ),
            make_row(
                player_id="rb_low",
                player_name="Low RB",
                rushing_attempts=1,
                targets=1,
            ),
        ],
        columns=OUTPUT_COLUMNS,
    )

    result = annotate_running_back_usage(data)

    assert result["player_id"].tolist() == ["rb_low", "rb_zero"]
    assert len(result) == len(data)
    assert result.columns.tolist() == RUNNING_BACK_USAGE_COLUMNS


def test_annotate_running_back_usage_returns_boolean_flag_columns():
    result = annotate_running_back_usage(make_running_back_rows())
    flag_columns = [
        "lead_team_rusher",
        "lead_targeted_rb",
        "backfield_opportunity_leader",
        "tied_backfield_opportunity_leader",
        "low_volume_rb",
        "shared_backfield",
    ]

    assert all(result[column].dtype == bool for column in flag_columns)


def test_annotate_running_back_usage_preserves_normalized_columns_first():
    result = annotate_running_back_usage(make_running_back_rows())

    assert result.columns.tolist()[: len(OUTPUT_COLUMNS)] == OUTPUT_COLUMNS
