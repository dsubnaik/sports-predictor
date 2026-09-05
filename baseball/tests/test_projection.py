import pandas as pd

from baseball.model import projection


def test_build_pitcher_projection_returns_none_without_rolling_feature_rows(monkeypatch):
    monkeypatch.setattr(projection, "get_player_id", lambda pitcher_name: 123)
    monkeypatch.setattr(
        projection,
        "fetch_pitcher_statcast",
        lambda player_id, start_date, end_date: pd.DataFrame({"raw": [1]}),
    )
    monkeypatch.setattr(
        projection,
        "aggregate_to_starts",
        lambda pitch_data: pd.DataFrame({"start": [1]}),
    )
    monkeypatch.setattr(
        projection,
        "rolling_features",
        lambda starts: pd.DataFrame(
            {
                "rolling_k": [pd.NA],
                "rolling_swstr": [pd.NA],
                "rolling_velocity": [pd.NA],
                "rolling_pitches": [pd.NA],
            }
        ),
    )

    result = projection.build_pitcher_projection("Test Pitcher")

    assert result is None


def test_build_pitcher_projection_wraps_missing_player(monkeypatch):
    def raise_missing_player(pitcher_name):
        raise ValueError("No MLB player was found")

    monkeypatch.setattr(projection, "get_player_id", raise_missing_player)

    try:
        projection.build_pitcher_projection("Unknown Pitcher")
    except projection.PitcherProjectionUnavailable as error:
        assert "No MLB player was found" in str(error)
    else:
        raise AssertionError("Expected PitcherProjectionUnavailable")


def test_build_pitcher_projection_returns_none_for_empty_statcast(monkeypatch):
    def fail_aggregate(pitch_data):
        raise AssertionError("Empty Statcast data should not be aggregated")

    monkeypatch.setattr(projection, "get_player_id", lambda pitcher_name: 123)
    monkeypatch.setattr(
        projection,
        "fetch_pitcher_statcast",
        lambda player_id, start_date, end_date: pd.DataFrame(),
    )
    monkeypatch.setattr(projection, "aggregate_to_starts", fail_aggregate)

    result = projection.build_pitcher_projection("Test Pitcher")

    assert result is None


def test_build_pitcher_projection_predicts_from_latest_feature_row(monkeypatch):
    prediction_inputs = {}
    monkeypatch.setattr(projection, "get_player_id", lambda pitcher_name: 123)
    monkeypatch.setattr(
        projection,
        "fetch_pitcher_statcast",
        lambda player_id, start_date, end_date: pd.DataFrame({"raw": [1]}),
    )
    monkeypatch.setattr(
        projection,
        "aggregate_to_starts",
        lambda pitch_data: pd.DataFrame({"start": [1]}),
    )
    monkeypatch.setattr(
        projection,
        "rolling_features",
        lambda starts: pd.DataFrame(
            [
                {
                    "rolling_k": 4.0,
                    "rolling_swstr": 0.10,
                    "rolling_velocity": 93.0,
                    "rolling_pitches": 82.0,
                },
                {
                    "rolling_k": 6.0,
                    "rolling_swstr": 0.12,
                    "rolling_velocity": 95.0,
                    "rolling_pitches": 91.0,
                },
            ]
        ),
    )

    def fake_predict_strikeouts(**kwargs):
        prediction_inputs.update(kwargs)
        return 6.4

    monkeypatch.setattr(projection, "predict_strikeouts", fake_predict_strikeouts)

    result = projection.build_pitcher_projection("Test Pitcher")

    assert result == 6.4
    assert prediction_inputs == {
        "rolling_k": 6.0,
        "rolling_swstr": 0.12,
        "rolling_velocity": 95.0,
        "rolling_pitches": 91.0,
    }
