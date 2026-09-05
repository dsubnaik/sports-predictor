"""Build saved-model pitcher strikeout projections from Statcast history."""

from __future__ import annotations

from baseball.data.fetch_statcast import (
    aggregate_to_starts,
    fetch_pitcher_statcast,
    get_player_id,
)
from baseball.features.engineer import rolling_features
from baseball.model.predict import predict_strikeouts


DEFAULT_PROJECTION_START_DATE = "2025-03-01"
DEFAULT_PROJECTION_END_DATE = "2025-12-31"


class PitcherProjectionUnavailable(ValueError):
    """Raised when a pitcher has no usable projection input data."""


def build_pitcher_projection(
    pitcher_name: str,
    start_date: str = DEFAULT_PROJECTION_START_DATE,
    end_date: str = DEFAULT_PROJECTION_END_DATE,
) -> float | None:
    """Return the latest saved-model strikeout projection for one pitcher."""

    try:
        player_id = get_player_id(pitcher_name)
    except ValueError as error:
        raise PitcherProjectionUnavailable(str(error)) from error

    pitch_data = fetch_pitcher_statcast(player_id, start_date, end_date)
    if pitch_data.empty:
        return None

    starts = aggregate_to_starts(pitch_data)
    features = rolling_features(starts).dropna()

    if features.empty:
        return None

    last = features.iloc[-1]
    return predict_strikeouts(
        rolling_k=last["rolling_k"],
        rolling_swstr=last["rolling_swstr"],
        rolling_velocity=last["rolling_velocity"],
        rolling_pitches=last["rolling_pitches"],
    )
