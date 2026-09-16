"""Deliberately load the external inputs required by settlement composition.

This module only retrieves and normalizes source inputs. It neither opens
SQLite nor maps completed events or settles decisions. A frozen return object
does not deeply freeze its contained list or pandas DataFrames.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import pandas as pd

from football.data.build_running_back_dataset import build_running_back_dataset
from football.data.build_schedule_dataset import normalize_schedule_dataset
from football.data.fetch_nflverse import (
    load_player_game_stats,
    load_schedules,
    normalize_quarterback_game_stats,
)
from football.odds.scores import fetch_nfl_scores


class LiveSettlementInputsValidationError(ValueError):
    """Raised when local input or source provenance cannot be safely used."""


@dataclass(frozen=True)
class LiveSettlementInputs:
    """One requested season's scores payload and normalized settlement inputs.

    The dataclass is frozen, but its ``scores_payload`` list and DataFrames are
    still mutable containers owned by the caller after return.
    """

    season: int
    scores_payload: list[dict[str, object]]
    normalized_schedule: pd.DataFrame
    quarterback_games: pd.DataFrame
    running_back_games: pd.DataFrame


def load_live_settlement_inputs(
    season: int,
    *,
    scores_fetcher: Callable[..., Any] | None = None,
    schedules_loader: Callable[..., pd.DataFrame] | None = None,
    player_stats_loader: Callable[..., pd.DataFrame] | None = None,
    days_from: int = 3,
) -> LiveSettlementInputs:
    """Deliberately retrieve and normalize inputs for one NFL season.

    Source calls occur once each in this order: schedules, player statistics,
    schedule normalization, quarterback normalization, running-back
    normalization, then scores. Scores are intentionally fetched last so a
    failed nflverse source or builder cannot consume Odds API credits.
    """

    requested_season = _season(season)
    requested_days_from = _days_from(days_from)
    schedule_source = (schedules_loader or load_schedules)([requested_season])
    player_stats_source = (player_stats_loader or load_player_game_stats)([requested_season])

    normalized_schedule = normalize_schedule_dataset(schedule_source)
    quarterback_games = normalize_quarterback_game_stats(player_stats_source)
    running_back_games = build_running_back_dataset(player_stats_source)
    _validate_provenance(normalized_schedule, requested_season, "normalized_schedule")
    _validate_provenance(quarterback_games, requested_season, "quarterback_games")
    _validate_provenance(running_back_games, requested_season, "running_back_games")

    payload = (scores_fetcher or fetch_nfl_scores)(days_from=requested_days_from)
    scores_payload = _scores_payload(payload)
    return LiveSettlementInputs(
        season=requested_season,
        scores_payload=scores_payload,
        normalized_schedule=normalized_schedule,
        quarterback_games=quarterback_games,
        running_back_games=running_back_games,
    )


def _season(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise LiveSettlementInputsValidationError("season must be a positive integer")
    return int(value)


def _days_from(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value not in {1, 2, 3}:
        raise LiveSettlementInputsValidationError("days_from must be an integer from 1 through 3")
    return int(value)


def _validate_provenance(
    data: object,
    requested_season: int,
    label: str,
) -> None:
    if not isinstance(data, pd.DataFrame) or "season" not in data.columns:
        raise LiveSettlementInputsValidationError(
            f"{label} must be a pandas DataFrame with a season column"
        )
    if data.empty:
        return
    matching_season = data["season"].eq(requested_season)
    if bool(matching_season.all()):
        return
    unexpected = sorted({str(value) for value in data.loc[~matching_season, "season"]})
    raise LiveSettlementInputsValidationError(
        f"{label} contains seasons other than requested season: {unexpected}"
    )


def _scores_payload(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or not all(isinstance(event, Mapping) for event in value):
        raise LiveSettlementInputsValidationError("scores_fetcher must return a list of event objects")
    return [dict(event) for event in value]
