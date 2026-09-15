"""Purely map explicitly completed Odds API events to nflverse game IDs.

Only an Odds API event whose ``completed`` value is the boolean ``True`` is
eligible. Scores, dates, player data, and elapsed time never establish
finality here. The resulting IDs are intended for the existing decision-result
matcher and batch settlement caller; this module does not fetch or settle.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from football.odds.event_discovery import (
    match_nfl_events_to_schedule,
    normalize_nfl_events,
)


class CompletedGameMappingError(ValueError):
    """Base exception for completed-event mapping errors."""


class CompletedGameValidationError(CompletedGameMappingError):
    """Raised when completed-event inputs cannot be safely mapped."""


class CompletedGameConflictError(CompletedGameMappingError):
    """Raised when eligible provider identities conflict."""


@dataclass(frozen=True)
class CompletedGameDiagnostic:
    """A completed provider event that did not uniquely map to a game."""

    provider_event_id: str
    match_status: str
    diagnostic: str


@dataclass(frozen=True)
class CompletedGameReport:
    """Deterministic completed-game IDs and unresolved mapping diagnostics."""

    completed_game_ids: tuple[str, ...]
    diagnostics: tuple[CompletedGameDiagnostic, ...]


def identify_completed_nflverse_games(
    scores_payload: Sequence[Mapping[str, Any]],
    normalized_schedule: pd.DataFrame,
) -> CompletedGameReport:
    """Return uniquely mapped game IDs for events whose ``completed is True``.

    Incomplete event mappings are intentionally ignored before event identity
    validation. Eligible completed events must satisfy the existing public
    Odds API event-normalization and schedule-matching contracts.
    """

    events = _score_events(scores_payload)
    eligible = [event for event in events if event.get("completed") is True]
    normalized_events = _normalize_eligible_events(eligible)
    matched_events = _match_eligible_events(normalized_events, normalized_schedule)

    mapped: dict[str, str] = {}
    diagnostics: list[CompletedGameDiagnostic] = []
    for row in matched_events.itertuples(index=False):
        event_id = _required_text(row.event_id, "provider event_id")
        if row.event_match_status == "matched":
            mapped[event_id] = _required_text(
                row.nflverse_game_id,
                "matched nflverse_game_id",
            )
        elif row.event_match_status in {"unmatched", "ambiguous"}:
            diagnostics.append(
                CompletedGameDiagnostic(
                    provider_event_id=event_id,
                    match_status=row.event_match_status,
                    diagnostic=_required_text(row.event_match_note, "event match note"),
                )
            )
        else:
            raise CompletedGameValidationError("completed event matcher returned an invalid status")

    _reject_multiple_provider_events_per_game(mapped)
    return CompletedGameReport(
        completed_game_ids=tuple(sorted(mapped.values())),
        diagnostics=tuple(
            sorted(
                diagnostics,
                key=lambda item: (
                    item.provider_event_id,
                    item.match_status,
                    item.diagnostic,
                ),
            )
        ),
    )


def _score_events(
    scores_payload: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    if isinstance(scores_payload, (str, bytes)) or not isinstance(scores_payload, Sequence):
        raise CompletedGameValidationError("scores_payload must be a non-string sequence of objects")
    events = tuple(scores_payload)
    if not all(isinstance(event, Mapping) for event in events):
        raise CompletedGameValidationError("scores_payload items must be objects")
    return events


def _normalize_eligible_events(
    eligible: list[Mapping[str, Any]],
) -> pd.DataFrame:
    try:
        return normalize_nfl_events(eligible)
    except ValueError as error:
        if str(error).startswith("Conflicting NFL event metadata"):
            raise CompletedGameConflictError(
                "eligible completed provider event IDs conflict"
            ) from error
        raise CompletedGameValidationError(
            "eligible completed event fields are invalid"
        ) from error
    except TypeError as error:
        raise CompletedGameValidationError(
            "eligible completed event fields are invalid"
        ) from error


def _match_eligible_events(
    normalized_events: pd.DataFrame,
    normalized_schedule: pd.DataFrame,
) -> pd.DataFrame:
    try:
        return match_nfl_events_to_schedule(normalized_events, normalized_schedule)
    except (TypeError, ValueError) as error:
        raise CompletedGameValidationError(
            "normalized schedule cannot be safely matched to completed events"
        ) from error


def _reject_multiple_provider_events_per_game(mapped: Mapping[str, str]) -> None:
    by_game: dict[str, list[str]] = {}
    for event_id, game_id in mapped.items():
        by_game.setdefault(game_id, []).append(event_id)
    conflicts = sorted(
        game_id for game_id, event_ids in by_game.items() if len(event_ids) > 1
    )
    if conflicts:
        raise CompletedGameConflictError(
            "distinct completed provider event IDs map to nflverse game_id: "
            f"{conflicts[0]}"
        )


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not (normalized := value.strip()):
        raise CompletedGameValidationError(f"{field} must be nonblank text")
    return normalized
