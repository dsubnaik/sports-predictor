"""Match pending decisions to completed normalized player-game results.

Matching is strictly by season, week, game ID, and player ID.  Callers must
provide completed game IDs because the presence of a player row does not prove
the game is final. Player-name fallback is intentionally forbidden, and a
missing player row is never treated as zero. This module does not mutate,
persist, or settle decisions; a later caller passes ``matched_results`` to
``football.decisions.settle_decision``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from numbers import Integral, Real

import pandas as pd

from football.decisions import StoredDecision


_QB_REQUIRED_COLUMNS = ("season", "week", "game_id", "player_id", "passing_yards")
_RB_REQUIRED_COLUMNS = ("season", "week", "game_id", "player_id", "rushing_yards")
_SETTLED_STATUSES = frozenset({"win", "loss", "push"})
_PENDING_STATUS = "pending"
_MATCHED = "matched"
_GAME_PENDING = "game_pending"
_PLAYER_RESULT_MISSING = "player_result_missing"


class DecisionResultMatchError(ValueError):
    """Base exception for decision-result matching errors."""


class DecisionResultValidationError(DecisionResultMatchError):
    """Raised when matcher inputs cannot be safely interpreted."""


class ConflictingResultDataError(DecisionResultMatchError):
    """Raised when relevant normalized source rows disagree on a result."""


@dataclass(frozen=True)
class DecisionResultMatch:
    """One pending decision's settlement-readiness diagnostic."""

    decision_id: str
    position: str
    market_key: str
    season: int
    week: int
    game_id: str
    player_id: str
    match_status: str
    actual_result: Decimal | None
    diagnostic: str | None


@dataclass(frozen=True)
class DecisionResultMatchReport:
    """Deterministically ordered result diagnostics for pending decisions only."""

    matches: tuple[DecisionResultMatch, ...]

    @property
    def matched_results(self) -> tuple[DecisionResultMatch, ...]:
        """Return only settlement-ready matches in report order."""

        return tuple(match for match in self.matches if match.match_status == _MATCHED)


def match_decision_results(
    decisions: Iterable[StoredDecision],
    quarterback_games: pd.DataFrame,
    running_back_games: pd.DataFrame,
    completed_game_ids: Iterable[object],
) -> DecisionResultMatchReport:
    """Return deterministic completed-result diagnostics for pending decisions.

    Both source schemas are validated up front. Result values and duplicate
    conflicts are evaluated only for exact player-game keys requested by a
    pending decision whose game is explicitly completed.
    """

    _validate_source_schema(quarterback_games, _QB_REQUIRED_COLUMNS, "Quarterback games")
    _validate_source_schema(running_back_games, _RB_REQUIRED_COLUMNS, "Running-back games")
    completed_ids = _normalize_completed_game_ids(completed_game_ids)
    pending = _pending_decisions(decisions)

    qb_keys = {
        _decision_key(decision)
        for decision in pending
        if decision.position == "QB" and decision.game_id in completed_ids
    }
    rb_keys = {
        _decision_key(decision)
        for decision in pending
        if decision.position == "RB" and decision.game_id in completed_ids
    }
    qb_results = _relevant_result_lookup(
        quarterback_games, _QB_REQUIRED_COLUMNS, "passing_yards", qb_keys, "quarterback"
    )
    rb_results = _relevant_result_lookup(
        running_back_games, _RB_REQUIRED_COLUMNS, "rushing_yards", rb_keys, "running-back"
    )

    matches: list[DecisionResultMatch] = []
    for decision in pending:
        key = _decision_key(decision)
        if decision.game_id not in completed_ids:
            matches.append(_unresolved_match(decision, _GAME_PENDING, "Game is not in completed_game_ids."))
            continue
        results = qb_results if decision.position == "QB" else rb_results
        actual_result = results.get(key)
        if actual_result is None:
            matches.append(
                _unresolved_match(
                    decision,
                    _PLAYER_RESULT_MISSING,
                    "No normalized player result exists for the completed game.",
                )
            )
            continue
        matches.append(
            DecisionResultMatch(
                decision_id=decision.decision_id,
                position=decision.position,
                market_key=decision.market_key,
                season=decision.season,
                week=decision.week,
                game_id=decision.game_id,
                player_id=decision.player_id,
                match_status=_MATCHED,
                actual_result=actual_result,
                diagnostic=None,
            )
        )

    return DecisionResultMatchReport(tuple(sorted(matches, key=_match_sort_key)))


def _validate_source_schema(data: object, required_columns: tuple[str, ...], label: str) -> None:
    if not isinstance(data, pd.DataFrame):
        raise DecisionResultValidationError(f"{label} must be a pandas DataFrame")
    missing = sorted(set(required_columns).difference(data.columns))
    if missing:
        raise DecisionResultValidationError(f"{label} is missing required columns: {missing}")


def _normalize_completed_game_ids(completed_game_ids: Iterable[object]) -> frozenset[str]:
    if isinstance(completed_game_ids, (str, bytes)):
        raise DecisionResultValidationError("completed_game_ids must be a non-string iterable")
    try:
        values = tuple(completed_game_ids)
    except TypeError as error:
        raise DecisionResultValidationError("completed_game_ids must be an iterable") from error
    normalized: list[str] = []
    for value in values:
        normalized.append(_required_text(value, "completed_game_ids item"))
    return frozenset(normalized)


def _pending_decisions(decisions: Iterable[StoredDecision]) -> tuple[StoredDecision, ...]:
    if isinstance(decisions, (str, bytes)):
        raise DecisionResultValidationError("decisions must be a non-string iterable of StoredDecision objects")
    try:
        values = tuple(decisions)
    except TypeError as error:
        raise DecisionResultValidationError("decisions must be an iterable of StoredDecision objects") from error

    by_id: dict[str, StoredDecision] = {}
    for decision in values:
        if not isinstance(decision, StoredDecision):
            raise DecisionResultValidationError("decisions must contain StoredDecision objects")
        decision_id = _required_text(decision.decision_id, "decision_id")
        if decision_id in by_id and by_id[decision_id] != decision:
            raise DecisionResultValidationError(f"conflicting decisions share decision_id: {decision_id}")
        by_id[decision_id] = decision

    pending: list[StoredDecision] = []
    for decision_id in sorted(by_id):
        decision = by_id[decision_id]
        if decision.status in _SETTLED_STATUSES:
            continue
        if decision.status != _PENDING_STATUS:
            raise DecisionResultValidationError("decision status must be pending, win, loss, or push")
        _validate_pending_decision(decision)
        pending.append(decision)
    return tuple(sorted(pending, key=_decision_sort_key))


def _validate_pending_decision(decision: StoredDecision) -> None:
    if (decision.position, decision.market_key) not in {
        ("QB", "player_pass_yds"),
        ("RB", "player_rush_yds"),
    }:
        raise DecisionResultValidationError("pending decision has an unsupported position/market_key combination")
    _positive_integer(decision.season, "decision season")
    _positive_integer(decision.week, "decision week")
    _required_text(decision.game_id, "decision game_id")
    _required_text(decision.player_id, "decision player_id")


def _relevant_result_lookup(
    data: pd.DataFrame,
    required_columns: tuple[str, ...],
    result_column: str,
    target_keys: set[tuple[int, int, str, str]],
    label: str,
) -> dict[tuple[int, int, str, str], Decimal]:
    if not target_keys:
        return {}

    target_game_players = {(game_id, player_id) for _, _, game_id, player_id in target_keys}
    candidates: dict[tuple[int, int, str, str], list[object]] = {}
    for row in data.loc[:, required_columns].itertuples(index=False, name=None):
        season, week, game_id, player_id, result = row
        game_identifier = _optional_text(game_id)
        if game_identifier is None:
            continue
        player_identifier = _optional_text(player_id)
        if player_identifier is None or (game_identifier, player_identifier) not in target_game_players:
            # This row cannot be a result source for a requested player. Its
            # values must not block an unrelated decision.
            continue
        key = (
            _positive_integer(season, f"{label} season"),
            _positive_integer(week, f"{label} week"),
            game_identifier,
            player_identifier,
        )
        if key in target_keys:
            candidates.setdefault(key, []).append(result)

    invalid_keys: list[tuple[int, int, str, str]] = []
    normalized: dict[tuple[int, int, str, str], set[Decimal]] = {}
    for key in sorted(candidates):
        values: set[Decimal] = set()
        for result in candidates[key]:
            try:
                values.add(_canonical_decimal(result, result_column))
            except DecisionResultValidationError:
                invalid_keys.append(key)
        normalized[key] = values
    if invalid_keys:
        raise DecisionResultValidationError(
            f"{result_column} is invalid for key: {min(invalid_keys)}"
        )

    conflicts = [key for key, values in normalized.items() if len(values) > 1]
    if conflicts:
        raise ConflictingResultDataError(
            f"Conflicting {label} results for key: {min(conflicts)}"
        )
    return {key: next(iter(values)) for key, values in normalized.items() if values}


def _canonical_decimal(value: object, field: str) -> Decimal:
    if isinstance(value, bool):
        raise DecisionResultValidationError(f"{field} must be a finite numeric value and not a boolean")
    if isinstance(value, Decimal):
        decimal_value = value
    elif isinstance(value, Real):
        if not math.isfinite(value):
            raise DecisionResultValidationError(f"{field} must be a finite numeric value and not a boolean")
        try:
            decimal_value = Decimal(str(value))
        except InvalidOperation as error:
            raise DecisionResultValidationError(f"{field} must be a finite numeric value and not a boolean") from error
    else:
        raise DecisionResultValidationError(f"{field} must be a finite numeric value and not a boolean")
    if not decimal_value.is_finite():
        raise DecisionResultValidationError(f"{field} must be a finite numeric value and not a boolean")
    return decimal_value.normalize() if decimal_value != 0 else Decimal(0)


def _unresolved_match(decision: StoredDecision, status: str, diagnostic: str) -> DecisionResultMatch:
    return DecisionResultMatch(
        decision_id=decision.decision_id,
        position=decision.position,
        market_key=decision.market_key,
        season=decision.season,
        week=decision.week,
        game_id=decision.game_id,
        player_id=decision.player_id,
        match_status=status,
        actual_result=None,
        diagnostic=diagnostic,
    )


def _decision_key(decision: StoredDecision) -> tuple[int, int, str, str]:
    return decision.season, decision.week, decision.game_id, decision.player_id


def _decision_sort_key(decision: StoredDecision) -> tuple[int, int, str, str, str]:
    return (*_decision_key(decision), decision.decision_id)


def _match_sort_key(match: DecisionResultMatch) -> tuple[int, int, str, str, str]:
    return match.season, match.week, match.game_id, match.player_id, match.decision_id


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not (normalized := value.strip()):
        raise DecisionResultValidationError(f"{field} must be nonblank text")
    return normalized


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise DecisionResultValidationError(f"{field} must be a positive integer")
    return int(value)
