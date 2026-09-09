"""SQLite storage for football research decisions and single-decision settlement.

This module records the immutable odds snapshot selected during research.  The
``sportsbook`` field is the stable Odds API ``bookmaker_key``, not its mutable
display title. Recording creates only pending decisions. Settlement is a
separate operation that derives ``win``, ``loss``, or ``push`` from the stored
selection, stored line, and official actual result. It may change only
``status``, ``actual_result``, and ``settled_at``; exact retries are idempotent
and conflicting retries never overwrite stored results. Profit/loss and
external result retrieval remain outside this module's scope.
"""

from __future__ import annotations

import math
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from numbers import Integral, Real
from typing import Any


_SUPPORTED_MARKETS = {"QB": "player_pass_yds", "RB": "player_rush_yds"}
_SETTLED_STATUSES = frozenset({"win", "loss", "push"})


class DecisionStoreError(ValueError):
    """Base exception for decision-store input and conflict errors."""


class DecisionValidationError(DecisionStoreError):
    """Raised when a decision cannot be recorded as a pending decision."""


class DuplicateDecisionError(DecisionStoreError):
    """Raised when the same canonical odds selection is recorded twice."""


class DecisionIdConflictError(DecisionStoreError):
    """Raised when a supplied or generated decision ID already exists."""


class DecisionNotFoundError(DecisionStoreError):
    """Raised when settlement is requested for an unknown decision ID."""


class SettlementConflictError(DecisionStoreError):
    """Raised when a settlement conflicts with an existing settled decision."""


@dataclass(frozen=True)
class PendingDecision:
    """Input for an immutable pending research-decision snapshot.

    ``decision_id`` and ``recorded_at`` may be omitted only when callers pass
    deterministic ``id_factory`` and ``clock`` functions to ``record_decision``.
    """

    position: object
    player_id: object
    player_name: object
    team: object
    opponent: object
    season: object
    week: object
    game_id: object
    market_key: object
    sportsbook: object
    line: object
    selection: object
    selected_price: object
    odds_retrieved_at: object
    research_notes: object | None = None
    status: object = "pending"
    actual_result: object | None = None
    settled_at: object | None = None
    decision_id: object | None = None
    recorded_at: object | None = None


@dataclass(frozen=True)
class StoredDecision:
    """A normalized decision row read from or written to the store."""

    decision_id: str
    position: str
    player_id: str
    player_name: str
    team: str
    opponent: str
    season: int
    week: int
    game_id: str
    market_key: str
    sportsbook: str
    line: Decimal
    selection: str
    selected_price: int
    recorded_at: datetime
    odds_retrieved_at: datetime
    research_notes: str | None
    status: str
    actual_result: Decimal | None
    settled_at: datetime | None


def initialize_decision_store(connection: sqlite3.Connection) -> None:
    """Create the decisions schema without altering existing rows.

    A savepoint scopes the DDL to this operation, preserving any transaction
    work owned by the caller.
    """

    _require_connection(connection)
    connection.execute("SAVEPOINT initialize_decision_store")
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS decisions (
                decision_id TEXT PRIMARY KEY,
                position TEXT NOT NULL CHECK (position IN ('QB', 'RB')),
                player_id TEXT NOT NULL,
                player_name TEXT NOT NULL,
                team TEXT NOT NULL,
                opponent TEXT NOT NULL,
                season INTEGER NOT NULL CHECK (typeof(season) = 'integer' AND season > 0),
                week INTEGER NOT NULL CHECK (typeof(week) = 'integer' AND week > 0),
                game_id TEXT NOT NULL,
                market_key TEXT NOT NULL CHECK (market_key IN ('player_pass_yds', 'player_rush_yds')),
                sportsbook TEXT NOT NULL,
                line TEXT NOT NULL,
                selection TEXT NOT NULL CHECK (selection IN ('over', 'under')),
                selected_price INTEGER NOT NULL CHECK (
                    typeof(selected_price) = 'integer'
                    AND (selected_price <= -100 OR selected_price >= 100)
                ),
                recorded_at TEXT NOT NULL,
                odds_retrieved_at TEXT NOT NULL,
                research_notes TEXT,
                status TEXT NOT NULL CHECK (status IN ('pending', 'win', 'loss', 'push')),
                actual_result TEXT,
                settled_at TEXT,
                CHECK (
                    (position = 'QB' AND market_key = 'player_pass_yds')
                    OR (position = 'RB' AND market_key = 'player_rush_yds')
                ),
                CHECK (
                    (status = 'pending' AND actual_result IS NULL AND settled_at IS NULL)
                    OR (status IN ('win', 'loss', 'push') AND actual_result IS NOT NULL AND settled_at IS NOT NULL)
                ),
                UNIQUE (game_id, player_id, market_key, sportsbook, line, selection, selected_price)
            )
            """
        )
    except Exception:
        connection.execute("ROLLBACK TO initialize_decision_store")
        connection.execute("RELEASE initialize_decision_store")
        raise
    connection.execute("RELEASE initialize_decision_store")


def record_decision(
    connection: sqlite3.Connection,
    decision: PendingDecision,
    *,
    id_factory: Callable[[], object] | None = None,
    clock: Callable[[], object] | None = None,
) -> StoredDecision:
    """Validate and append one pending decision without committing caller work."""

    _require_connection(connection)
    stored = _normalize_pending_decision(
        decision,
        id_factory=id_factory or (lambda: str(uuid.uuid4())),
        clock=clock or (lambda: datetime.now(timezone.utc)),
    )

    connection.execute("SAVEPOINT record_decision")
    try:
        if _decision_id_exists(connection, stored.decision_id):
            raise DecisionIdConflictError("decision_id already exists")
        if _duplicate_exists(connection, stored):
            raise DuplicateDecisionError("duplicate decision already exists")
        connection.execute(
            """
            INSERT INTO decisions (
                decision_id, position, player_id, player_name, team, opponent,
                season, week, game_id, market_key, sportsbook, line, selection,
                selected_price, recorded_at, odds_retrieved_at, research_notes,
                status, actual_result, settled_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            _stored_values(stored),
        )
    except DecisionStoreError:
        connection.execute("ROLLBACK TO record_decision")
        connection.execute("RELEASE record_decision")
        raise
    except sqlite3.IntegrityError as error:
        connection.execute("ROLLBACK TO record_decision")
        connection.execute("RELEASE record_decision")
        raise DecisionStoreError("decision insert violated a database constraint") from error
    except Exception:
        connection.execute("ROLLBACK TO record_decision")
        connection.execute("RELEASE record_decision")
        raise
    connection.execute("RELEASE record_decision")
    return stored


def get_decision(connection: sqlite3.Connection, decision_id: object) -> StoredDecision | None:
    """Return one decision by its identifier, or ``None`` when it is absent."""

    _require_connection(connection)
    identifier = _required_text(decision_id, "decision_id")
    row = connection.execute(
        "SELECT * FROM decisions WHERE decision_id = ?", (identifier,)
    ).fetchone()
    return _row_to_decision(row) if row is not None else None


def list_decisions(connection: sqlite3.Connection) -> list[StoredDecision]:
    """Return all decisions ordered by recorded timestamp then decision ID."""

    _require_connection(connection)
    rows = connection.execute(
        "SELECT * FROM decisions ORDER BY recorded_at ASC, decision_id ASC"
    ).fetchall()
    return [_row_to_decision(row) for row in rows]


def settle_decision(
    connection: sqlite3.Connection,
    decision_id: object,
    actual_result: object,
    *,
    settled_at: object | None = None,
    clock: Callable[[], object] | None = None,
) -> StoredDecision:
    """Settle one pending decision from its immutable line and selection.

    Exact canonical-result retries return the original settled row unchanged.
    A different result or an inconsistent stored settlement raises
    ``SettlementConflictError`` without overwriting anything.
    """

    _require_connection(connection)
    identifier = _required_text(decision_id, "decision_id")
    result = _actual_result(actual_result)
    selected_clock = clock or (lambda: datetime.now(timezone.utc))

    connection.execute("SAVEPOINT settle_decision")
    try:
        existing = _get_decision_by_id(connection, identifier)
        if existing is None:
            raise DecisionNotFoundError("decision_id was not found")
        timestamp = _settlement_timestamp(
            settled_at if settled_at is not None else selected_clock(), existing.recorded_at
        )
        requested_status = _settlement_status(existing.selection, existing.line, result)

        if existing.status != "pending":
            _validate_existing_settlement(existing)
            if existing.actual_result == result:
                connection.execute("RELEASE settle_decision")
                return existing
            raise SettlementConflictError("settlement result conflicts with the stored settlement")

        cursor = connection.execute(
            """
            UPDATE decisions
            SET status = ?, actual_result = ?, settled_at = ?
            WHERE decision_id = ? AND status = 'pending'
            """,
            (requested_status, _decimal_text(result), _timestamp_text(timestamp), identifier),
        )
        if cursor.rowcount == 1:
            settled = _get_decision_by_id(connection, identifier)
            if settled is None:
                raise DecisionNotFoundError("decision_id was not found")
            connection.execute("RELEASE settle_decision")
            return settled

        # A concurrent writer may have settled the row after the initial read.
        current = _get_decision_by_id(connection, identifier)
        if current is None:
            raise DecisionNotFoundError("decision_id was not found")
        _validate_existing_settlement(current)
        if current.actual_result == result:
            connection.execute("RELEASE settle_decision")
            return current
        raise SettlementConflictError("settlement result conflicts with the stored settlement")
    except DecisionStoreError:
        connection.execute("ROLLBACK TO settle_decision")
        connection.execute("RELEASE settle_decision")
        raise
    except sqlite3.IntegrityError as error:
        connection.execute("ROLLBACK TO settle_decision")
        connection.execute("RELEASE settle_decision")
        raise DecisionStoreError("decision settlement violated a database constraint") from error
    except Exception:
        connection.execute("ROLLBACK TO settle_decision")
        connection.execute("RELEASE settle_decision")
        raise


def _normalize_pending_decision(
    decision: PendingDecision,
    *,
    id_factory: Callable[[], object],
    clock: Callable[[], object],
) -> StoredDecision:
    if not isinstance(decision, PendingDecision):
        raise DecisionValidationError("decision must be a PendingDecision")
    decision_id = _required_text(
        id_factory() if decision.decision_id is None else decision.decision_id,
        "decision_id",
    )
    recorded_at = _normalize_timestamp(
        clock() if decision.recorded_at is None else decision.recorded_at,
        "recorded_at",
    )
    odds_retrieved_at = _normalize_timestamp(decision.odds_retrieved_at, "odds_retrieved_at")
    if recorded_at < odds_retrieved_at:
        raise DecisionValidationError("recorded_at must not precede odds_retrieved_at")

    position = _required_text(decision.position, "position").upper()
    market_key = _required_text(decision.market_key, "market_key")
    if position not in _SUPPORTED_MARKETS or _SUPPORTED_MARKETS[position] != market_key:
        raise DecisionValidationError("position and market_key must be QB/player_pass_yds or RB/player_rush_yds")
    selection = _required_text(decision.selection, "selection").lower()
    if selection not in {"over", "under"}:
        raise DecisionValidationError("selection must be 'over' or 'under'")
    status = _required_text(decision.status, "status").lower()
    if status != "pending":
        raise DecisionValidationError("status must be 'pending' when recording")
    if decision.actual_result is not None:
        raise DecisionValidationError("actual_result must be null when recording")
    if decision.settled_at is not None:
        raise DecisionValidationError("settled_at must be null when recording")

    return StoredDecision(
        decision_id=decision_id,
        position=position,
        player_id=_required_text(decision.player_id, "player_id"),
        player_name=_required_text(decision.player_name, "player_name"),
        team=_required_text(decision.team, "team"),
        opponent=_required_text(decision.opponent, "opponent"),
        season=_positive_integer(decision.season, "season"),
        week=_positive_integer(decision.week, "week"),
        game_id=_required_text(decision.game_id, "game_id"),
        market_key=market_key,
        sportsbook=_required_text(decision.sportsbook, "sportsbook"),
        line=_decimal_line(decision.line),
        selection=selection,
        selected_price=_american_price(decision.selected_price),
        recorded_at=recorded_at,
        odds_retrieved_at=odds_retrieved_at,
        research_notes=_optional_notes(decision.research_notes),
        status=status,
        actual_result=None,
        settled_at=None,
    )


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not (normalized := value.strip()):
        raise DecisionValidationError(f"{field} must be nonblank text")
    return normalized


def _optional_notes(value: object | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise DecisionValidationError("research_notes must be text or null")
    return value.strip() or None


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise DecisionValidationError(f"{field} must be a positive integer")
    return int(value)


def _american_price(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise DecisionValidationError("selected_price must be an integer")
    price = int(value)
    if -99 <= price <= 99:
        raise DecisionValidationError("selected_price must be <= -100 or >= 100")
    return price


def _decimal_line(value: object) -> Decimal:
    return _finite_decimal(value, "line")


def _actual_result(value: object) -> Decimal:
    return _finite_decimal(value, "actual_result")


def _finite_decimal(value: object, field: str) -> Decimal:
    if isinstance(value, bool):
        raise DecisionValidationError(f"{field} must be a finite numeric value and not a boolean")
    if isinstance(value, Decimal):
        line = value
    elif isinstance(value, Real):
        if not math.isfinite(value):
            raise DecisionValidationError(f"{field} must be a finite numeric value and not a boolean")
        try:
            line = Decimal(str(value))
        except InvalidOperation as error:
            raise DecisionValidationError(f"{field} must be a finite numeric value and not a boolean") from error
    else:
        raise DecisionValidationError(f"{field} must be a finite numeric value and not a boolean")
    if not line.is_finite():
        raise DecisionValidationError(f"{field} must be a finite numeric value and not a boolean")
    return line.normalize() if line != 0 else Decimal(0)


def _normalize_timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise DecisionValidationError(f"{field} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _timestamp_text(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _stored_values(decision: StoredDecision) -> tuple[object, ...]:
    return (
        decision.decision_id, decision.position, decision.player_id, decision.player_name,
        decision.team, decision.opponent, decision.season, decision.week, decision.game_id,
        decision.market_key, decision.sportsbook, _decimal_text(decision.line),
        decision.selection, decision.selected_price, _timestamp_text(decision.recorded_at),
        _timestamp_text(decision.odds_retrieved_at), decision.research_notes, decision.status,
        decision.actual_result, decision.settled_at,
    )


def _decision_id_exists(connection: sqlite3.Connection, decision_id: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM decisions WHERE decision_id = ?", (decision_id,)
    ).fetchone() is not None


def _duplicate_exists(connection: sqlite3.Connection, decision: StoredDecision) -> bool:
    return connection.execute(
        """
        SELECT 1 FROM decisions
        WHERE game_id = ? AND player_id = ? AND market_key = ? AND sportsbook = ?
          AND line = ? AND selection = ? AND selected_price = ?
        """,
        (
            decision.game_id, decision.player_id, decision.market_key, decision.sportsbook,
            _decimal_text(decision.line), decision.selection, decision.selected_price,
        ),
    ).fetchone() is not None


def _get_decision_by_id(connection: sqlite3.Connection, decision_id: str) -> StoredDecision | None:
    row = connection.execute(
        "SELECT * FROM decisions WHERE decision_id = ?", (decision_id,)
    ).fetchone()
    return _row_to_decision(row) if row is not None else None


def _settlement_timestamp(value: object, recorded_at: datetime) -> datetime:
    timestamp = _normalize_timestamp(value, "settled_at")
    if timestamp < recorded_at:
        raise DecisionValidationError("settled_at must not precede recorded_at")
    return timestamp


def _settlement_status(selection: str, line: Decimal, actual_result: Decimal) -> str:
    if actual_result == line:
        return "push"
    if selection == "over":
        return "win" if actual_result > line else "loss"
    if selection == "under":
        return "win" if actual_result < line else "loss"
    raise SettlementConflictError("stored selection is inconsistent")


def _validate_existing_settlement(decision: StoredDecision) -> None:
    if decision.status not in _SETTLED_STATUSES:
        raise SettlementConflictError("stored settlement status is inconsistent")
    if decision.actual_result is None or decision.settled_at is None:
        raise SettlementConflictError("stored settlement fields are inconsistent")
    if _settlement_status(decision.selection, decision.line, decision.actual_result) != decision.status:
        raise SettlementConflictError("stored settlement status is inconsistent")


def _row_to_decision(row: sqlite3.Row | tuple[Any, ...]) -> StoredDecision:
    values = dict(row) if isinstance(row, sqlite3.Row) else {
        name: value for name, value in zip(_COLUMN_NAMES, row, strict=True)
    }
    return StoredDecision(
        decision_id=values["decision_id"], position=values["position"],
        player_id=values["player_id"], player_name=values["player_name"],
        team=values["team"], opponent=values["opponent"], season=values["season"],
        week=values["week"], game_id=values["game_id"], market_key=values["market_key"],
        sportsbook=values["sportsbook"], line=Decimal(values["line"]),
        selection=values["selection"], selected_price=values["selected_price"],
        recorded_at=_parse_timestamp(values["recorded_at"]),
        odds_retrieved_at=_parse_timestamp(values["odds_retrieved_at"]),
        research_notes=values["research_notes"], status=values["status"],
        actual_result=(
            Decimal(values["actual_result"])
            if values["actual_result"] is not None
            else None
        ),
        settled_at=(
            _parse_timestamp(values["settled_at"])
            if values["settled_at"] is not None
            else None
        ),
    )


_COLUMN_NAMES = (
    "decision_id", "position", "player_id", "player_name", "team", "opponent", "season",
    "week", "game_id", "market_key", "sportsbook", "line", "selection", "selected_price",
    "recorded_at", "odds_retrieved_at", "research_notes", "status", "actual_result", "settled_at",
)


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _require_connection(connection: object) -> None:
    if not isinstance(connection, sqlite3.Connection):
        raise TypeError("connection must be a sqlite3.Connection")
