"""Open the local file-backed SQLite store for football research decisions.

The default file is ``football/data/local/decisions.sqlite3`` beneath this
package. Each returned connection belongs to the caller and must be closed by
the caller. The local file is intentionally ignored by Git. This is local
development storage only: an ephemeral Streamlit host can lose it, so durable
cloud deployment needs a separate persistence plan such as PostgreSQL.

Opening this database initializes only the existing decision-store schema; it
does not automatically record or settle decisions.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from football.config import FOOTBALL_DIR
from football.decisions import initialize_decision_store


LOCAL_DECISION_DATABASE_PATH = (
    FOOTBALL_DIR / "data" / "local" / "decisions.sqlite3"
)


class DecisionDatabaseValidationError(ValueError):
    """Raised when a requested local decision-database path is invalid."""


def get_local_decision_database_path() -> Path:
    """Return the deterministic default local database path without creating it."""

    return LOCAL_DECISION_DATABASE_PATH


def open_local_decision_database(
    path: Path | str | None = None,
) -> sqlite3.Connection:
    """Open and initialize one caller-owned file-backed decision-store connection.

    If schema initialization fails, this function closes only the connection it
    created and preserves the original exception. It never substitutes an
    in-memory database for an unavailable requested file.
    """

    database_path = _database_path(path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(database_path))
    try:
        initialize_decision_store(connection)
    except Exception:
        connection.close()
        raise
    return connection


def _database_path(path: Path | str | None) -> Path:
    if path is None:
        return get_local_decision_database_path()
    if isinstance(path, str):
        if not path.strip():
            raise DecisionDatabaseValidationError("path must be nonblank file path text")
        if path == ":memory:":
            raise DecisionDatabaseValidationError("path must not be ':memory:'")
        return Path(path)
    if isinstance(path, Path):
        if str(path) == ":memory:":
            raise DecisionDatabaseValidationError("path must not be ':memory:'")
        return path
    raise DecisionDatabaseValidationError("path must be a pathlib.Path, string, or null")
