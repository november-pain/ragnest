"""SQLite backend for local state storage — zero-config, file-based."""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)


class _SQLiteCursorWrapper:
    """Wraps a ``sqlite3.Cursor`` to translate ``%s`` placeholders to ``?``.

    This allows repositories to use the same SQL strings as for PostgreSQL
    without modifications.  ``psycopg2.sql`` composables are **not** supported;
    only plain-string queries with ``%s`` value placeholders.
    """

    _PARAM_RE = re.compile(r"%s")

    def __init__(self, cursor: sqlite3.Cursor) -> None:
        self._cursor = cursor

    # -- delegated attributes --------------------------------------------------

    @property
    def description(self) -> Any:
        return self._cursor.description

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount

    @property
    def lastrowid(self) -> int | None:
        return self._cursor.lastrowid

    # -- query translation -----------------------------------------------------

    def _translate(self, sql: str) -> str:
        """Replace ``%s`` with ``?`` for SQLite parameter binding."""
        return self._PARAM_RE.sub("?", sql)

    def execute(
        self,
        sql: str | Any,
        params: Any = None,
    ) -> _SQLiteCursorWrapper:
        """Execute *sql* after translating ``%s`` → ``?``.

        Multi-statement SQL (e.g. DDL scripts) is split on ``;`` and each
        statement is executed individually, keeping everything within the
        normal transaction flow managed by ``connection()``.
        """
        query = str(sql)
        query = self._translate(query)
        if params is not None:
            # Convert Python booleans to int for SQLite
            if isinstance(params, (list, tuple)):
                converted: list[Any] = [
                    int(p) if isinstance(p, bool) else p  # pyright: ignore[reportUnknownVariableType]
                    for p in params  # pyright: ignore[reportUnknownVariableType]
                ]
                self._cursor.execute(query, tuple(converted))
            else:
                self._cursor.execute(query, params)
        else:
            # Split multi-statement SQL and execute each individually.
            # Never use executescript() — it issues an implicit COMMIT
            # before running, which breaks our transaction model.
            statements = [s.strip() for s in query.split(";") if s.strip()]
            if len(statements) > 1:
                for stmt in statements:
                    self._cursor.execute(stmt)
            else:
                self._cursor.execute(query)
        return self

    def executemany(
        self,
        sql: str | Any,
        params_seq: Any,
    ) -> _SQLiteCursorWrapper:
        query = str(sql)
        query = self._translate(query)
        self._cursor.executemany(query, params_seq)
        return self

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._cursor.fetchone()  # type: ignore[no-any-return]

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._cursor.fetchall()

    def fetchmany(self, size: int = 1) -> list[tuple[Any, ...]]:
        return self._cursor.fetchmany(size)

    def close(self) -> None:
        self._cursor.close()

    def __enter__(self) -> _SQLiteCursorWrapper:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class SQLiteBackend:
    """SQLite backend implementing the ``DatabaseBackend`` protocol.

    Auto-creates the database file at the given *path*.  Uses
    ``check_same_thread=False`` and an internal lock for thread safety.
    """

    def __init__(self, path: str) -> None:
        if path == ":memory:":
            self._path = ":memory:"
        else:
            resolved = Path(path).expanduser().resolve()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            self._path = str(resolved)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        logger.info("SQLiteBackend opened: %s", self._path)

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield the shared connection. Commit on success, rollback on error."""
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    @contextmanager
    def cursor(self) -> Generator[_SQLiteCursorWrapper, None, None]:
        """Yield a cursor-wrapper within a managed connection."""
        with self.connection() as conn:
            raw = conn.cursor()
            wrapper = _SQLiteCursorWrapper(raw)
            try:
                yield wrapper
            finally:
                raw.close()

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()
        logger.info("SQLiteBackend closed: %s", self._path)
