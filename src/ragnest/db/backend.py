"""DatabaseBackend protocol — abstract interface for all database backends."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Generator


@runtime_checkable
class DatabaseBackend(Protocol):
    """Protocol that every database backend must satisfy.

    Provides connection pooling, automatic commit/rollback,
    and cursor convenience methods.

    The yielded connection and cursor types are ``Any`` to allow both
    PostgreSQL (psycopg2) and SQLite backends to satisfy the protocol
    without type conflicts.
    """

    @contextmanager
    def connection(self) -> Generator[Any, None, None]:
        """Yield a connection. Commit on success, rollback on error, release."""
        ...

    @contextmanager
    def cursor(self) -> Generator[Any, None, None]:
        """Yield a cursor within a managed connection."""
        ...

    def close(self) -> None:
        """Close all connections in the pool."""
        ...
