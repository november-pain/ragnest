"""PostgreSQL backend using psycopg2 with connection pooling."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, cast

from pgvector.psycopg2 import register_vector  # pyright: ignore[reportUnknownVariableType]
from psycopg2.pool import ThreadedConnectionPool

if TYPE_CHECKING:
    from collections.abc import Generator

    from psycopg2.extensions import connection, cursor

logger = logging.getLogger(__name__)


class PostgresBackend:
    """Local PostgreSQL backend wrapping a ThreadedConnectionPool.

    On each new connection: registers pgvector types and sets
    hnsw.ef_search for better recall.
    """

    def __init__(
        self,
        dsn: str,
        min_conn: int = 1,
        max_conn: int = 10,
    ) -> None:
        self._dsn = dsn
        self._pool = ThreadedConnectionPool(min_conn, max_conn, dsn)
        logger.info(
            "PostgresBackend pool created (min=%d, max=%d)",
            min_conn,
            max_conn,
        )

    def _setup_connection(self, conn: connection) -> None:
        """Register pgvector types and tune HNSW search on a fresh connection."""
        try:
            register_vector(conn)  # pyright: ignore[reportUnknownArgumentType]
            with conn.cursor() as cur:
                cur.execute("SET hnsw.ef_search = 100")
            conn.commit()
        except Exception:
            # Vector extension may not exist yet (first run before schema init)
            conn.rollback()

    @contextmanager
    def connection(self) -> Generator[connection, None, None]:
        """Yield a pooled connection. Commit on success, rollback on error."""
        conn = cast("connection", self._pool.getconn())  # pyright: ignore[reportUnknownMemberType]
        try:
            self._setup_connection(conn)
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)  # pyright: ignore[reportUnknownMemberType]

    @contextmanager
    def cursor(self) -> Generator[cursor, None, None]:
        """Yield a cursor within a managed connection."""
        with self.connection() as conn, conn.cursor() as cur:
            yield cur

    def close(self) -> None:
        """Close all connections in the pool."""
        self._pool.closeall()
        logger.info("PostgresBackend pool closed")
