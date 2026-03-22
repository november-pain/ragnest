"""Watch path repository — directory watch management (SQLite state)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ragnest.db.repositories.base import BaseRepository
from ragnest.models.domain import WatchPathInfo

if TYPE_CHECKING:
    from ragnest.db.backend import DatabaseBackend

logger = logging.getLogger(__name__)


class WatchPathRepository(BaseRepository):
    """Repository for watch_paths table operations.

    Operates on the SQLite state backend.
    """

    def __init__(self, backend: DatabaseBackend) -> None:
        super().__init__(backend)

    def add(
        self,
        kb_name: str,
        dir_path: str,
        recursive: bool = True,
        file_patterns: str = "*",
    ) -> WatchPathInfo:
        """Add or update a watch path. Returns the resulting WatchPathInfo."""
        with self._backend.cursor() as cur:
            # Try insert; on conflict update
            cur.execute(
                "INSERT INTO watch_paths "
                "(kb_name, dir_path, recursive, file_patterns) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (kb_name, dir_path) DO UPDATE SET "
                "recursive = excluded.recursive, "
                "file_patterns = excluded.file_patterns, "
                "enabled = 1",
                (kb_name, dir_path, recursive, file_patterns),
            )

            # Fetch the resulting row
            cur.execute(
                "SELECT id, kb_name, dir_path, recursive, enabled, "
                "file_patterns, last_scanned_at "
                "FROM watch_paths WHERE kb_name = %s AND dir_path = %s",
                (kb_name, dir_path),
            )
            row = cur.fetchone()

        if row is None:
            msg = "Failed to insert/update watch path"
            raise RuntimeError(msg)

        logger.info(
            "Watch path added: '%s' -> KB '%s'", dir_path, kb_name
        )
        return WatchPathInfo(
            id=row[0],
            kb_name=row[1],
            dir_path=row[2],
            recursive=bool(row[3]),
            enabled=bool(row[4]),
            file_patterns=row[5],
            last_scanned_at=row[6],
        )

    def remove(self, kb_name: str, dir_path: str) -> bool:
        """Remove a watch path. Returns True if it existed."""
        with self._backend.cursor() as cur:
            cur.execute(
                "DELETE FROM watch_paths "
                "WHERE kb_name = %s AND dir_path = %s",
                (kb_name, dir_path),
            )
            return cur.rowcount > 0  # type: ignore[return-value]

    def list_all(
        self, kb_name: str | None = None
    ) -> list[WatchPathInfo]:
        """List watch paths, optionally filtered by KB."""
        with self._backend.cursor() as cur:
            if kb_name is not None:
                cur.execute(
                    "SELECT id, kb_name, dir_path, recursive, enabled, "
                    "file_patterns, last_scanned_at "
                    "FROM watch_paths WHERE kb_name = %s "
                    "ORDER BY dir_path",
                    (kb_name,),
                )
            else:
                cur.execute(
                    "SELECT id, kb_name, dir_path, recursive, enabled, "
                    "file_patterns, last_scanned_at "
                    "FROM watch_paths ORDER BY kb_name, dir_path"
                )
            rows = cur.fetchall()
        return [
            WatchPathInfo(
                id=r[0],
                kb_name=r[1],
                dir_path=r[2],
                recursive=bool(r[3]),
                enabled=bool(r[4]),
                file_patterns=r[5],
                last_scanned_at=r[6],
            )
            for r in rows
        ]

    def set_enabled(
        self, kb_name: str, dir_path: str, enabled: bool
    ) -> bool:
        """Enable or disable a watch path. Returns True if it existed."""
        with self._backend.cursor() as cur:
            cur.execute(
                "UPDATE watch_paths SET enabled = %s "
                "WHERE kb_name = %s AND dir_path = %s",
                (enabled, kb_name, dir_path),
            )
            return cur.rowcount > 0  # type: ignore[return-value]

    def update_last_scanned(self, watch_path_id: int) -> None:
        """Update the last_scanned_at timestamp."""
        with self._backend.cursor() as cur:
            cur.execute(
                "UPDATE watch_paths SET last_scanned_at = datetime('now') "
                "WHERE id = %s",
                (watch_path_id,),
            )

    def get_active(self) -> list[WatchPathInfo]:
        """List all enabled watch paths across all KBs."""
        with self._backend.cursor() as cur:
            cur.execute(
                "SELECT id, kb_name, dir_path, recursive, enabled, "
                "file_patterns, last_scanned_at "
                "FROM watch_paths WHERE enabled = 1 "
                "ORDER BY kb_name, dir_path"
            )
            rows = cur.fetchall()
        return [
            WatchPathInfo(
                id=r[0],
                kb_name=r[1],
                dir_path=r[2],
                recursive=bool(r[3]),
                enabled=bool(r[4]),
                file_patterns=r[5],
                last_scanned_at=r[6],
            )
            for r in rows
        ]
