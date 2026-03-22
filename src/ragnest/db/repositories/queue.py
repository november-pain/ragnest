"""Queue repository — ingestion queue management (SQLite state)."""

from __future__ import annotations

import fnmatch
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ragnest.db.repositories.base import BaseRepository
from ragnest.models.db import QueueItemRow

if TYPE_CHECKING:
    from ragnest.db.backend import DatabaseBackend

logger = logging.getLogger(__name__)


class QueueRepository(BaseRepository):
    """Repository for ingestion_queue table operations.

    Operates on the SQLite state backend.  Uses ``datetime('now')`` for
    timestamps (SQLite compatible).
    """

    def __init__(self, backend: DatabaseBackend) -> None:
        super().__init__(backend)

    def enqueue_file(self, kb_name: str, file_path: str, batch_id: str) -> bool:
        """Add a single file to the ingestion queue.

        Skips if the file is already pending/processing for this KB.
        Returns True if queued.
        """
        with self._backend.cursor() as cur:
            cur.execute(
                "SELECT id FROM ingestion_queue "
                "WHERE kb_name = %s AND file_path = %s "
                "AND status IN ('pending', 'processing')",
                (kb_name, file_path),
            )
            if cur.fetchone():
                return False

            cur.execute(
                "INSERT INTO ingestion_queue (kb_name, file_path, batch_id) VALUES (%s, %s, %s)",
                (kb_name, file_path, batch_id),
            )
        return True

    def enqueue_directory(
        self,
        kb_name: str,
        dir_path: str,
        batch_id: str,
        recursive: bool = True,
        file_patterns: str = "*",
    ) -> int:
        """Queue all matching files in a directory. Returns count queued."""
        path = Path(dir_path)
        patterns = [p.strip() for p in file_patterns.split(",")]
        glob_pattern = "**/*" if recursive else "*"
        files = sorted(f for f in path.glob(glob_pattern) if f.is_file())

        if patterns != ["*"]:
            files = [f for f in files if any(fnmatch.fnmatch(f.name, p) for p in patterns)]

        queued = 0
        with self._backend.cursor() as cur:
            for f in files:
                fpath = str(f.absolute())
                cur.execute(
                    "SELECT id FROM ingestion_queue "
                    "WHERE kb_name = %s AND file_path = %s "
                    "AND status IN ('pending', 'processing')",
                    (kb_name, fpath),
                )
                if not cur.fetchone():
                    cur.execute(
                        "INSERT INTO ingestion_queue "
                        "(kb_name, file_path, batch_id) "
                        "VALUES (%s, %s, %s)",
                        (kb_name, fpath, batch_id),
                    )
                    queued += 1

            # Update batch total_files to actual queued count
            cur.execute(
                "UPDATE batches SET total_files = %s WHERE id = %s",
                (queued, batch_id),
            )

        logger.info(
            "Queued %d files from '%s' for KB '%s' (batch %s)",
            queued,
            dir_path,
            kb_name,
            batch_id,
        )
        return queued

    def claim_next(self, kb_name: str | None = None) -> QueueItemRow | None:
        """Claim the next pending queue item.

        SQLite does not support ``FOR UPDATE SKIP LOCKED``, so we use a
        simple SELECT + UPDATE within the same transaction (the SQLite
        backend holds a lock for the duration).

        Returns None if no work is available.
        """
        with self._backend.cursor() as cur:
            if kb_name is not None:
                cur.execute(
                    "SELECT id, batch_id, kb_name, file_path, status, "
                    "error_message, chunk_count, created_at, "
                    "started_at, completed_at "
                    "FROM ingestion_queue "
                    "WHERE status = 'pending' AND kb_name = %s "
                    "ORDER BY id "
                    "LIMIT 1",
                    (kb_name,),
                )
            else:
                cur.execute(
                    "SELECT id, batch_id, kb_name, file_path, status, "
                    "error_message, chunk_count, created_at, "
                    "started_at, completed_at "
                    "FROM ingestion_queue "
                    "WHERE status = 'pending' "
                    "ORDER BY id "
                    "LIMIT 1"
                )
            row = cur.fetchone()
            if row is None:
                return None

            # Mark as processing
            cur.execute(
                "UPDATE ingestion_queue "
                "SET status = 'processing', started_at = datetime('now') "
                "WHERE id = %s",
                (row[0],),
            )

        return QueueItemRow(
            id=row[0],
            batch_id=row[1],
            kb_name=row[2],
            source_path=row[3],
            status="processing",
            error_message=row[5],
            chunk_count=row[6],
            queued_at=row[7],
            started_at=row[8],
            completed_at=row[9],
        )

    def mark_done(self, queue_id: int, chunk_count: int) -> None:
        """Mark a queue item as done with the number of chunks produced."""
        with self._backend.cursor() as cur:
            cur.execute(
                "UPDATE ingestion_queue "
                "SET status = 'done', chunk_count = %s, "
                "completed_at = datetime('now') "
                "WHERE id = %s",
                (chunk_count, queue_id),
            )

    def mark_failed(self, queue_id: int, error: str) -> None:
        """Mark a queue item as failed with the error message."""
        with self._backend.cursor() as cur:
            cur.execute(
                "UPDATE ingestion_queue "
                "SET status = 'failed', error_message = %s, "
                "completed_at = datetime('now') "
                "WHERE id = %s",
                (error, queue_id),
            )

    def mark_skipped(self, queue_id: int, reason: str) -> None:
        """Mark a queue item as skipped with the reason."""
        with self._backend.cursor() as cur:
            cur.execute(
                "UPDATE ingestion_queue "
                "SET status = 'skipped', error_message = %s, "
                "completed_at = datetime('now') "
                "WHERE id = %s",
                (reason, queue_id),
            )

    def get_pending_count(self, kb_name: str | None = None) -> int:
        """Count pending items, optionally filtered by KB."""
        with self._backend.cursor() as cur:
            if kb_name is not None:
                cur.execute(
                    "SELECT COUNT(*) FROM ingestion_queue "
                    "WHERE status = 'pending' AND kb_name = %s",
                    (kb_name,),
                )
            else:
                cur.execute("SELECT COUNT(*) FROM ingestion_queue WHERE status = 'pending'")
            row = cur.fetchone()
        return int(row[0]) if row else 0

    def get_failed(self, kb_name: str | None = None) -> list[QueueItemRow]:
        """List all failed queue items, optionally filtered by KB."""
        with self._backend.cursor() as cur:
            if kb_name is not None:
                cur.execute(
                    "SELECT id, batch_id, kb_name, file_path, status, "
                    "error_message, chunk_count, created_at, "
                    "started_at, completed_at "
                    "FROM ingestion_queue "
                    "WHERE status = 'failed' AND kb_name = %s "
                    "ORDER BY id",
                    (kb_name,),
                )
            else:
                cur.execute(
                    "SELECT id, batch_id, kb_name, file_path, status, "
                    "error_message, chunk_count, created_at, "
                    "started_at, completed_at "
                    "FROM ingestion_queue "
                    "WHERE status = 'failed' ORDER BY id"
                )
            rows = cur.fetchall()
        return [
            QueueItemRow(
                id=r[0],
                batch_id=r[1],
                kb_name=r[2],
                source_path=r[3],
                status=r[4],
                error_message=r[5],
                chunk_count=r[6],
                queued_at=r[7],
                started_at=r[8],
                completed_at=r[9],
            )
            for r in rows
        ]

    def reset_failed(self, kb_name: str | None = None) -> int:
        """Reset all failed items back to pending. Returns count reset."""
        with self._backend.cursor() as cur:
            if kb_name is not None:
                cur.execute(
                    "UPDATE ingestion_queue "
                    "SET status = 'pending', error_message = NULL, "
                    "started_at = NULL, completed_at = NULL "
                    "WHERE status = 'failed' AND kb_name = %s",
                    (kb_name,),
                )
            else:
                cur.execute(
                    "UPDATE ingestion_queue "
                    "SET status = 'pending', error_message = NULL, "
                    "started_at = NULL, completed_at = NULL "
                    "WHERE status = 'failed'"
                )
            count: int = int(cur.rowcount)
        logger.info("Reset %d failed queue items to pending", count)
        return count
