"""Batch repository — ingestion batch lifecycle management (SQLite state)."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from ragnest.db.repositories.base import BaseRepository
from ragnest.exceptions import BatchAlreadyUndoneError, BatchNotFoundError
from ragnest.models.domain import BatchDetail, BatchInfo, BatchStatus

if TYPE_CHECKING:
    from ragnest.db.backend import DatabaseBackend

logger = logging.getLogger(__name__)


class BatchRepository(BaseRepository):
    """Repository for batches table operations.

    Operates on the SQLite state backend.  Does **not** touch the chunks
    table (which lives on the vector backend).
    """

    def __init__(self, backend: DatabaseBackend) -> None:
        super().__init__(backend)

    def create(
        self, kb_name: str, description: str | None = None
    ) -> str:
        """Create a new batch and return its id."""
        batch_id = str(uuid.uuid4())[:12]
        with self._backend.cursor() as cur:
            cur.execute(
                "INSERT INTO batches "
                "(id, kb_name, description, status) "
                "VALUES (%s, %s, %s, %s)",
                (batch_id, kb_name, description, BatchStatus.PENDING),
            )
        logger.info("Created batch %s for KB '%s'", batch_id, kb_name)
        return batch_id

    def get_status(self, batch_id: str) -> BatchDetail:
        """Fetch detailed batch status including failed file info.

        Raises ``BatchNotFoundError`` if the batch does not exist.
        """
        with self._backend.cursor() as cur:
            cur.execute(
                "SELECT id, kb_name, description, status, total_files, "
                "processed_files, failed_files, skipped_files, "
                "total_chunks, created_at, completed_at "
                "FROM batches WHERE id = %s",
                (batch_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise BatchNotFoundError(batch_id)

            # Failed file details
            cur.execute(
                "SELECT file_path, error_message "
                "FROM ingestion_queue "
                "WHERE batch_id = %s AND status = 'failed'",
                (batch_id,),
            )
            failed = [
                {"file": r[0], "error": r[1] or ""} for r in cur.fetchall()
            ]

            # Pending count
            cur.execute(
                "SELECT COUNT(*) FROM ingestion_queue "
                "WHERE batch_id = %s AND status = 'pending'",
                (batch_id,),
            )
            pending_row = cur.fetchone()
            pending = int(pending_row[0]) if pending_row else 0

        return BatchDetail(
            id=row[0],
            kb_name=row[1],
            description=row[2],
            status=BatchStatus(row[3]),
            total_files=row[4],
            processed_files=row[5],
            failed_files=row[6],
            skipped_files=row[7],
            total_chunks=row[8],
            created_at=row[9],
            completed_at=row[10],
            pending_count=pending,
            failed_details=failed,
        )

    def get_kb_name(self, batch_id: str) -> str | None:
        """Return the kb_name for a batch, or None if not found."""
        with self._backend.cursor() as cur:
            cur.execute(
                "SELECT kb_name FROM batches WHERE id = %s",
                (batch_id,),
            )
            row = cur.fetchone()
        return row[0] if row else None

    def list_by_kb(
        self, kb_name: str, limit: int = 20
    ) -> list[BatchInfo]:
        """List recent batches for a knowledge base."""
        with self._backend.cursor() as cur:
            cur.execute(
                "SELECT id, kb_name, description, status, total_files, "
                "processed_files, failed_files, skipped_files, "
                "total_chunks, created_at, completed_at "
                "FROM batches WHERE kb_name = %s "
                "ORDER BY created_at DESC LIMIT %s",
                (kb_name, limit),
            )
            rows = cur.fetchall()
        return [
            BatchInfo(
                id=r[0],
                kb_name=r[1],
                description=r[2],
                status=BatchStatus(r[3]),
                total_files=r[4],
                processed_files=r[5],
                failed_files=r[6],
                skipped_files=r[7],
                total_chunks=r[8],
                created_at=r[9],
                completed_at=r[10],
            )
            for r in rows
        ]

    def undo(self, batch_id: str) -> list[str]:
        """Undo a batch: delete documents from state, mark as undone.

        Returns the list of document IDs that were deleted (so the caller
        can clean up chunks on the vector backend).

        Raises ``BatchNotFoundError`` or ``BatchAlreadyUndoneError``.
        """
        with self._backend.cursor() as cur:
            cur.execute(
                "SELECT kb_name, status FROM batches WHERE id = %s",
                (batch_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise BatchNotFoundError(batch_id)

            status: str = row[1]
            if status == BatchStatus.UNDONE:
                raise BatchAlreadyUndoneError(batch_id)

            # Collect document IDs before deleting
            cur.execute(
                "SELECT id FROM documents WHERE batch_id = %s",
                (batch_id,),
            )
            doc_ids = [r[0] for r in cur.fetchall()]

            # Delete documents from state
            cur.execute(
                "DELETE FROM documents WHERE batch_id = %s",
                (batch_id,),
            )

            # Mark batch as undone
            cur.execute(
                "UPDATE batches SET status = %s, completed_at = datetime('now') "
                "WHERE id = %s",
                (BatchStatus.UNDONE, batch_id),
            )

        logger.info("Undone batch %s (%d docs removed from state)", batch_id, len(doc_ids))
        return doc_ids

    def update_stats(
        self,
        batch_id: str,
        total_files: int | None = None,
        processed_files: int | None = None,
        failed_files: int | None = None,
        skipped_files: int | None = None,
        total_chunks: int | None = None,
    ) -> None:
        """Update batch statistics (called by the worker during processing)."""
        sets: list[str] = []
        params: list[int | str] = []

        if total_files is not None:
            sets.append("total_files = %s")
            params.append(total_files)
        if processed_files is not None:
            sets.append("processed_files = %s")
            params.append(processed_files)
        if failed_files is not None:
            sets.append("failed_files = %s")
            params.append(failed_files)
        if skipped_files is not None:
            sets.append("skipped_files = %s")
            params.append(skipped_files)
        if total_chunks is not None:
            sets.append("total_chunks = %s")
            params.append(total_chunks)

        if not sets:
            return

        params.append(batch_id)
        query = f"UPDATE batches SET {', '.join(sets)} WHERE id = %s"
        with self._backend.cursor() as cur:
            cur.execute(query, params)

    def mark_completed(self, batch_id: str) -> None:
        """Mark a batch as completed with a timestamp."""
        with self._backend.cursor() as cur:
            cur.execute(
                "UPDATE batches SET status = %s, completed_at = datetime('now') "
                "WHERE id = %s",
                (BatchStatus.COMPLETED, batch_id),
            )
        logger.info("Batch %s completed", batch_id)

    def mark_running(self, batch_id: str) -> None:
        """Mark a batch as running (best-effort, only if still pending)."""
        with self._backend.cursor() as cur:
            cur.execute(
                "UPDATE batches SET status = %s "
                "WHERE id = %s AND status = %s",
                (BatchStatus.RUNNING, batch_id, BatchStatus.PENDING),
            )
