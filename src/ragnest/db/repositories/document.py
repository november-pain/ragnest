"""Document repository — CRUD for the documents table (SQLite state)."""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

from ragnest.db.repositories.base import BaseRepository
from ragnest.models.db import DocumentRow
from ragnest.models.domain import DocumentInfo

if TYPE_CHECKING:
    from ragnest.db.backend import DatabaseBackend

logger = logging.getLogger(__name__)


class DocumentRepository(BaseRepository):
    """Repository for documents table operations.

    Operates on the SQLite state backend.  Does **not** touch the chunks
    table (which lives on the vector backend).  Aggregate count refreshes
    are handled by the service layer.
    """

    def __init__(self, backend: DatabaseBackend) -> None:
        super().__init__(backend)

    def create(
        self,
        kb_name: str,
        source_path: str | None = None,
        filename: str | None = None,
        file_type: str | None = None,
        content_hash: str | None = None,
        file_mtime: float | None = None,
        file_size: int | None = None,
        batch_id: str | None = None,
    ) -> str:
        """Insert a new document row. Returns the generated document id."""
        doc_id = str(uuid.uuid4())
        with self._backend.cursor() as cur:
            cur.execute(
                "INSERT INTO documents "
                "(id, kb_name, source_path, filename, file_type, "
                "content_hash, file_mtime, file_size, batch_id) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    doc_id,
                    kb_name,
                    source_path,
                    filename,
                    file_type,
                    content_hash,
                    file_mtime,
                    file_size,
                    batch_id,
                ),
            )
        logger.debug("Created document %s in KB '%s'", doc_id, kb_name)
        return doc_id

    def _parse_metadata(self, raw: Any) -> dict[str, Any] | None:
        """Parse metadata which may be a JSON string (SQLite) or dict (Postgres)."""
        if raw is None:
            return None
        if isinstance(raw, dict):
            return raw  # pyright: ignore[reportUnknownVariableType]
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)  # pyright: ignore[reportAny]
            except (json.JSONDecodeError, TypeError):
                return None
            else:
                if isinstance(parsed, dict):
                    result: dict[str, Any] = {str(k): v for k, v in parsed.items()}  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType,reportUnknownArgumentType]
                    return result
                return None
        return None

    def get(self, document_id: str) -> DocumentRow | None:
        """Fetch a single document by id."""
        with self._backend.cursor() as cur:
            cur.execute(
                "SELECT id, kb_name, source_path, filename, file_type, "
                "content_hash, file_mtime, file_size, chunk_count, "
                "metadata, ingested_at, batch_id "
                "FROM documents WHERE id = %s",
                (document_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return DocumentRow(
            id=row[0],
            kb_name=row[1],
            source_path=row[2],
            filename=row[3],
            file_type=row[4],
            content_hash=row[5],
            file_mtime=row[6],
            file_size=row[7],
            chunk_count=row[8],
            metadata=self._parse_metadata(row[9]),
            ingested_at=row[10],
            batch_id=row[11],
        )

    def list_by_kb(self, kb_name: str) -> list[DocumentInfo]:
        """List all documents in a knowledge base."""
        with self._backend.cursor() as cur:
            cur.execute(
                "SELECT id, filename, file_type, chunk_count, "
                "ingested_at, batch_id "
                "FROM documents WHERE kb_name = %s "
                "ORDER BY ingested_at DESC",
                (kb_name,),
            )
            rows = cur.fetchall()
        return [
            DocumentInfo(
                id=r[0],
                filename=r[1],
                file_type=r[2],
                chunk_count=r[3],
                ingested_at=r[4],
                batch_id=r[5],
            )
            for r in rows
        ]

    def delete(self, document_id: str) -> bool:
        """Delete a document from state. Returns True if it existed.

        Does NOT delete chunks — that is the service layer's job
        (chunks live on the vector backend).
        """
        with self._backend.cursor() as cur:
            cur.execute(
                "SELECT kb_name FROM documents WHERE id = %s",
                (document_id,),
            )
            row = cur.fetchone()
            if row is None:
                return False

            cur.execute("DELETE FROM documents WHERE id = %s", (document_id,))

        logger.info("Deleted document %s from state", document_id)
        return True

    def get_kb_name(self, document_id: str) -> str | None:
        """Return the kb_name for a document, or None if not found."""
        with self._backend.cursor() as cur:
            cur.execute(
                "SELECT kb_name FROM documents WHERE id = %s",
                (document_id,),
            )
            row = cur.fetchone()
        return row[0] if row else None

    def get_doc_ids_for_batch(self, batch_id: str) -> list[str]:
        """Return all document IDs belonging to a batch."""
        with self._backend.cursor() as cur:
            cur.execute(
                "SELECT id FROM documents WHERE batch_id = %s",
                (batch_id,),
            )
            rows = cur.fetchall()
        return [r[0] for r in rows]

    def delete_by_batch(self, batch_id: str) -> int:
        """Delete all documents in a batch. Returns count deleted."""
        with self._backend.cursor() as cur:
            cur.execute(
                "DELETE FROM documents WHERE batch_id = %s",
                (batch_id,),
            )
            return int(cur.rowcount)

    def find_by_hash(self, kb_name: str, content_hash: str) -> DocumentRow | None:
        """Find a document by content hash within a KB."""
        with self._backend.cursor() as cur:
            cur.execute(
                "SELECT id, kb_name, source_path, filename, file_type, "
                "content_hash, file_mtime, file_size, chunk_count, "
                "metadata, ingested_at, batch_id "
                "FROM documents "
                "WHERE kb_name = %s AND content_hash = %s",
                (kb_name, content_hash),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return DocumentRow(
            id=row[0],
            kb_name=row[1],
            source_path=row[2],
            filename=row[3],
            file_type=row[4],
            content_hash=row[5],
            file_mtime=row[6],
            file_size=row[7],
            chunk_count=row[8],
            metadata=self._parse_metadata(row[9]),
            ingested_at=row[10],
            batch_id=row[11],
        )

    def find_by_path(self, kb_name: str, source_path: str) -> DocumentRow | None:
        """Find a document by source path within a KB."""
        with self._backend.cursor() as cur:
            cur.execute(
                "SELECT id, kb_name, source_path, filename, file_type, "
                "content_hash, file_mtime, file_size, chunk_count, "
                "metadata, ingested_at, batch_id "
                "FROM documents "
                "WHERE kb_name = %s AND source_path = %s",
                (kb_name, source_path),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return DocumentRow(
            id=row[0],
            kb_name=row[1],
            source_path=row[2],
            filename=row[3],
            file_type=row[4],
            content_hash=row[5],
            file_mtime=row[6],
            file_size=row[7],
            chunk_count=row[8],
            metadata=self._parse_metadata(row[9]),
            ingested_at=row[10],
            batch_id=row[11],
        )

    def count_by_kb(self, kb_name: str) -> int:
        """Count total documents in a knowledge base."""
        with self._backend.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM documents WHERE kb_name = %s",
                (kb_name,),
            )
            row = cur.fetchone()
        return int(row[0]) if row else 0

    def update_chunk_count(self, document_id: str, chunk_count: int) -> None:
        """Update the chunk count for a document."""
        with self._backend.cursor() as cur:
            cur.execute(
                "UPDATE documents SET chunk_count = %s WHERE id = %s",
                (chunk_count, document_id),
            )
