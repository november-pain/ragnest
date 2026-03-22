"""Chunk repository — vector storage, search, and bulk operations."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

import psycopg2.extras
from psycopg2 import sql

from ragnest.db.repositories.base import BaseRepository
from ragnest.models.domain import SearchResult

if TYPE_CHECKING:
    from ragnest.db.backend import DatabaseBackend

logger = logging.getLogger(__name__)


class ChunkRepository(BaseRepository):
    """Repository for chunks table operations on the vector backend.

    Chunks store inline metadata (filename, source_path) so search is
    self-contained — no cross-system JOINs needed.
    """

    def __init__(self, backend: DatabaseBackend) -> None:
        super().__init__(backend)

    def search(
        self,
        kb_name: str,
        query_embedding: list[float],
        dimensions: int,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Vector similarity search using cosine distance.

        Uses ``psycopg2.sql.Literal`` for the dimensions cast so no
        f-string interpolation touches SQL.  Reads inline metadata from
        the chunks table — no JOIN to documents.
        """
        dim = sql.Literal(dimensions)
        query = sql.SQL(
            "SELECT c.content, "
            "1 - (c.embedding::vector({dim}) <=> %s::vector({dim})) AS score, "
            "c.document_id, COALESCE(c.filename, ''), c.kb_name, "
            "c.chunk_index, c.metadata "
            "FROM chunks c "
            "WHERE c.kb_name = %s "
            "ORDER BY c.embedding::vector({dim}) <=> %s::vector({dim}) "
            "LIMIT %s"
        ).format(dim=dim)

        with self._backend.cursor() as cur:
            cur.execute(query, (query_embedding, kb_name, query_embedding, top_k))
            rows = cur.fetchall()

        return [
            SearchResult(
                content=row[0],
                score=float(row[1]),
                document_id=row[2],
                filename=row[3],
                kb_name=row[4],
                chunk_index=row[5],
                metadata=row[6] if row[6] else {},
            )
            for row in rows
        ]

    def add_batch(
        self,
        kb_name: str,
        document_id: str,
        chunks: list[dict[str, Any]],
        filename: str | None = None,
        source_path: str | None = None,
    ) -> int:
        """Insert a batch of chunks for a document with inline metadata.

        Each dict in *chunks* must have keys: ``content``, ``embedding``,
        and optionally ``metadata``.  *filename* and *source_path* are
        stored inline on each chunk for self-contained search results.

        Returns the number of chunks inserted.
        """
        with self._backend.cursor() as cur:
            for i, chunk in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                meta = chunk.get("metadata", {})
                cur.execute(
                    "INSERT INTO chunks "
                    "(id, kb_name, document_id, content, chunk_index, "
                    "metadata, embedding, filename, source_path) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        chunk_id,
                        kb_name,
                        document_id,
                        chunk["content"],
                        i,
                        psycopg2.extras.Json(meta),
                        chunk["embedding"],
                        filename,
                        source_path,
                    ),
                )

        logger.debug(
            "Inserted %d chunks for document %s in KB '%s'",
            len(chunks),
            document_id,
            kb_name,
        )
        return len(chunks)

    def delete_by_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a document. Returns rows deleted."""
        with self._backend.cursor() as cur:
            cur.execute(
                "DELETE FROM chunks WHERE document_id = %s",
                (document_id,),
            )
            return int(cur.rowcount)

    def delete_by_documents(self, document_ids: list[str]) -> int:
        """Delete all chunks belonging to multiple documents. Returns rows deleted."""
        if not document_ids:
            return 0
        total = 0
        with self._backend.cursor() as cur:
            for doc_id in document_ids:
                cur.execute(
                    "DELETE FROM chunks WHERE document_id = %s",
                    (doc_id,),
                )
                total += int(cur.rowcount)
        return total

    def delete_by_kb(self, kb_name: str) -> int:
        """Delete all chunks in a knowledge base. Returns rows deleted."""
        with self._backend.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE kb_name = %s", (kb_name,))
            return int(cur.rowcount)

    def count_by_kb(self, kb_name: str) -> int:
        """Count total chunks in a knowledge base."""
        with self._backend.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM chunks WHERE kb_name = %s",
                (kb_name,),
            )
            row = cur.fetchone()
        return int(row[0]) if row else 0
