"""Knowledge base repository — CRUD operations on the knowledge_bases table."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ragnest.db.repositories.base import BaseRepository
from ragnest.models.db import KBRow

if TYPE_CHECKING:
    from ragnest.db.backend import DatabaseBackend
    from ragnest.models.domain import KBConfig

logger = logging.getLogger(__name__)


class KBRepository(BaseRepository):
    """Repository for knowledge_bases table operations.

    Works with both SQLite and PostgreSQL backends — uses only plain SQL
    with ``%s`` placeholders (translated by the SQLite cursor wrapper).
    """

    def __init__(self, backend: DatabaseBackend) -> None:
        super().__init__(backend)

    def create(self, config: KBConfig) -> bool:
        """Insert a new knowledge base. Returns True if created, False if exists."""
        with self._backend.cursor() as cur:
            # Check existence first for reliable created detection
            cur.execute(
                "SELECT 1 FROM knowledge_bases WHERE name = %s",
                (config.name,),
            )
            if cur.fetchone() is not None:
                return False

            cur.execute(
                "INSERT INTO knowledge_bases "
                "(name, description, model, dimensions, chunk_size, chunk_overlap, "
                "index_type, backend, external, mode) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    config.name,
                    config.description,
                    config.model,
                    config.dimensions,
                    config.chunk_size,
                    config.chunk_overlap,
                    config.index_type.value,
                    config.backend,
                    config.external,
                    config.mode,
                ),
            )

        logger.info(
            "Created KB '%s' (model=%s, dim=%d, external=%s, mode=%s)",
            config.name,
            config.model,
            config.dimensions,
            config.external,
            config.mode,
        )
        return True

    _SELECT_COLS = (
        "name, description, model, dimensions, "
        "chunk_size, chunk_overlap, "
        "COALESCE(index_type, 'hnsw'), "
        "COALESCE(backend, 'default'), "
        "COALESCE(external, 0), "
        "COALESCE(mode, 'read_write'), "
        "created_at, document_count, chunk_count"
    )

    def _row_to_model(self, r: tuple[Any, ...]) -> KBRow:
        return KBRow(
            name=r[0],
            description=r[1],
            model=r[2],
            dimensions=r[3],
            chunk_size=r[4],
            chunk_overlap=r[5],
            index_type=r[6],
            backend=r[7],
            external=bool(r[8]),
            mode=r[9],
            created_at=r[10],
            document_count=r[11],
            chunk_count=r[12],
        )

    def get(self, name: str) -> KBRow | None:
        """Fetch a single knowledge base by name."""
        with self._backend.cursor() as cur:
            cur.execute(
                f"SELECT {self._SELECT_COLS} FROM knowledge_bases WHERE name = %s",  # nosec B608
                (name,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_model(row)

    def list_all(self) -> list[KBRow]:
        """List all knowledge bases ordered by name."""
        with self._backend.cursor() as cur:
            cur.execute(f"SELECT {self._SELECT_COLS} FROM knowledge_bases ORDER BY name")  # nosec B608
            rows = cur.fetchall()
        return [self._row_to_model(r) for r in rows]

    def delete(self, name: str) -> bool:
        """Delete a KB and all associated state data. Returns True if deleted."""
        with self._backend.cursor() as cur:
            # Delete child rows explicitly for clarity
            cur.execute("DELETE FROM ingestion_queue WHERE kb_name = %s", (name,))
            cur.execute("DELETE FROM documents WHERE kb_name = %s", (name,))
            cur.execute("DELETE FROM batches WHERE kb_name = %s", (name,))
            cur.execute("DELETE FROM watch_paths WHERE kb_name = %s", (name,))
            cur.execute("DELETE FROM knowledge_bases WHERE name = %s", (name,))
            deleted = bool(cur.rowcount > 0)

        if deleted:
            logger.info("Deleted KB '%s' from state", name)
        return deleted

    def update(
        self,
        name: str,
        description: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> bool:
        """Update mutable fields of a KB. Returns True if the row existed."""
        sets: list[str] = []
        params: list[str | int] = []

        if description is not None:
            sets.append("description = %s")
            params.append(description)
        if chunk_size is not None:
            sets.append("chunk_size = %s")
            params.append(chunk_size)
        if chunk_overlap is not None:
            sets.append("chunk_overlap = %s")
            params.append(chunk_overlap)

        if not sets:
            return False

        params.append(name)
        query = f"UPDATE knowledge_bases SET {', '.join(sets)} WHERE name = %s"  # nosec B608
        with self._backend.cursor() as cur:
            cur.execute(query, params)
            updated = bool(cur.rowcount > 0)
        if updated:
            logger.info("Updated KB '%s'", name)
        return updated

    def update_counts(
        self,
        name: str,
        document_count: int | None = None,
        chunk_count: int | None = None,
    ) -> None:
        """Update aggregate counts for a KB in state."""
        sets: list[str] = []
        params: list[str | int] = []

        if document_count is not None:
            sets.append("document_count = %s")
            params.append(document_count)
        if chunk_count is not None:
            sets.append("chunk_count = %s")
            params.append(chunk_count)

        if not sets:
            return

        params.append(name)
        query = f"UPDATE knowledge_bases SET {', '.join(sets)} WHERE name = %s"  # nosec B608
        with self._backend.cursor() as cur:
            cur.execute(query, params)
