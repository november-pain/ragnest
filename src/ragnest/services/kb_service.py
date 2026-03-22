"""Core knowledge base service — search, CRUD, watch paths, batches, documents.

Routes operations to the correct backend:
- SQLite state backend: KB config, documents, batches, queue, watch paths
- Vector backends (per-KB): chunk storage and similarity search
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from ragnest.db.repositories.batch import BatchRepository
from ragnest.db.repositories.chunk import ChunkRepository
from ragnest.db.repositories.document import DocumentRepository
from ragnest.db.repositories.knowledge_base import KBRepository
from ragnest.db.repositories.queue import QueueRepository
from ragnest.db.repositories.watch_path import WatchPathRepository
from ragnest.db.schema import create_vector_index
from ragnest.exceptions import (
    DocumentNotFoundError,
    KBAlreadyExistsError,
    KBNotFoundError,
    RagnestError,
)
from ragnest.models.domain import (
    BatchDetail,
    BatchInfo,
    DocumentInfo,
    KBConfig,
    KBStats,
    SearchResult,
    WatchPathInfo,
)

if TYPE_CHECKING:
    from ragnest.app import BackendRegistry
    from ragnest.db.backend import DatabaseBackend
    from ragnest.db.backends.sqlite import SQLiteBackend
    from ragnest.models.db import KBRow
    from ragnest.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class KBService:
    """Core service orchestrating knowledge base operations.

    All state (KB config, documents, batches, queue, watch paths) lives in
    the SQLite ``state_backend``.  Chunks live in the per-KB vector backend
    obtained from ``BackendRegistry``.
    """

    def __init__(
        self,
        state_backend: SQLiteBackend,
        registry: BackendRegistry,
        embedding_service: EmbeddingService,
    ) -> None:
        self._state: DatabaseBackend = state_backend
        self._registry = registry
        self._embedding_service = embedding_service

    # ── Repo Factories ──────────────────────────────────────────

    def _kb_repo(self) -> KBRepository:
        """KBRepository on SQLite state."""
        return KBRepository(self._state)

    def _doc_repo(self) -> DocumentRepository:
        """DocumentRepository on SQLite state."""
        return DocumentRepository(self._state)

    def _chunk_repo(self, backend_name: str = "default") -> ChunkRepository:
        """ChunkRepository on the vector backend for a given KB."""
        return ChunkRepository(self._registry.get(backend_name))

    def _batch_repo(self) -> BatchRepository:
        """BatchRepository on SQLite state."""
        return BatchRepository(self._state)

    def _queue_repo(self) -> QueueRepository:
        """QueueRepository on SQLite state."""
        return QueueRepository(self._state)

    def _watch_path_repo(self) -> WatchPathRepository:
        """WatchPathRepository on SQLite state."""
        return WatchPathRepository(self._state)

    # ── Helpers ──────────────────────────────────────────────────

    def _find_kb(self, kb_name: str) -> KBRow:
        """Find a KB in SQLite state.

        Raises:
            KBNotFoundError: If the KB is not found.
        """
        kb_repo = self._kb_repo()
        row = kb_repo.get(kb_name)
        if row is None:
            raise KBNotFoundError(kb_name)
        return row

    def _require_kb(self, kb_name: str) -> KBStats:
        """Fetch a KB from state or raise ``KBNotFoundError``."""
        row = self._find_kb(kb_name)
        mode = row.mode if row.mode in ("read_write", "read_only") else "read_write"
        return KBStats(
            name=row.name,
            description=row.description or "",
            model=row.model,
            dimensions=row.dimensions,
            document_count=row.document_count,
            chunk_count=row.chunk_count,
            backend=row.backend,
            external=row.external,
            mode=mode,  # type: ignore[arg-type]
        )

    def _backend_for_kb(self, kb_name: str) -> str:
        """Return the vector backend name that holds chunks for a given KB."""
        row = self._find_kb(kb_name)
        return row.backend

    def _require_writable(self, kb: KBStats) -> None:
        """Raise if the KB is read-only."""
        if kb.external and kb.mode == "read_only":
            msg = (
                f"KB '{kb.name}' is an external read-only knowledge base. "
                f"Write operations are not allowed."
            )
            raise RagnestError(msg)

    def _require_local(self, kb: KBStats) -> None:
        """Raise if the KB is external (any mode)."""
        if kb.external:
            msg = (
                f"KB '{kb.name}' is an external knowledge base. "
                f"Watch paths and worker operations are not supported."
            )
            raise RagnestError(msg)

    # ── Public Guard Methods ───────────────────────────────────

    def check_writable(self, kb_name: str) -> None:
        """Raise if the KB does not exist or is read-only."""
        kb = self._require_kb(kb_name)
        self._require_writable(kb)

    def check_writable_local(self, kb_name: str) -> None:
        """Raise if the KB does not exist, is read-only, or is external."""
        kb = self._require_kb(kb_name)
        self._require_writable(kb)
        self._require_local(kb)

    # ── Search & Retrieval ──────────────────────────────────────

    def search(
        self,
        kb_name: str,
        query: str,
        top_k: int = 5,
        threshold: float | None = None,
    ) -> list[SearchResult]:
        """Vector similarity search within a single knowledge base."""
        kb = self._require_kb(kb_name)
        backend_name = kb.backend
        provider = self._embedding_service.get_provider(kb.model)
        query_embedding = provider.embed_query(query)
        chunk_repo = self._chunk_repo(backend_name)
        results = chunk_repo.search(
            kb_name, query_embedding, kb.dimensions, top_k=top_k
        )
        if threshold is not None:
            results = [r for r in results if r.score >= threshold]
        return results

    def search_all(
        self,
        query: str,
        top_k_per_kb: int = 3,
        threshold: float | None = None,
    ) -> dict[str, list[SearchResult]]:
        """Search across all knowledge bases that have chunks."""
        kbs = self.list_kbs()
        all_results: dict[str, list[SearchResult]] = {}
        for kb in kbs:
            if kb.chunk_count > 0:
                results = self.search(
                    kb.name, query, top_k=top_k_per_kb, threshold=threshold
                )
                if results:
                    all_results[kb.name] = results
        return all_results

    def get_similar_documents(
        self,
        kb_name: str,
        document_id: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Find documents similar to a given document."""
        kb = self._require_kb(kb_name)
        backend_name = kb.backend
        doc_repo = self._doc_repo()

        doc = doc_repo.get(document_id)
        if doc is None:
            raise DocumentNotFoundError(document_id)

        query_text = doc.filename or document_id
        provider = self._embedding_service.get_provider(kb.model)
        query_embedding = provider.embed_query(query_text)

        chunk_repo = self._chunk_repo(backend_name)
        results = chunk_repo.search(
            kb_name,
            query_embedding,
            kb.dimensions,
            top_k=top_k + 5,
        )
        return [r for r in results if r.document_id != document_id][:top_k]

    # ── KB Lifecycle ────────────────────────────────────────────

    def create_kb(self, config: KBConfig) -> KBStats:
        """Create a new knowledge base.

        Writes config to SQLite state and creates vector index on the
        appropriate vector backend.
        """
        kb_repo = self._kb_repo()
        created = kb_repo.create(config)
        if not created:
            raise KBAlreadyExistsError(config.name)

        # Create vector index on the target backend
        if not config.external or config.mode == "read_write":
            vector_backend = self._registry.get(config.backend)
            create_vector_index(
                vector_backend, config.name, config.dimensions, config.index_type.value
            )

        return self._require_kb(config.name)

    def update_kb(
        self,
        kb_name: str,
        description: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> KBStats:
        """Update mutable fields of a knowledge base."""
        self._find_kb(kb_name)  # ensure exists
        kb_repo = self._kb_repo()
        kb_repo.update(
            kb_name,
            description=description,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return self._require_kb(kb_name)

    def delete_kb(self, kb_name: str) -> None:
        """Delete a knowledge base and all its data from both backends."""
        row = self._find_kb(kb_name)
        backend_name = row.backend

        # Delete chunks from vector backend
        chunk_repo = self._chunk_repo(backend_name)
        chunk_repo.delete_by_kb(kb_name)

        # Drop vector index
        try:
            from psycopg2 import sql  # noqa: PLC0415

            vector_backend = self._registry.get(backend_name)
            index_name = sql.Identifier(f"idx_chunks_{kb_name}_embedding")
            with vector_backend.cursor() as cur:
                cur.execute(
                    sql.SQL("DROP INDEX IF EXISTS {}").format(index_name)
                )
        except Exception:
            logger.warning("Could not drop vector index for KB '%s'", kb_name)

        # Delete from SQLite state (cascading removes docs, batches, etc.)
        kb_repo = self._kb_repo()
        kb_repo.delete(kb_name)
        logger.info("Deleted KB '%s' from state and vector backend '%s'", kb_name, backend_name)

    def get_kb(self, kb_name: str) -> KBStats:
        """Get a single knowledge base's stats."""
        return self._require_kb(kb_name)

    def list_kbs(self) -> list[KBStats]:
        """List all knowledge bases from SQLite state."""
        kb_repo = self._kb_repo()
        results: list[KBStats] = []
        for r in kb_repo.list_all():
            mode = r.mode if r.mode in ("read_write", "read_only") else "read_write"
            results.append(KBStats(
                name=r.name,
                description=r.description or "",
                model=r.model,
                dimensions=r.dimensions,
                document_count=r.document_count,
                chunk_count=r.chunk_count,
                backend=r.backend,
                external=r.external,
                mode=mode,  # type: ignore[arg-type]
            ))
        return results

    def init_kb(
        self,
        name: str,
        folder_path: str,
        model: str,
        description: str = "",
        dimensions: int = 1024,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        backend: str = "default",
        external: bool = False,
        mode: Literal["read_write", "read_only"] = "read_write",
        file_patterns: str = "*",
    ) -> KBStats:
        """Convenience: create KB + add watch path in one call.

        External KBs cannot have watch paths — raises an error.
        """
        config = KBConfig(
            name=name,
            description=description,
            model=model,
            dimensions=dimensions,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            backend=backend,
            external=external,
            mode=mode,
        )
        if config.external:
            msg = "External KBs cannot have watch paths. Use create_kb() instead."
            raise RagnestError(msg)

        kb = self.create_kb(config)
        self.add_watch_path(name, folder_path, file_patterns=file_patterns)
        logger.info(
            "Initialized KB '%s' with watch path '%s'", name, folder_path
        )
        return kb

    # ── Watch Paths ─────────────────────────────────────────────

    def add_watch_path(
        self,
        kb_name: str,
        dir_path: str,
        recursive: bool = True,
        file_patterns: str = "*",
    ) -> WatchPathInfo:
        """Add or update a watch path for a knowledge base.

        External KBs cannot have watch paths.
        """
        kb = self._require_kb(kb_name)
        self._require_local(kb)
        watch_path_repo = self._watch_path_repo()
        return watch_path_repo.add(
            kb_name,
            dir_path,
            recursive=recursive,
            file_patterns=file_patterns,
        )

    def remove_watch_path(self, kb_name: str, dir_path: str) -> None:
        """Remove a watch path."""
        self._find_kb(kb_name)  # ensure exists
        watch_path_repo = self._watch_path_repo()
        watch_path_repo.remove(kb_name, dir_path)

    def list_watch_paths(
        self, kb_name: str | None = None
    ) -> list[WatchPathInfo]:
        """List watch paths, optionally filtered by KB."""
        if kb_name is not None:
            self._find_kb(kb_name)  # ensure exists
        return self._watch_path_repo().list_all(kb_name)

    def pause_watch_path(self, kb_name: str, dir_path: str) -> None:
        """Disable a watch path (keep the record, stop scanning)."""
        self._find_kb(kb_name)  # ensure exists
        self._watch_path_repo().set_enabled(kb_name, dir_path, enabled=False)

    def resume_watch_path(self, kb_name: str, dir_path: str) -> None:
        """Re-enable a previously paused watch path."""
        self._find_kb(kb_name)  # ensure exists
        self._watch_path_repo().set_enabled(kb_name, dir_path, enabled=True)

    # ── Batches ─────────────────────────────────────────────────

    def batch_status(self, batch_id: str) -> BatchDetail:
        """Get detailed status for a batch (from SQLite state)."""
        return self._batch_repo().get_status(batch_id)

    def list_batches(self, kb_name: str) -> list[BatchInfo]:
        """List recent batches for a knowledge base."""
        self._find_kb(kb_name)  # ensure exists
        return self._batch_repo().list_by_kb(kb_name)

    def undo_batch(self, batch_id: str) -> None:
        """Undo a batch — remove documents from state and chunks from vector backend."""
        batch_repo = self._batch_repo()

        # Get KB name for this batch to determine vector backend
        kb_name = batch_repo.get_kb_name(batch_id)
        if kb_name is None:
            from ragnest.exceptions import BatchNotFoundError  # noqa: PLC0415

            raise BatchNotFoundError(batch_id)

        row = self._find_kb(kb_name)
        backend_name = row.backend

        # Undo in state — returns doc IDs for vector cleanup
        doc_ids = batch_repo.undo(batch_id)

        # Delete chunks from vector backend
        if doc_ids:
            chunk_repo = self._chunk_repo(backend_name)
            chunk_repo.delete_by_documents(doc_ids)

        # Refresh aggregate counts
        self._refresh_kb_counts(kb_name, backend_name)

    # ── Documents ───────────────────────────────────────────────

    def list_documents(self, kb_name: str) -> list[DocumentInfo]:
        """List all documents in a knowledge base (from SQLite state)."""
        self._find_kb(kb_name)  # ensure exists
        return self._doc_repo().list_by_kb(kb_name)

    def delete_document(self, document_id: str) -> None:
        """Delete a document from state and its chunks from vector backend."""
        doc_repo = self._doc_repo()
        doc = doc_repo.get(document_id)
        if doc is None:
            raise DocumentNotFoundError(document_id)

        kb_name = doc.kb_name
        row = self._find_kb(kb_name)
        backend_name = row.backend

        # Delete chunks from vector backend first
        chunk_repo = self._chunk_repo(backend_name)
        chunk_repo.delete_by_document(document_id)

        # Delete document from state
        doc_repo.delete(document_id)

        # Refresh aggregate counts
        self._refresh_kb_counts(kb_name, backend_name)

    def add_chunks(
        self,
        kb_name: str,
        doc_id: str,
        chunks: list[str],
        filename: str | None = None,
        source_path: str | None = None,
    ) -> int:
        """Embed and store chunks for a document.

        Returns the number of chunks stored.
        """
        kb = self._require_kb(kb_name)
        self._require_writable(kb)
        backend_name = kb.backend
        provider = self._embedding_service.get_provider(kb.model)
        embeddings = provider.embed_batch(chunks)

        chunk_dicts = [
            {"content": text, "embedding": emb}
            for text, emb in zip(chunks, embeddings, strict=True)
        ]
        chunk_repo = self._chunk_repo(backend_name)
        count = chunk_repo.add_batch(
            kb_name, doc_id, chunk_dicts,
            filename=filename, source_path=source_path,
        )

        # Update document chunk count in state
        doc_repo = self._doc_repo()
        doc_repo.update_chunk_count(doc_id, count)

        # Refresh aggregate counts
        self._refresh_kb_counts(kb_name, backend_name)

        return count

    # ── Count Refresh ───────────────────────────────────────────

    def _refresh_kb_counts(self, kb_name: str, backend_name: str) -> None:
        """Refresh document_count and chunk_count in SQLite state."""
        doc_repo = self._doc_repo()
        chunk_repo = self._chunk_repo(backend_name)
        doc_count = doc_repo.count_by_kb(kb_name)
        chunk_count = chunk_repo.count_by_kb(kb_name)
        kb_repo = self._kb_repo()
        kb_repo.update_counts(kb_name, document_count=doc_count, chunk_count=chunk_count)
