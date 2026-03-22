"""System service — DB health, model listing, system information.

Reports status of both the SQLite state backend and vector backends.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ragnest.models.domain import DBStatus, SystemInfo
from ragnest.services.file_reader import list_supported_formats

if TYPE_CHECKING:
    from ragnest.app import BackendRegistry
    from ragnest.config import AppSettings
    from ragnest.db.backend import DatabaseBackend
    from ragnest.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

_VERSION = "0.1.0"


class SystemService:
    """System-level operations: DB health, config info, model listing."""

    def __init__(
        self,
        state_backend: DatabaseBackend,
        registry: BackendRegistry,
        embedding_service: EmbeddingService,
        settings: AppSettings,
    ) -> None:
        self._state = state_backend
        self._registry = registry
        self._embedding_service = embedding_service
        self._settings = settings

    def _state_status(self) -> DBStatus:
        """Check connectivity and stats for the SQLite state backend."""
        try:
            with self._state.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM knowledge_bases")
                row = cur.fetchone()
                total_kbs = int(row[0]) if row else 0

                cur.execute("SELECT COUNT(*) FROM documents")
                row = cur.fetchone()
                total_docs = int(row[0]) if row else 0

                table_sizes: dict[str, int] = {}
                for table in (
                    "knowledge_bases",
                    "documents",
                    "batches",
                    "ingestion_queue",
                    "watch_paths",
                ):
                    cur.execute(f"SELECT COUNT(*) FROM {table}")
                    trow = cur.fetchone()
                    table_sizes[table] = int(trow[0]) if trow else 0

            return DBStatus(
                connected=True,
                backend="sqlite (state)",
                total_documents=total_docs,
                total_chunks=0,
                total_kbs=total_kbs,
                table_sizes=table_sizes,
            )
        except Exception:
            logger.exception("SQLite state status check failed")
            return DBStatus(
                connected=False,
                backend="sqlite (state)",
            )

    def _vector_backend_status(self, name: str) -> DBStatus:
        """Check connectivity and stats for a single vector backend."""
        backend = self._registry.get(name)
        db_settings = self._settings.databases.get(name)
        backend_type = db_settings.backend if db_settings else "unknown"
        try:
            with backend.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM chunks")
                row = cur.fetchone()
                total_chunks = int(row[0]) if row else 0

                table_sizes: dict[str, int] = {"chunks": total_chunks}

            return DBStatus(
                connected=True,
                backend=f"{name} ({backend_type})",
                total_documents=0,
                total_chunks=total_chunks,
                total_kbs=0,
                table_sizes=table_sizes,
            )
        except Exception:
            logger.exception("Vector backend status check failed for '%s'", name)
            return DBStatus(
                connected=False,
                backend=f"{name} ({backend_type})",
            )

    def db_status(self) -> DBStatus:
        """Check database connectivity and gather aggregate statistics.

        Aggregates state from SQLite and chunk counts from vector backends.
        """
        state = self._state_status()

        # Aggregate vector backend stats
        total_chunks = 0
        all_connected = state.connected
        backend_labels = [state.backend]
        all_table_sizes = dict(state.table_sizes)

        for name in self._registry.names:
            vec_status = self._vector_backend_status(name)
            backend_labels.append(vec_status.backend)
            if not vec_status.connected:
                all_connected = False
                continue
            total_chunks += vec_status.total_chunks
            for table, count in vec_status.table_sizes.items():
                all_table_sizes[table] = all_table_sizes.get(table, 0) + count

        return DBStatus(
            connected=all_connected,
            backend=", ".join(backend_labels),
            total_documents=state.total_documents,
            total_chunks=total_chunks,
            total_kbs=state.total_kbs,
            table_sizes=all_table_sizes,
        )

    def list_models(self) -> list[str]:
        """List available Ollama embedding models."""
        return self._embedding_service.list_models()

    def system_info(self) -> SystemInfo:
        """Gather overall system information."""
        db = self.db_status()
        configured_kbs = sorted(self._settings.knowledge_bases.keys())
        return SystemInfo(
            version=_VERSION,
            db_status=db,
            ollama_url=self._settings.ollama.base_url,
            configured_kbs=configured_kbs,
            supported_formats=list_supported_formats(),
            configured_backends=self._registry.names,
        )
