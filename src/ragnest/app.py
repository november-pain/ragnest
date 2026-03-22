"""Application container — wires all repositories and services together."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ragnest.db.backends import create_backend, create_state_backend
from ragnest.db.repositories.batch import BatchRepository
from ragnest.db.repositories.document import DocumentRepository
from ragnest.db.repositories.knowledge_base import KBRepository
from ragnest.db.repositories.queue import QueueRepository
from ragnest.db.repositories.watch_path import WatchPathRepository
from ragnest.db.schema import create_database_if_not_exists, init_vector_schema
from ragnest.db.sqlite_schema import init_sqlite_schema
from ragnest.exceptions import ConfigError
from ragnest.services.embedding_service import EmbeddingService
from ragnest.services.export_service import ExportService
from ragnest.services.ingest_service import IngestService
from ragnest.services.kb_service import KBService
from ragnest.services.system_service import SystemService
from ragnest.services.worker_service import WorkerService

if TYPE_CHECKING:
    from ragnest.config import AppSettings, DBSettings
    from ragnest.db.backend import DatabaseBackend
    from ragnest.db.backends.sqlite import SQLiteBackend

logger = logging.getLogger(__name__)


class BackendRegistry:
    """Manages multiple named vector database backends.

    Each backend is initialized with its own connection pool and vector schema.
    Services use the registry to route chunk operations to the correct backend.
    """

    def __init__(self, configs: dict[str, DBSettings]) -> None:
        self._backends: dict[str, DatabaseBackend] = {}
        for name, settings in configs.items():
            create_database_if_not_exists(settings)
            backend = create_backend(settings)
            init_vector_schema(backend)
            self._backends[name] = backend
            logger.info("Initialized vector backend '%s'", name)

    def get(self, name: str = "default") -> DatabaseBackend:
        """Get a backend by name, raising ``ConfigError`` if not found."""
        if name not in self._backends:
            available = list(self._backends)
            msg = f"Backend '{name}' not configured. Available: {available}"
            raise ConfigError(msg)
        return self._backends[name]

    @property
    def default(self) -> DatabaseBackend:
        """Return the first configured backend."""
        return next(iter(self._backends.values()))

    @property
    def names(self) -> list[str]:
        """List all configured backend names."""
        return list(self._backends)

    def all(self) -> dict[str, DatabaseBackend]:
        """Return a copy of all backends."""
        return dict(self._backends)

    def close(self) -> None:
        """Close all backend connection pools."""
        for name, backend in self._backends.items():
            backend.close()
            logger.info("Closed vector backend '%s'", name)


class Application:
    """Top-level container that owns backends, repositories, and services.

    Separates local state (SQLite) from vector storage (PostgreSQL backends).

    Usage::

        from ragnest.config import load_settings
        app = Application(load_settings())
        try:
            results = app.kb_service.search("my_kb", "query")
        finally:
            app.close()
    """

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

        # Local state (SQLite)
        self.state_backend: SQLiteBackend = create_state_backend(settings.state)
        init_sqlite_schema(self.state_backend)

        # Vector backends (per-KB routing)
        self.registry = BackendRegistry(settings.databases)

        # Backward-compatible single backend reference
        self.backend: DatabaseBackend = self.registry.default

        # State repositories (SQLite)
        self.kb_repo = KBRepository(self.state_backend)
        self.doc_repo = DocumentRepository(self.state_backend)
        self.batch_repo = BatchRepository(self.state_backend)
        self.queue_repo = QueueRepository(self.state_backend)
        self.watch_path_repo = WatchPathRepository(self.state_backend)

        # Services
        self.embedding_service = EmbeddingService(settings.ollama.base_url)

        self.kb_service = KBService(
            state_backend=self.state_backend,
            registry=self.registry,
            embedding_service=self.embedding_service,
        )

        self.ingest_service = IngestService(
            state_backend=self.state_backend,
            kb_service=self.kb_service,
        )

        self.worker_service = WorkerService(
            kb_service=self.kb_service,
            state_backend=self.state_backend,
            registry=self.registry,
            embedding_service=self.embedding_service,
            settings=settings,
        )

        self.export_service = ExportService(registry=self.registry)

        self.system_service = SystemService(
            state_backend=self.state_backend,
            registry=self.registry,
            embedding_service=self.embedding_service,
            settings=settings,
        )

        logger.info(
            "Application initialized — SQLite state + %d vector backend(s): %s",
            len(self.registry.names),
            ", ".join(self.registry.names),
        )

    def close(self) -> None:
        """Release all resources (connection pools, etc.)."""
        self.registry.close()
        self.state_backend.close()
        logger.info("Application closed")
