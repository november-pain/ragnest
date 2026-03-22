"""Unit tests for KBService — search, CRUD, watch paths with mocked repositories."""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from ragnest.db.backends.sqlite import SQLiteBackend
from ragnest.db.repositories.chunk import ChunkRepository
from ragnest.db.repositories.document import DocumentRepository
from ragnest.db.repositories.knowledge_base import KBRepository
from ragnest.db.repositories.watch_path import WatchPathRepository
from ragnest.db.sqlite_schema import init_sqlite_schema
from ragnest.exceptions import KBAlreadyExistsError, KBNotFoundError, RagnestError
from ragnest.models.db import KBRow
from ragnest.models.domain import (
    KBConfig,
    KBStats,
    SearchResult,
    WatchPathInfo,
)
from ragnest.services.kb_service import KBService


def _make_mock_registry(
    mock_backend: MagicMock,
    backend_names: list[str] | None = None,
) -> MagicMock:
    """Create a mock BackendRegistry with the given backend."""
    from ragnest.app import BackendRegistry  # noqa: PLC0415

    if backend_names is None:
        backend_names = ["default"]
    registry = MagicMock(spec=BackendRegistry)
    registry.get.return_value = mock_backend
    registry.default = mock_backend
    type(registry).names = PropertyMock(return_value=backend_names)
    registry.all.return_value = dict.fromkeys(backend_names, mock_backend)
    return registry


def _make_kb_row(
    name: str = "test_kb",
    model: str = "bge-m3",
    dimensions: int = 1024,
    document_count: int = 5,
    chunk_count: int = 42,
    backend: str = "default",
    external: bool = False,
    mode: str = "read_write",
) -> KBRow:
    """Helper to create a KBRow for mock returns."""
    return KBRow(
        name=name,
        description="Test KB",
        model=model,
        dimensions=dimensions,
        chunk_size=1000,
        chunk_overlap=200,
        backend=backend,
        external=external,
        mode=mode,
        document_count=document_count,
        chunk_count=chunk_count,
    )


@pytest.fixture
def mock_vector_backend() -> MagicMock:
    """A mock vector database backend."""
    return MagicMock()


@pytest.fixture
def in_memory_state() -> SQLiteBackend:
    """In-memory SQLite state backend with schema initialized."""
    backend = SQLiteBackend(":memory:")
    init_sqlite_schema(backend)
    return backend


@pytest.fixture
def kb_service_with_mocks(
    in_memory_state: SQLiteBackend,
    mock_vector_backend: MagicMock,
    mock_embedding_service: MagicMock,
) -> tuple[KBService, dict[str, MagicMock]]:
    """KBService with in-memory SQLite state and mocked vector backend.

    Returns the service and a dict of mock repos for assertion.
    """
    registry = _make_mock_registry(mock_vector_backend)

    service = KBService(
        state_backend=in_memory_state,
        registry=registry,
        embedding_service=mock_embedding_service,
    )

    # Create mock repos for assertions
    mock_chunk = MagicMock(spec=ChunkRepository)
    repos = {
        "chunk": mock_chunk,
    }

    # Patch the chunk repo factory to return our mock
    service._chunk_repo = MagicMock(return_value=repos["chunk"])  # type: ignore[method-assign]  # noqa: SLF001

    return service, repos


# -- Search --


class TestKBServiceSearch:
    """KBService.search calls embedding service then chunk repo."""

    def test_search_returns_results(
        self,
        kb_service_with_mocks: tuple[KBService, dict[str, MagicMock]],
        mock_embedding_service: MagicMock,
        in_memory_state: SQLiteBackend,
    ) -> None:
        kb_service, repos = kb_service_with_mocks

        # Insert KB into state
        kb_repo = KBRepository(in_memory_state)
        kb_repo.create(KBConfig(name="test_kb", model="bge-m3", dimensions=1024))

        expected_results = [
            SearchResult(
                content="result",
                score=0.9,
                document_id="d1",
                filename="f.md",
                kb_name="test_kb",
                chunk_index=0,
            ),
        ]
        repos["chunk"].search.return_value = expected_results

        results = kb_service.search("test_kb", "test query")

        mock_embedding_service.get_provider.assert_called_once_with("bge-m3")
        provider = mock_embedding_service.get_provider.return_value
        provider.embed_query.assert_called_once_with("test query")
        repos["chunk"].search.assert_called_once()
        assert results == expected_results

    def test_search_with_threshold_filters_low_scores(
        self,
        kb_service_with_mocks: tuple[KBService, dict[str, MagicMock]],
        in_memory_state: SQLiteBackend,
    ) -> None:
        kb_service, repos = kb_service_with_mocks

        kb_repo = KBRepository(in_memory_state)
        kb_repo.create(KBConfig(name="test_kb", model="bge-m3", dimensions=1024))

        repos["chunk"].search.return_value = [
            SearchResult(
                content="high",
                score=0.9,
                document_id="d1",
                filename="f.md",
                kb_name="test_kb",
                chunk_index=0,
            ),
            SearchResult(
                content="low",
                score=0.2,
                document_id="d2",
                filename="g.md",
                kb_name="test_kb",
                chunk_index=0,
            ),
        ]

        results = kb_service.search("test_kb", "query", threshold=0.5)

        assert len(results) == 1
        expected_score = 0.9
        assert results[0].score == expected_score

    def test_search_raises_kb_not_found_when_missing(
        self,
        kb_service_with_mocks: tuple[KBService, dict[str, MagicMock]],
    ) -> None:
        kb_service, _repos = kb_service_with_mocks

        with pytest.raises(KBNotFoundError, match="nonexistent"):
            kb_service.search("nonexistent", "query")


# -- Create KB --


class TestKBServiceCreateKB:
    """KBService.create_kb validates config and writes to SQLite."""

    def test_create_kb_success(
        self,
        kb_service_with_mocks: tuple[KBService, dict[str, MagicMock]],
        sample_kb_config: KBConfig,
    ) -> None:
        kb_service, _repos = kb_service_with_mocks

        with patch("ragnest.services.kb_service.create_vector_index"):
            result = kb_service.create_kb(sample_kb_config)

        assert isinstance(result, KBStats)
        assert result.name == "test_kb"

    def test_create_kb_raises_already_exists(
        self,
        kb_service_with_mocks: tuple[KBService, dict[str, MagicMock]],
        sample_kb_config: KBConfig,
        in_memory_state: SQLiteBackend,
    ) -> None:
        kb_service, _repos = kb_service_with_mocks

        # Insert first
        kb_repo = KBRepository(in_memory_state)
        kb_repo.create(sample_kb_config)

        with pytest.raises(KBAlreadyExistsError, match="test_kb"):
            kb_service.create_kb(sample_kb_config)


# -- Delete KB --


class TestKBServiceDeleteKB:
    """KBService.delete_kb ensures KB exists before deleting."""

    def test_delete_existing_kb(
        self,
        kb_service_with_mocks: tuple[KBService, dict[str, MagicMock]],
        in_memory_state: SQLiteBackend,
    ) -> None:
        kb_service, repos = kb_service_with_mocks

        kb_repo = KBRepository(in_memory_state)
        kb_repo.create(KBConfig(name="test_kb", model="bge-m3", dimensions=1024))

        repos["chunk"].delete_by_kb.return_value = 0

        kb_service.delete_kb("test_kb")

        # KB should be gone from state
        assert kb_repo.get("test_kb") is None

    def test_delete_nonexistent_kb_raises(
        self,
        kb_service_with_mocks: tuple[KBService, dict[str, MagicMock]],
    ) -> None:
        kb_service, _repos = kb_service_with_mocks

        with pytest.raises(KBNotFoundError, match="missing_kb"):
            kb_service.delete_kb("missing_kb")


# -- List KBs --


class TestKBServiceListKBs:
    """KBService.list_kbs returns list of KBStats from SQLite state."""

    def test_list_returns_stats(
        self,
        kb_service_with_mocks: tuple[KBService, dict[str, MagicMock]],
        in_memory_state: SQLiteBackend,
    ) -> None:
        kb_service, _repos = kb_service_with_mocks

        kb_repo = KBRepository(in_memory_state)
        kb_repo.create(KBConfig(name="kb_a", model="bge-m3", dimensions=1024))
        kb_repo.create(KBConfig(name="kb_b", model="bge-m3", dimensions=1024))

        result = kb_service.list_kbs()

        expected_count = 2
        assert len(result) == expected_count
        assert all(isinstance(s, KBStats) for s in result)
        assert result[0].name == "kb_a"
        assert result[1].name == "kb_b"

    def test_list_empty(
        self,
        kb_service_with_mocks: tuple[KBService, dict[str, MagicMock]],
    ) -> None:
        kb_service, _repos = kb_service_with_mocks

        result = kb_service.list_kbs()

        assert result == []

    def test_list_kbs_includes_backend_name(
        self,
        kb_service_with_mocks: tuple[KBService, dict[str, MagicMock]],
        in_memory_state: SQLiteBackend,
    ) -> None:
        kb_service, _repos = kb_service_with_mocks

        kb_repo = KBRepository(in_memory_state)
        kb_repo.create(KBConfig(name="test_kb", model="bge-m3", dimensions=1024))

        result = kb_service.list_kbs()

        assert len(result) == 1
        assert result[0].backend == "default"


# -- Init KB --


class TestKBServiceInitKB:
    """KBService.init_kb creates KB and adds watch path."""

    def test_init_creates_kb_and_watch_path(
        self,
        kb_service_with_mocks: tuple[KBService, dict[str, MagicMock]],
        in_memory_state: SQLiteBackend,
    ) -> None:
        kb_service, _repos = kb_service_with_mocks

        with patch("ragnest.services.kb_service.create_vector_index"):
            result = kb_service.init_kb(
                name="test_kb",
                folder_path="/tmp/docs",
                model="bge-m3",
            )

        assert isinstance(result, KBStats)

        # Verify watch path was created
        wp_repo = WatchPathRepository(in_memory_state)
        paths = wp_repo.list_all("test_kb")
        assert len(paths) == 1
        assert paths[0].dir_path == "/tmp/docs"

    def test_init_rejects_external_kb(
        self,
        kb_service_with_mocks: tuple[KBService, dict[str, MagicMock]],
    ) -> None:
        kb_service, _repos = kb_service_with_mocks

        with pytest.raises(RagnestError, match="External KBs cannot have watch paths"):
            kb_service.init_kb(
                name="ext_kb",
                folder_path="/tmp/docs",
                model="bge-m3",
                external=True,
            )


# -- Watch paths --


class TestKBServiceWatchPaths:
    """KBService watch path operations delegate to repo."""

    def test_add_watch_path_checks_kb_exists(
        self,
        kb_service_with_mocks: tuple[KBService, dict[str, MagicMock]],
        in_memory_state: SQLiteBackend,
    ) -> None:
        kb_service, _repos = kb_service_with_mocks

        kb_repo = KBRepository(in_memory_state)
        kb_repo.create(KBConfig(name="test_kb", model="bge-m3", dimensions=1024))

        result = kb_service.add_watch_path("test_kb", "/data")

        assert isinstance(result, WatchPathInfo)

    def test_add_watch_path_raises_when_kb_missing(
        self,
        kb_service_with_mocks: tuple[KBService, dict[str, MagicMock]],
    ) -> None:
        kb_service, _repos = kb_service_with_mocks

        with pytest.raises(KBNotFoundError):
            kb_service.add_watch_path("missing", "/data")

    def test_add_watch_path_rejects_external_kb(
        self,
        kb_service_with_mocks: tuple[KBService, dict[str, MagicMock]],
        in_memory_state: SQLiteBackend,
    ) -> None:
        kb_service, _repos = kb_service_with_mocks

        # Create an external KB
        kb_repo = KBRepository(in_memory_state)
        kb_repo.create(
            KBConfig(
                name="ext_kb",
                model="bge-m3",
                dimensions=1024,
                external=True,
            )
        )

        with pytest.raises(RagnestError, match="external"):
            kb_service.add_watch_path("ext_kb", "/data")

    def test_list_watch_paths_delegates(
        self,
        kb_service_with_mocks: tuple[KBService, dict[str, MagicMock]],
        in_memory_state: SQLiteBackend,
    ) -> None:
        kb_service, _repos = kb_service_with_mocks

        kb_repo = KBRepository(in_memory_state)
        kb_repo.create(KBConfig(name="test_kb", model="bge-m3", dimensions=1024))

        wp_repo = WatchPathRepository(in_memory_state)
        wp_repo.add("test_kb", "/a")

        result = kb_service.list_watch_paths("test_kb")

        assert len(result) == 1
        assert result[0].dir_path == "/a"


# -- Batches --


class TestKBServiceBatches:
    """KBService batch operations delegate to batch_repo."""

    def test_list_batches_checks_kb_exists(
        self,
        kb_service_with_mocks: tuple[KBService, dict[str, MagicMock]],
        in_memory_state: SQLiteBackend,
    ) -> None:
        kb_service, _repos = kb_service_with_mocks

        kb_repo = KBRepository(in_memory_state)
        kb_repo.create(KBConfig(name="test_kb", model="bge-m3", dimensions=1024))

        result = kb_service.list_batches("test_kb")

        assert result == []


# -- Documents --


class TestKBServiceDocuments:
    """KBService document operations."""

    def test_list_documents_checks_kb_exists(
        self,
        kb_service_with_mocks: tuple[KBService, dict[str, MagicMock]],
        in_memory_state: SQLiteBackend,
    ) -> None:
        kb_service, _repos = kb_service_with_mocks

        kb_repo = KBRepository(in_memory_state)
        kb_repo.create(KBConfig(name="test_kb", model="bge-m3", dimensions=1024))

        result = kb_service.list_documents("test_kb")

        assert result == []


# -- External KB restrictions --


class TestExternalKBRestrictions:
    """External KBs have restrictions on write operations and watch paths."""

    def test_read_only_external_rejects_writes(
        self,
        kb_service_with_mocks: tuple[KBService, dict[str, MagicMock]],
        in_memory_state: SQLiteBackend,
    ) -> None:
        kb_service, _repos = kb_service_with_mocks

        # Create an external read-only KB
        kb_repo = KBRepository(in_memory_state)
        kb_repo.create(
            KBConfig(
                name="readonly_kb",
                model="bge-m3",
                dimensions=1024,
                external=True,
                mode="read_only",
            )
        )

        with pytest.raises(RagnestError, match="read-only"):
            kb_service.add_chunks("readonly_kb", "doc-1", ["chunk text"])

    def test_external_read_write_allows_chunks(
        self,
        kb_service_with_mocks: tuple[KBService, dict[str, MagicMock]],
        in_memory_state: SQLiteBackend,
    ) -> None:
        kb_service, repos = kb_service_with_mocks

        # Create an external read-write KB
        kb_repo = KBRepository(in_memory_state)
        kb_repo.create(
            KBConfig(
                name="rw_ext_kb",
                model="bge-m3",
                dimensions=1024,
                external=True,
                mode="read_write",
            )
        )

        # Create a document in state
        doc_repo = DocumentRepository(in_memory_state)
        doc_id = doc_repo.create(kb_name="rw_ext_kb", filename="test.txt")

        repos["chunk"].add_batch.return_value = 1
        repos["chunk"].count_by_kb.return_value = 1

        result = kb_service.add_chunks(
            "rw_ext_kb",
            doc_id,
            ["chunk text"],
            filename="test.txt",
        )

        assert result == 1

    def test_external_rejects_watch_paths(
        self,
        kb_service_with_mocks: tuple[KBService, dict[str, MagicMock]],
        in_memory_state: SQLiteBackend,
    ) -> None:
        kb_service, _repos = kb_service_with_mocks

        kb_repo = KBRepository(in_memory_state)
        kb_repo.create(
            KBConfig(
                name="ext_kb",
                model="bge-m3",
                dimensions=1024,
                external=True,
                mode="read_write",
            )
        )

        with pytest.raises(RagnestError, match="external"):
            kb_service.add_watch_path("ext_kb", "/data")
