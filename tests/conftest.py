"""Shared pytest fixtures for all test modules."""

from __future__ import annotations

from datetime import datetime as datetime  # noqa: PLC0414
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest
from pydantic import BaseModel as _BaseModel
from pydantic import SecretStr

import ragnest.models.db as _db_mod
import ragnest.models.domain as _domain_mod
from ragnest.config import AppSettings, DBSettings, DefaultsSettings, OllamaSettings, StateSettings
from ragnest.db.backends.sqlite import SQLiteBackend
from ragnest.db.repositories.batch import BatchRepository
from ragnest.db.repositories.chunk import ChunkRepository
from ragnest.db.repositories.document import DocumentRepository
from ragnest.db.repositories.knowledge_base import KBRepository
from ragnest.db.repositories.queue import QueueRepository
from ragnest.db.repositories.watch_path import WatchPathRepository
from ragnest.db.sqlite_schema import init_sqlite_schema
from ragnest.models.db import KBRow
from ragnest.models.domain import (
    BatchInfo,
    BatchStatus,
    KBConfig,
    KBStats,
    SearchResult,
    WatchPathInfo,
)
from ragnest.services.embedding_service import EmbeddingProvider, EmbeddingService

# Pydantic models use TYPE_CHECKING guard for datetime import; inject the real
# type into the module globals so model_rebuild() can resolve forward refs.
_domain_mod.datetime = datetime  # type: ignore[attr-defined]
_db_mod.datetime = datetime  # type: ignore[attr-defined]
for _model_cls in list(_domain_mod.__dict__.values()):
    if (
        isinstance(_model_cls, type)
        and issubclass(_model_cls, _BaseModel)
        and _model_cls is not _BaseModel
    ):
        _model_cls.model_rebuild()
for _model_cls in list(_db_mod.__dict__.values()):
    if (
        isinstance(_model_cls, type)
        and issubclass(_model_cls, _BaseModel)
        and _model_cls is not _BaseModel
    ):
        _model_cls.model_rebuild()

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_DOCS_DIR = FIXTURES_DIR / "sample_docs"


# -- Settings --


@pytest.fixture
def test_settings() -> AppSettings:
    """Minimal test settings with no external dependencies."""
    return AppSettings(
        databases={"default": DBSettings(name="ragnest_test", password=SecretStr("testpass"))},
        ollama=OllamaSettings(base_url="http://localhost:11434"),
        defaults=DefaultsSettings(chunk_size=500, chunk_overlap=100),
        state=StateSettings(path=":memory:"),
    )


# -- Mock Backend & Registry --


@pytest.fixture
def mock_backend() -> MagicMock:
    """Mock DatabaseBackend."""
    return MagicMock()


@pytest.fixture
def mock_registry(mock_backend: MagicMock) -> MagicMock:
    """Mock BackendRegistry with a single 'default' backend."""
    from ragnest.app import BackendRegistry  # noqa: PLC0415

    registry = MagicMock(spec=BackendRegistry)
    registry.get.return_value = mock_backend
    registry.default = mock_backend
    type(registry).names = PropertyMock(return_value=["default"])
    registry.all.return_value = {"default": mock_backend}
    return registry


@pytest.fixture
def state_backend() -> SQLiteBackend:
    """In-memory SQLite backend for state testing."""
    backend = SQLiteBackend(":memory:")
    init_sqlite_schema(backend)
    return backend


# -- Mock Repositories --


@pytest.fixture
def mock_kb_repo() -> MagicMock:
    """Mock KBRepository with spec-checked interface."""
    return MagicMock(spec=KBRepository)


@pytest.fixture
def mock_doc_repo() -> MagicMock:
    """Mock DocumentRepository with spec-checked interface."""
    return MagicMock(spec=DocumentRepository)


@pytest.fixture
def mock_chunk_repo() -> MagicMock:
    """Mock ChunkRepository with spec-checked interface."""
    return MagicMock(spec=ChunkRepository)


@pytest.fixture
def mock_batch_repo() -> MagicMock:
    """Mock BatchRepository with spec-checked interface."""
    return MagicMock(spec=BatchRepository)


@pytest.fixture
def mock_watch_path_repo() -> MagicMock:
    """Mock WatchPathRepository with spec-checked interface."""
    return MagicMock(spec=WatchPathRepository)


@pytest.fixture
def mock_queue_repo() -> MagicMock:
    """Mock QueueRepository with spec-checked interface."""
    return MagicMock(spec=QueueRepository)


# -- Mock Services --


@pytest.fixture
def mock_embedding_service() -> MagicMock:
    """Mock EmbeddingService that returns predictable embeddings."""
    service = MagicMock(spec=EmbeddingService)
    mock_provider = MagicMock()
    mock_provider.embed_query.return_value = [0.1] * 1024
    mock_provider.embed_batch.return_value = [[0.1] * 1024]
    service.get_provider.return_value = mock_provider
    return service


@pytest.fixture
def mock_embedding_provider() -> MagicMock:
    """Mock EmbeddingProvider for direct provider testing."""
    provider = MagicMock(spec=EmbeddingProvider)
    provider.embed_query.return_value = [0.1] * 1024
    provider.embed_batch.return_value = [[0.1] * 1024]
    return provider


# -- Sample Domain Objects --


@pytest.fixture
def sample_kb_config() -> KBConfig:
    """Valid KBConfig for testing."""
    return KBConfig(
        name="test_kb",
        description="A test knowledge base",
        model="bge-m3",
        dimensions=1024,
        chunk_size=1000,
        chunk_overlap=200,
    )


@pytest.fixture
def sample_kb_row() -> KBRow:
    """KBRow as returned by the database."""
    return KBRow(
        name="test_kb",
        description="A test knowledge base",
        model="bge-m3",
        dimensions=1024,
        chunk_size=1000,
        chunk_overlap=200,
        document_count=5,
        chunk_count=42,
    )


@pytest.fixture
def sample_kb_stats() -> KBStats:
    """KBStats domain model."""
    return KBStats(
        name="test_kb",
        description="A test knowledge base",
        model="bge-m3",
        dimensions=1024,
        document_count=5,
        chunk_count=42,
    )


@pytest.fixture
def sample_search_results() -> list[SearchResult]:
    """List of SearchResult objects for testing."""
    return [
        SearchResult(
            content="First result content",
            score=0.95,
            document_id="doc-1",
            filename="file1.md",
            kb_name="test_kb",
            chunk_index=0,
        ),
        SearchResult(
            content="Second result content",
            score=0.82,
            document_id="doc-2",
            filename="file2.txt",
            kb_name="test_kb",
            chunk_index=1,
        ),
        SearchResult(
            content="Third result content",
            score=0.45,
            document_id="doc-3",
            filename="file3.py",
            kb_name="test_kb",
            chunk_index=0,
        ),
    ]


@pytest.fixture
def sample_batch_info() -> BatchInfo:
    """BatchInfo domain model."""
    return BatchInfo(
        id="batch-001",
        kb_name="test_kb",
        description="Test batch",
        status=BatchStatus.COMPLETED,
        total_files=3,
        processed_files=2,
        failed_files=1,
        skipped_files=0,
        total_chunks=15,
    )


@pytest.fixture
def sample_watch_path() -> WatchPathInfo:
    """WatchPathInfo domain model."""
    return WatchPathInfo(
        id=1,
        kb_name="test_kb",
        dir_path="/tmp/test_docs",
        recursive=True,
        enabled=True,
        file_patterns="*",
    )
