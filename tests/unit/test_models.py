"""Unit tests for Pydantic domain models — validation, construction, enums."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ragnest.models.domain import (
    BatchInfo,
    BatchStatus,
    DBStatus,
    KBConfig,
    KBStats,
    QueueItemStatus,
    SearchResult,
    SystemInfo,
    WorkerStats,
    WorkerStatus,
)

# -- KBConfig name validation --


class TestKBConfigNameValidation:
    """KBConfig.name must match ^[a-z][a-z0-9_]*$."""

    def test_valid_name_simple(self) -> None:
        config = KBConfig(name="myknowledge", model="bge-m3", dimensions=1024)
        assert config.name == "myknowledge"

    def test_valid_name_with_underscores_and_digits(self) -> None:
        config = KBConfig(name="kb_v2_test", model="bge-m3", dimensions=1024)
        assert config.name == "kb_v2_test"

    def test_rejects_name_with_spaces(self) -> None:
        with pytest.raises(ValidationError, match="name"):
            KBConfig(name="my kb", model="bge-m3", dimensions=1024)

    def test_rejects_name_with_uppercase(self) -> None:
        with pytest.raises(ValidationError, match="name"):
            KBConfig(name="MyKB", model="bge-m3", dimensions=1024)

    def test_rejects_name_with_special_chars(self) -> None:
        with pytest.raises(ValidationError, match="name"):
            KBConfig(name="my-kb!", model="bge-m3", dimensions=1024)

    def test_rejects_name_starting_with_digit(self) -> None:
        with pytest.raises(ValidationError, match="name"):
            KBConfig(name="1kb", model="bge-m3", dimensions=1024)

    def test_rejects_sql_injection_attempt(self) -> None:
        with pytest.raises(ValidationError, match="name"):
            KBConfig(name="'; DROP TABLE --", model="bge-m3", dimensions=1024)

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValidationError, match="name"):
            KBConfig(name="", model="bge-m3", dimensions=1024)

    def test_rejects_name_with_hyphen(self) -> None:
        with pytest.raises(ValidationError, match="name"):
            KBConfig(name="my-kb", model="bge-m3", dimensions=1024)


# -- KBConfig chunk validation --


class TestKBConfigChunkValidation:
    """KBConfig chunk_overlap must be less than chunk_size."""

    def test_valid_overlap_less_than_size(self) -> None:
        config = KBConfig(
            name="kb",
            model="bge-m3",
            dimensions=1024,
            chunk_size=1000,
            chunk_overlap=200,
        )
        assert config.chunk_overlap < config.chunk_size

    def test_rejects_overlap_equal_to_size(self) -> None:
        with pytest.raises(ValidationError, match="chunk_overlap must be less than chunk_size"):
            KBConfig(
                name="kb",
                model="bge-m3",
                dimensions=1024,
                chunk_size=500,
                chunk_overlap=500,
            )

    def test_rejects_overlap_greater_than_size(self) -> None:
        with pytest.raises(ValidationError, match="chunk_overlap must be less than chunk_size"):
            KBConfig(
                name="kb",
                model="bge-m3",
                dimensions=1024,
                chunk_size=500,
                chunk_overlap=600,
            )

    def test_zero_overlap_is_valid(self) -> None:
        config = KBConfig(
            name="kb",
            model="bge-m3",
            dimensions=1024,
            chunk_size=500,
            chunk_overlap=0,
        )
        assert config.chunk_overlap == 0


# -- KBConfig dimensions validation --


class TestKBConfigDimensionsValidation:
    """KBConfig dimensions must be gt=0 and le=8192."""

    def test_valid_dimensions(self) -> None:
        config = KBConfig(name="kb", model="bge-m3", dimensions=1024)
        assert config.dimensions == 1024  # noqa: PLR2004

    def test_rejects_zero_dimensions(self) -> None:
        with pytest.raises(ValidationError, match="dimensions"):
            KBConfig(name="kb", model="bge-m3", dimensions=0)

    def test_rejects_negative_dimensions(self) -> None:
        with pytest.raises(ValidationError, match="dimensions"):
            KBConfig(name="kb", model="bge-m3", dimensions=-1)

    def test_rejects_dimensions_above_8192(self) -> None:
        with pytest.raises(ValidationError, match="dimensions"):
            KBConfig(name="kb", model="bge-m3", dimensions=8193)

    def test_max_dimensions_is_valid(self) -> None:
        config = KBConfig(name="kb", model="bge-m3", dimensions=8192)
        assert config.dimensions == 8192  # noqa: PLR2004

    def test_single_dimension_is_valid(self) -> None:
        config = KBConfig(name="kb", model="bge-m3", dimensions=1)
        assert config.dimensions == 1


# -- Domain model construction --


class TestSearchResultConstruction:
    """SearchResult constructs from valid data."""

    def test_construct_with_required_fields(self) -> None:
        result = SearchResult(
            content="text",
            score=0.95,
            document_id="doc-1",
            filename="file.md",
            kb_name="kb",
            chunk_index=0,
        )
        assert result.content == "text"
        assert result.score == 0.95  # noqa: PLR2004
        assert result.metadata == {}

    def test_construct_with_metadata(self) -> None:
        result = SearchResult(
            content="text",
            score=0.5,
            document_id="doc-1",
            filename="file.md",
            kb_name="kb",
            chunk_index=2,
            metadata={"source": "test"},
        )
        assert result.metadata == {"source": "test"}


class TestBatchInfoConstruction:
    """BatchInfo constructs from valid data."""

    def test_construct_with_all_fields(self) -> None:
        info = BatchInfo(
            id="batch-1",
            kb_name="kb",
            status=BatchStatus.COMPLETED,
            total_files=10,
            processed_files=8,
            failed_files=1,
            skipped_files=1,
            total_chunks=50,
        )
        assert info.id == "batch-1"
        assert info.status == BatchStatus.COMPLETED
        assert info.description is None

    def test_construct_with_optional_fields(self) -> None:
        info = BatchInfo(
            id="batch-2",
            kb_name="kb",
            description="My batch",
            status=BatchStatus.PENDING,
            total_files=0,
            processed_files=0,
            failed_files=0,
            skipped_files=0,
            total_chunks=0,
        )
        assert info.description == "My batch"
        assert info.created_at is None


class TestKBStatsConstruction:
    """KBStats constructs from valid data."""

    def test_construct_with_zero_counts(self) -> None:
        stats = KBStats(
            name="empty_kb",
            description="",
            model="bge-m3",
            dimensions=1024,
            document_count=0,
            chunk_count=0,
        )
        assert stats.document_count == 0
        assert stats.chunk_count == 0

    def test_construct_with_empty_description(self) -> None:
        stats = KBStats(
            name="kb",
            description="",
            model="bge-m3",
            dimensions=1024,
            document_count=5,
            chunk_count=42,
        )
        assert stats.description == ""


# -- Enum coverage --


class TestBatchStatusEnum:
    """BatchStatus enum has all expected values."""

    def test_all_statuses_exist(self) -> None:
        assert BatchStatus.PENDING == "pending"
        assert BatchStatus.RUNNING == "running"
        assert BatchStatus.COMPLETED == "completed"
        assert BatchStatus.FAILED == "failed"
        assert BatchStatus.UNDONE == "undone"

    def test_status_count(self) -> None:
        expected_count = 5
        assert len(BatchStatus) == expected_count


class TestQueueItemStatusEnum:
    """QueueItemStatus enum has all expected values."""

    def test_all_statuses_exist(self) -> None:
        assert QueueItemStatus.PENDING == "pending"
        assert QueueItemStatus.PROCESSING == "processing"
        assert QueueItemStatus.DONE == "done"
        assert QueueItemStatus.FAILED == "failed"
        assert QueueItemStatus.SKIPPED == "skipped"

    def test_status_count(self) -> None:
        expected_count = 5
        assert len(QueueItemStatus) == expected_count


# -- Worker / System models --


class TestWorkerStatsConstruction:
    """WorkerStats constructs with defaults and explicit values."""

    def test_defaults(self) -> None:
        stats = WorkerStats()
        assert stats.processed == 0
        assert stats.failed == 0
        assert stats.skipped == 0
        assert stats.total_chunks == 0
        assert stats.duration_seconds == 0.0

    def test_explicit_values(self) -> None:
        stats = WorkerStats(
            processed=10,
            failed=2,
            skipped=1,
            total_chunks=80,
            duration_seconds=12.5,
        )
        expected_processed = 10
        expected_duration = 12.5
        assert stats.processed == expected_processed
        assert stats.duration_seconds == expected_duration


class TestWorkerStatusConstruction:
    """WorkerStatus constructs with defaults."""

    def test_defaults(self) -> None:
        status = WorkerStatus()
        assert status.queue_depth == 0
        assert status.is_processing is False
        assert status.current_kb is None
        assert status.last_run_at is None

    def test_with_active_state(self) -> None:
        expected_depth = 5
        status = WorkerStatus(
            queue_depth=expected_depth,
            is_processing=True,
            current_kb="my_kb",
        )
        assert status.queue_depth == expected_depth
        assert status.current_kb == "my_kb"


class TestDBStatusConstruction:
    """DBStatus constructs with required and optional fields."""

    def test_connected(self) -> None:
        db = DBStatus(connected=True, backend="postgres")
        assert db.connected is True
        assert db.total_documents == 0
        assert db.table_sizes == {}

    def test_disconnected(self) -> None:
        db = DBStatus(connected=False, backend="postgres")
        assert db.connected is False

    def test_with_counts(self) -> None:
        expected_docs = 100
        expected_chunks = 500
        db = DBStatus(
            connected=True,
            backend="postgres",
            total_documents=expected_docs,
            total_chunks=expected_chunks,
            total_kbs=3,
            table_sizes={"chunks": expected_chunks},
        )
        assert db.total_documents == expected_docs
        assert db.table_sizes["chunks"] == expected_chunks


class TestSystemInfoConstruction:
    """SystemInfo constructs from nested DBStatus."""

    def test_construct(self) -> None:
        db = DBStatus(connected=True, backend="postgres")
        info = SystemInfo(
            version="0.1.0",
            db_status=db,
            ollama_url="http://localhost:11434",
        )
        assert info.version == "0.1.0"
        assert info.db_status.connected is True
        assert info.configured_kbs == []
        assert info.supported_formats == []

    def test_with_kbs_and_formats(self) -> None:
        db = DBStatus(connected=True, backend="postgres")
        info = SystemInfo(
            version="0.1.0",
            db_status=db,
            ollama_url="http://localhost:11434",
            configured_kbs=["kb_a", "kb_b"],
            supported_formats=[".md", ".txt"],
        )
        expected_kb_count = 2
        assert len(info.configured_kbs) == expected_kb_count
        assert ".md" in info.supported_formats
