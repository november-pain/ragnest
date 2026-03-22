"""Domain models for Ragnest — Pydantic v2 models used across all layers."""

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

HNSW_MAX_DIMENSIONS = 2000


class IndexType(StrEnum):
    HNSW = "hnsw"
    IVFFLAT = "ivfflat"
    NONE = "none"


class BatchStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    UNDONE = "undone"


class QueueItemStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class KBConfig(BaseModel):
    """Knowledge base configuration with validated fields."""

    name: str = Field(min_length=1, max_length=63, pattern=r"^[a-z][a-z0-9_]*$")
    description: str = ""
    model: str
    dimensions: int = Field(gt=0, le=8192)
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    separator: str = "\n\n"
    index_type: IndexType = IndexType.HNSW
    backend: str = "default"
    external: bool = False
    mode: Literal["read_write", "read_only"] = "read_write"

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info: Any) -> int:
        if "chunk_size" in info.data and v >= info.data["chunk_size"]:
            msg = "chunk_overlap must be less than chunk_size"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def auto_select_index(self) -> "KBConfig":
        """Auto-select no-index when dimensions exceed HNSW limit."""
        if self.index_type == IndexType.HNSW and self.dimensions > HNSW_MAX_DIMENSIONS:
            self.index_type = IndexType.NONE
        return self


class KBStats(BaseModel):
    """Knowledge base summary with document/chunk counts."""

    name: str
    description: str
    model: str
    dimensions: int
    document_count: int
    chunk_count: int
    backend: str = "default"
    external: bool = False
    mode: Literal["read_write", "read_only"] = "read_write"


class SearchResult(BaseModel):
    """Single search result from vector similarity search."""

    content: str
    score: float
    document_id: str
    filename: str
    kb_name: str
    chunk_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchInfo(BaseModel):
    """Ingestion batch summary."""

    id: str
    kb_name: str
    description: str | None = None
    status: BatchStatus
    total_files: int
    processed_files: int
    failed_files: int
    skipped_files: int
    total_chunks: int
    created_at: datetime | None = None
    completed_at: datetime | None = None


class BatchDetail(BatchInfo):
    """Batch with additional detail for status inspection."""

    pending_count: int = 0
    failed_details: list[dict[str, str]] = Field(default_factory=list)


class DocumentInfo(BaseModel):
    """Document metadata."""

    id: str
    filename: str | None = None
    file_type: str | None = None
    chunk_count: int = 0
    ingested_at: datetime | None = None
    batch_id: str | None = None


class WatchPathInfo(BaseModel):
    """Registered watch path for auto-ingestion."""

    id: int
    kb_name: str
    dir_path: str
    recursive: bool = True
    enabled: bool = True
    file_patterns: str = "*"
    last_scanned_at: datetime | None = None


class WorkerStats(BaseModel):
    """Statistics from a worker run."""

    processed: int = 0
    failed: int = 0
    skipped: int = 0
    total_chunks: int = 0
    duration_seconds: float = 0.0


class WorkerStatus(BaseModel):
    """Current worker state."""

    queue_depth: int = 0
    last_run_at: datetime | None = None
    is_processing: bool = False
    current_kb: str | None = None


class DBStatus(BaseModel):
    """Database health and statistics."""

    connected: bool
    backend: str
    total_documents: int = 0
    total_chunks: int = 0
    total_kbs: int = 0
    table_sizes: dict[str, int] = Field(default_factory=dict)


class SystemInfo(BaseModel):
    """Overall system information."""

    version: str
    db_status: DBStatus
    ollama_url: str
    configured_kbs: list[str] = Field(default_factory=list)
    supported_formats: list[str] = Field(default_factory=list)
    configured_backends: list[str] = Field(default_factory=list)
