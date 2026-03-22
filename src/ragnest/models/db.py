"""Row-level models mapping database tuples to typed objects."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class KBRow(BaseModel):
    """Row from knowledge_bases table."""

    name: str
    description: str | None = None
    model: str
    dimensions: int
    chunk_size: int
    chunk_overlap: int
    index_type: str = "hnsw"
    backend: str = "default"
    external: bool = False
    mode: str = "read_write"
    created_at: datetime | None = None
    document_count: int = 0
    chunk_count: int = 0


class DocumentRow(BaseModel):
    """Row from documents table."""

    id: str
    kb_name: str
    source_path: str | None = None
    filename: str | None = None
    file_type: str | None = None
    content_hash: str | None = None
    file_mtime: float | None = None
    file_size: int | None = None
    chunk_count: int = 0
    metadata: dict[str, Any] | None = None
    ingested_at: datetime | None = None
    batch_id: str | None = None


class ChunkRow(BaseModel):
    """Row from chunks table."""

    id: int
    document_id: str
    kb_name: str
    chunk_index: int
    content: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] | None = None


class BatchRow(BaseModel):
    """Row from batches table."""

    id: str
    kb_name: str
    description: str | None = None
    status: str
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    total_chunks: int = 0
    created_at: datetime | None = None
    completed_at: datetime | None = None


class QueueItemRow(BaseModel):
    """Row from ingestion_queue table."""

    id: int
    batch_id: str
    kb_name: str
    source_path: str
    status: str
    error_message: str | None = None
    chunk_count: int = 0
    queued_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class WatchPathRow(BaseModel):
    """Row from watch_paths table."""

    id: int
    kb_name: str
    dir_path: str
    recursive: bool = True
    file_patterns: str = "*"
    enabled: bool = True
    last_scanned_at: datetime | None = None
    created_at: datetime | None = None
