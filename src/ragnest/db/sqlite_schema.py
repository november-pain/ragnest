"""SQLite schema for local state — all tables except chunks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragnest.db.backends.sqlite import SQLiteBackend

logger = logging.getLogger(__name__)

STATE_SCHEMA_SQL = """
-- Knowledge base registry (local state)
CREATE TABLE IF NOT EXISTS knowledge_bases (
    name TEXT PRIMARY KEY,
    description TEXT,
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    chunk_size INTEGER DEFAULT 1000,
    chunk_overlap INTEGER DEFAULT 200,
    index_type TEXT DEFAULT 'hnsw',
    backend TEXT DEFAULT 'default',
    external INTEGER DEFAULT 0,
    mode TEXT DEFAULT 'read_write',
    created_at TEXT DEFAULT (datetime('now')),
    document_count INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0
);

-- Watch paths: directories a KB auto-ingests from
CREATE TABLE IF NOT EXISTS watch_paths (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kb_name TEXT NOT NULL REFERENCES knowledge_bases(name) ON DELETE CASCADE,
    dir_path TEXT NOT NULL,
    recursive INTEGER DEFAULT 1,
    enabled INTEGER DEFAULT 1,
    file_patterns TEXT DEFAULT '*',
    created_at TEXT DEFAULT (datetime('now')),
    last_scanned_at TEXT,
    UNIQUE(kb_name, dir_path)
);

-- Batches: group of files ingested together (for undo)
CREATE TABLE IF NOT EXISTS batches (
    id TEXT PRIMARY KEY,
    kb_name TEXT NOT NULL REFERENCES knowledge_bases(name) ON DELETE CASCADE,
    description TEXT,
    source_dir TEXT,
    status TEXT DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'completed', 'failed', 'undone')),
    total_files INTEGER DEFAULT 0,
    processed_files INTEGER DEFAULT 0,
    failed_files INTEGER DEFAULT 0,
    skipped_files INTEGER DEFAULT 0,
    total_chunks INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_batches_kb ON batches(kb_name);
CREATE INDEX IF NOT EXISTS idx_batches_status ON batches(status);

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    kb_name TEXT NOT NULL REFERENCES knowledge_bases(name) ON DELETE CASCADE,
    source_path TEXT,
    filename TEXT,
    file_type TEXT,
    content_hash TEXT,
    file_mtime REAL,
    file_size INTEGER,
    chunk_count INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}',
    ingested_at TEXT DEFAULT (datetime('now')),
    batch_id TEXT REFERENCES batches(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_kb ON documents(kb_name);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_documents_batch ON documents(batch_id);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_path);

-- Ingestion queue: per-file tracking within a batch
-- States: pending -> processing -> done | failed | skipped
CREATE TABLE IF NOT EXISTS ingestion_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kb_name TEXT NOT NULL REFERENCES knowledge_bases(name) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    batch_id TEXT NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
    status TEXT DEFAULT 'pending'
        CHECK (status IN ('pending', 'processing', 'done', 'failed', 'skipped')),
    error_message TEXT,
    chunk_count INTEGER DEFAULT 0,
    document_id TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    started_at TEXT,
    completed_at TEXT,
    retry_count INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_queue_status ON ingestion_queue(status);
CREATE INDEX IF NOT EXISTS idx_queue_batch ON ingestion_queue(batch_id);
CREATE INDEX IF NOT EXISTS idx_queue_kb ON ingestion_queue(kb_name);
"""


def init_sqlite_schema(backend: SQLiteBackend) -> None:
    """Execute the full SQLite DDL for state tables.

    Also runs migrations for existing databases that may be missing
    newer columns.
    """
    with backend.cursor() as cur:
        cur.execute(STATE_SCHEMA_SQL)

        # Migration: add external column if missing
        cur.execute(
            "SELECT 1 FROM pragma_table_info('knowledge_bases') "
            "WHERE name = 'external'"
        )
        if not cur.fetchone():
            cur.execute(
                "ALTER TABLE knowledge_bases ADD COLUMN external INTEGER DEFAULT 0"
            )
            logger.info("Migrated: added external column to knowledge_bases")

        # Migration: add mode column if missing
        cur.execute(
            "SELECT 1 FROM pragma_table_info('knowledge_bases') "
            "WHERE name = 'mode'"
        )
        if not cur.fetchone():
            cur.execute(
                "ALTER TABLE knowledge_bases "
                "ADD COLUMN mode TEXT DEFAULT 'read_write'"
            )
            logger.info("Migrated: added mode column to knowledge_bases")

    logger.info("SQLite state schema initialized")
