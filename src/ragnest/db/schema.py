"""Vector schema management — chunks-only DDL, database creation, HNSW index."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

if TYPE_CHECKING:
    from ragnest.config import DBSettings
    from ragnest.db.backend import DatabaseBackend

logger = logging.getLogger(__name__)

VECTOR_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

-- Chunks table (vector storage with inline metadata)
-- No foreign keys — portable, self-contained vector store
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    kb_name TEXT NOT NULL,
    document_id TEXT NOT NULL,
    content TEXT NOT NULL,
    chunk_index INTEGER,
    metadata JSONB DEFAULT '{}',
    embedding vector,
    filename TEXT,
    source_path TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_kb ON chunks(kb_name);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
"""

# Legacy schema kept for backward compatibility during init_schema migration
_LEGACY_STATE_TABLES_SQL = """
-- Knowledge base registry
CREATE TABLE IF NOT EXISTS knowledge_bases (
    name TEXT PRIMARY KEY,
    description TEXT,
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    chunk_size INTEGER DEFAULT 1000,
    chunk_overlap INTEGER DEFAULT 200,
    index_type TEXT DEFAULT 'hnsw',
    backend TEXT DEFAULT 'default',
    created_at TIMESTAMP DEFAULT NOW(),
    document_count INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0
);

-- Watch paths: directories a KB auto-ingests from
CREATE TABLE IF NOT EXISTS watch_paths (
    id SERIAL PRIMARY KEY,
    kb_name TEXT NOT NULL REFERENCES knowledge_bases(name) ON DELETE CASCADE,
    dir_path TEXT NOT NULL,
    recursive BOOLEAN DEFAULT TRUE,
    enabled BOOLEAN DEFAULT TRUE,
    file_patterns TEXT DEFAULT '*',
    created_at TIMESTAMP DEFAULT NOW(),
    last_scanned_at TIMESTAMP,
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
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
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
    file_mtime DOUBLE PRECISION,
    file_size BIGINT,
    chunk_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    ingested_at TIMESTAMP DEFAULT NOW(),
    batch_id TEXT REFERENCES batches(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_kb ON documents(kb_name);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_documents_batch ON documents(batch_id);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_path);

-- Ingestion queue: per-file tracking within a batch
CREATE TABLE IF NOT EXISTS ingestion_queue (
    id SERIAL PRIMARY KEY,
    kb_name TEXT NOT NULL REFERENCES knowledge_bases(name) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    batch_id TEXT NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
    status TEXT DEFAULT 'pending'
        CHECK (status IN ('pending', 'processing', 'done', 'failed', 'skipped')),
    error_message TEXT,
    chunk_count INTEGER DEFAULT 0,
    document_id TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    retry_count INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_queue_status ON ingestion_queue(status);
CREATE INDEX IF NOT EXISTS idx_queue_batch ON ingestion_queue(batch_id);
CREATE INDEX IF NOT EXISTS idx_queue_kb ON ingestion_queue(kb_name);
"""


def create_database_if_not_exists(settings: DBSettings) -> None:
    """Create the application database if it does not already exist.

    Connects to the ``postgres`` maintenance database to check/create.
    Uses ``psycopg2.sql.Identifier`` for the database name.
    """
    conn = psycopg2.connect(
        host=settings.host,
        port=settings.port,
        user=settings.user,
        password=settings.password.get_secret_value(),
        dbname="postgres",
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (settings.name,),
            )
            if not cur.fetchone():
                cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(settings.name)))
                logger.info("Created database: %s", settings.name)
            else:
                logger.info("Database already exists: %s", settings.name)
    finally:
        conn.close()


def init_vector_schema(backend: DatabaseBackend) -> None:
    """Execute the vector-only DDL (chunks table + pgvector extension)."""
    with backend.cursor() as cur:
        cur.execute(VECTOR_SCHEMA_SQL)

        # Migration: add filename column to chunks if missing
        cur.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'chunks' AND column_name = 'filename'"
        )
        if not cur.fetchone():
            cur.execute("ALTER TABLE chunks ADD COLUMN filename TEXT")
            logger.info("Migrated: added filename column to chunks")

        # Migration: add source_path column to chunks if missing
        cur.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'chunks' AND column_name = 'source_path'"
        )
        if not cur.fetchone():
            cur.execute("ALTER TABLE chunks ADD COLUMN source_path TEXT")
            logger.info("Migrated: added source_path column to chunks")

    logger.info("Vector schema initialized")


def init_schema(backend: DatabaseBackend) -> None:
    """Execute the full DDL — legacy state tables + vector schema.

    Keeps backward compatibility for existing single-backend setups.
    New deployments should use ``init_vector_schema`` for vector backends
    and ``init_sqlite_schema`` for the local state database.
    """
    with backend.cursor() as cur:
        cur.execute(VECTOR_SCHEMA_SQL)
        cur.execute(_LEGACY_STATE_TABLES_SQL)

        # Migration: add index_type column if missing (existing DBs)
        cur.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'knowledge_bases' AND column_name = 'index_type'"
        )
        if not cur.fetchone():
            cur.execute("ALTER TABLE knowledge_bases ADD COLUMN index_type TEXT DEFAULT 'hnsw'")
            logger.info("Migrated: added index_type column")

        # Migration: add backend column if missing (existing DBs)
        cur.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'knowledge_bases' AND column_name = 'backend'"
        )
        if not cur.fetchone():
            cur.execute("ALTER TABLE knowledge_bases ADD COLUMN backend TEXT DEFAULT 'default'")
            logger.info("Migrated: added backend column")

        # Migration: add filename/source_path to chunks if missing
        cur.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'chunks' AND column_name = 'filename'"
        )
        if not cur.fetchone():
            cur.execute("ALTER TABLE chunks ADD COLUMN filename TEXT")
            logger.info("Migrated: added filename column to chunks")

        cur.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'chunks' AND column_name = 'source_path'"
        )
        if not cur.fetchone():
            cur.execute("ALTER TABLE chunks ADD COLUMN source_path TEXT")
            logger.info("Migrated: added source_path column to chunks")

    logger.info("Schema initialized")


def create_vector_index(
    backend: DatabaseBackend,
    kb_name: str,
    dimensions: int,
    index_type: str = "hnsw",
) -> None:
    """Create a vector similarity index for a knowledge base.

    Supports HNSW (dims <= 2000), IVFFlat (dims <= 2000), or none (exact scan).
    Uses ``psycopg2.sql.Identifier`` for the index name and
    ``sql.Literal`` for the dimensions — no f-string SQL.
    """
    if index_type == "none":
        logger.info("No index for KB '%s' (%dd) — using exact scan", kb_name, dimensions)
        return

    index_name = sql.Identifier(f"idx_chunks_{kb_name}_embedding")

    if index_type == "ivfflat":
        query = sql.SQL(
            "CREATE INDEX IF NOT EXISTS {} "
            "ON chunks USING ivfflat "
            "((embedding::vector({dimensions})) vector_cosine_ops) "
            "WITH (lists = 100) "
            "WHERE kb_name = {kb_name}"
        ).format(
            index_name,
            dimensions=sql.Literal(dimensions),
            kb_name=sql.Literal(kb_name),
        )
    else:
        query = sql.SQL(
            "CREATE INDEX IF NOT EXISTS {} "
            "ON chunks USING hnsw "
            "((embedding::vector({dimensions})) vector_cosine_ops) "
            "WHERE kb_name = {kb_name}"
        ).format(
            index_name,
            dimensions=sql.Literal(dimensions),
            kb_name=sql.Literal(kb_name),
        )

    with backend.cursor() as cur:
        cur.execute(query)
    logger.info("Created %s index for KB '%s' (%dd)", index_type.upper(), kb_name, dimensions)
