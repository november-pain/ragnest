"""Worker service — background scanning, embedding, and queue processing.

Routes state operations (queue, batches, documents) to the SQLite state
backend, and chunk operations to the per-KB vector backend.
"""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ragnest.db.repositories.batch import BatchRepository
from ragnest.db.repositories.document import DocumentRepository
from ragnest.db.repositories.knowledge_base import KBRepository
from ragnest.db.repositories.queue import QueueRepository
from ragnest.db.repositories.watch_path import WatchPathRepository
from ragnest.models.domain import WorkerStats, WorkerStatus
from ragnest.services.file_reader import read_file

if TYPE_CHECKING:
    from ragnest.app import BackendRegistry
    from ragnest.config import AppSettings
    from ragnest.db.backend import DatabaseBackend
    from ragnest.models.db import QueueItemRow
    from ragnest.models.domain import WatchPathInfo
    from ragnest.services.embedding_service import EmbeddingService
    from ragnest.services.kb_service import KBService

logger = logging.getLogger(__name__)

_DRY_RUN_PREVIEW_LIMIT = 10


def _content_hash(content: str) -> str:
    """SHA-256 prefix for content deduplication."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class WorkerService:
    """Background worker: scans watch paths and processes the ingestion queue.

    Designed for resilience — commits per file, handles SIGINT/SIGTERM,
    and can resume on restart.

    All state operations use the SQLite backend.  Chunk writes use the
    per-KB vector backend via ``KBService.add_chunks``.
    """

    def __init__(
        self,
        kb_service: KBService,
        state_backend: DatabaseBackend,
        registry: BackendRegistry,
        embedding_service: EmbeddingService,
        settings: AppSettings,
    ) -> None:
        self._kb_service = kb_service
        self._state = state_backend
        self._registry = registry
        self._embedding_service = embedding_service
        self._settings = settings
        self._shutdown = threading.Event()
        self._is_processing = False
        self._current_kb: str | None = None
        self._last_run_at: float | None = None

    # ── State Repo Factories ─────────────────────────────────────

    def _queue_repo(self) -> QueueRepository:
        return QueueRepository(self._state)

    def _batch_repo(self) -> BatchRepository:
        return BatchRepository(self._state)

    def _doc_repo(self) -> DocumentRepository:
        return DocumentRepository(self._state)

    def _watch_path_repo(self) -> WatchPathRepository:
        return WatchPathRepository(self._state)

    def _kb_repo(self) -> KBRepository:
        return KBRepository(self._state)

    # ── Lifecycle ───────────────────────────────────────────────

    def request_shutdown(self) -> None:
        """Signal the worker to stop after the current file."""
        logger.info("Shutdown requested, finishing current file...")
        self._shutdown.set()

    def get_status(self) -> WorkerStatus:
        """Return current worker state."""
        queue_depth = self._queue_repo().get_pending_count()
        last_run = (
            datetime.fromtimestamp(self._last_run_at, tz=UTC)
            if self._last_run_at is not None
            else None
        )
        return WorkerStatus(
            queue_depth=queue_depth,
            last_run_at=last_run,
            is_processing=self._is_processing,
            current_kb=self._current_kb,
        )

    # ── Scanner ─────────────────────────────────────────────────

    def scan_watch_paths(
        self,
        kb_name: str | None = None,
        dry_run: bool = False,
    ) -> int:
        """Scan enabled watch paths for new or modified files.

        Returns the total number of files queued (or would-be queued in
        dry-run mode).
        """
        total_queued = 0

        watch_path_repo = self._watch_path_repo()
        doc_repo = self._doc_repo()

        if kb_name is not None:
            watch_paths = [
                wp for wp in watch_path_repo.get_active()
                if wp.kb_name == kb_name
            ]
        else:
            watch_paths = watch_path_repo.get_active()

        for wp in watch_paths:
            new_files = self._find_new_files(wp, doc_repo)
            if not new_files:
                logger.info(
                    "Watch path '%s' -> KB '%s': no new files",
                    wp.dir_path, wp.kb_name,
                )
                continue

            if dry_run:
                self._log_dry_run(wp, new_files)
                total_queued += len(new_files)
                continue

            total_queued += self._queue_new_files(wp, new_files)

        return total_queued

    def _find_new_files(
        self,
        wp: WatchPathInfo,
        doc_repo: DocumentRepository,
    ) -> list[str]:
        """Find new or modified files for a watch path."""
        dir_path = Path(wp.dir_path)
        if not dir_path.is_dir():
            logger.warning("Watch path does not exist: %s", wp.dir_path)
            return []

        patterns = [p.strip() for p in wp.file_patterns.split(",")]
        glob_pattern = "**/*" if wp.recursive else "*"
        files = sorted(f for f in dir_path.glob(glob_pattern) if f.is_file())

        # Filter by patterns
        if patterns != ["*"]:
            files = [
                f for f in files
                if any(fnmatch.fnmatch(f.name, p) for p in patterns)
            ]

        new_files: list[str] = []
        for f in files:
            fpath = str(f.absolute())

            # Check for modification by looking up the document
            existing_doc = doc_repo.find_by_path(wp.kb_name, fpath)
            if (
                existing_doc is not None
                and existing_doc.file_mtime is not None
                and abs(f.stat().st_mtime - existing_doc.file_mtime) < 1.0
            ):
                continue  # Not modified

            new_files.append(fpath)

        return new_files

    @staticmethod
    def _log_dry_run(
        wp: WatchPathInfo,
        new_files: list[str],
    ) -> None:
        """Log what would be queued in dry-run mode."""
        logger.info(
            "Watch path '%s' -> KB '%s': would queue %d files",
            wp.dir_path, wp.kb_name, len(new_files),
        )
        for fpath in new_files[:_DRY_RUN_PREVIEW_LIMIT]:
            logger.info("  %s", fpath)
        if len(new_files) > _DRY_RUN_PREVIEW_LIMIT:
            logger.info(
                "  ... and %d more", len(new_files) - _DRY_RUN_PREVIEW_LIMIT
            )

    def _queue_new_files(
        self,
        wp: WatchPathInfo,
        new_files: list[str],
    ) -> int:
        """Create batch and queue discovered files."""
        batch_repo = self._batch_repo()
        queue_repo = self._queue_repo()
        watch_path_repo = self._watch_path_repo()

        batch_id = batch_repo.create(
            wp.kb_name, description=f"Auto-scan: {wp.dir_path}"
        )
        batch_repo.update_stats(batch_id, total_files=len(new_files))

        for fpath in new_files:
            queue_repo.enqueue_file(wp.kb_name, fpath, batch_id)

        watch_path_repo.update_last_scanned(wp.id)

        logger.info(
            "Watch path '%s' -> KB '%s': queued %d files (batch %s)",
            wp.dir_path, wp.kb_name, len(new_files), batch_id,
        )
        return len(new_files)

    # ── Queue Processor ─────────────────────────────────────────

    def process_queue(
        self, kb_name: str | None = None
    ) -> WorkerStats:
        """Process all pending files in the ingestion queue.

        Per-file commits for resilience. Returns processing statistics.
        """
        self._is_processing = True
        self._current_kb = kb_name
        self._last_run_at = time.time()
        stats = WorkerStats()
        start_time = time.monotonic()

        try:
            queue_repo = self._queue_repo()
            batch_repo = self._batch_repo()
            doc_repo = self._doc_repo()

            while not self._shutdown.is_set():
                item = queue_repo.claim_next(kb_name)
                if item is None:
                    break

                batch_repo.mark_running(item.batch_id)
                self._process_single_item(
                    item, stats, queue_repo, batch_repo, doc_repo,
                )

            # Finalize completed batches
            self._finalize_batches(batch_repo)

        finally:
            self._is_processing = False
            self._current_kb = None

        elapsed = time.monotonic() - start_time
        stats.duration_seconds = round(elapsed, 2)
        return stats

    def _process_single_item(
        self,
        item: QueueItemRow,
        stats: WorkerStats,
        queue_repo: QueueRepository,
        batch_repo: BatchRepository,
        doc_repo: DocumentRepository,
    ) -> None:
        """Process a single queue item — read, chunk, embed, store."""
        from langchain_text_splitters import (  # noqa: PLC0415
            RecursiveCharacterTextSplitter,
        )

        try:
            file_path = Path(item.source_path)
            logger.info("Processing: %s -> %s", file_path.name, item.kb_name)

            if not file_path.exists():
                queue_repo.mark_failed(
                    item.id, f"File not found: {item.source_path}"
                )
                batch_repo.update_stats(item.batch_id, failed_files=1)
                stats.failed += 1
                return

            # Read file
            content = read_file(file_path)
            if not content.strip():
                queue_repo.mark_skipped(item.id, "Empty file")
                batch_repo.update_stats(item.batch_id, skipped_files=1)
                stats.skipped += 1
                return

            c_hash = _content_hash(content)

            # Check for duplicate content
            existing = doc_repo.find_by_hash(item.kb_name, c_hash)
            if existing is not None:
                queue_repo.mark_skipped(item.id, "Duplicate content")
                batch_repo.update_stats(item.batch_id, skipped_files=1)
                stats.skipped += 1
                return

            # Remove old version if file was modified
            old_doc = doc_repo.find_by_path(
                item.kb_name, item.source_path
            )
            if old_doc is not None:
                self._kb_service.delete_document(old_doc.id)
                logger.info("  Replacing old version of %s", file_path.name)

            # Get chunking settings
            kb_repo = self._kb_repo()
            kb_row = kb_repo.get(item.kb_name)
            chunk_size = (
                kb_row.chunk_size
                if kb_row is not None
                else self._settings.defaults.chunk_size
            )
            chunk_overlap = (
                kb_row.chunk_overlap
                if kb_row is not None
                else self._settings.defaults.chunk_overlap
            )

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            chunks = splitter.split_text(content)

            if not chunks:
                queue_repo.mark_skipped(item.id, "No chunks produced")
                batch_repo.update_stats(item.batch_id, skipped_files=1)
                stats.skipped += 1
                return

            # Create document record in SQLite state
            fstat = file_path.stat()
            doc_id = doc_repo.create(
                kb_name=item.kb_name,
                source_path=item.source_path,
                filename=file_path.name,
                file_type=file_path.suffix.lstrip("."),
                content_hash=c_hash,
                file_mtime=fstat.st_mtime,
                file_size=fstat.st_size,
                batch_id=item.batch_id,
            )

            # Embed and store chunks (writes to vector backend)
            num_chunks = self._kb_service.add_chunks(
                item.kb_name, doc_id, chunks,
                filename=file_path.name,
                source_path=item.source_path,
            )

            # Mark queue item as done
            queue_repo.mark_done(item.id, num_chunks)
            batch_repo.update_stats(
                item.batch_id,
                processed_files=1,
                total_chunks=num_chunks,
            )

            stats.processed += 1
            stats.total_chunks += num_chunks
            logger.info("  Done: %d chunks", num_chunks)

        except Exception:
            logger.exception("  Failed processing %s", item.source_path)
            error_msg = "Processing failed"
            queue_repo.mark_failed(item.id, error_msg)
            batch_repo.update_stats(item.batch_id, failed_files=1)
            stats.failed += 1

    def _finalize_batches(self, batch_repo: BatchRepository) -> None:  # noqa: ARG002
        """Mark batches as completed when they have no pending items left."""
        with self._state.cursor() as cur:
            cur.execute(
                "UPDATE batches SET status = 'completed', "
                "completed_at = datetime('now') "
                "WHERE status = 'running' AND id NOT IN ("
                "  SELECT DISTINCT batch_id FROM ingestion_queue "
                "  WHERE status IN ('pending', 'processing')"
                ")"
            )

    # ── Retry ───────────────────────────────────────────────────

    def retry_failed(self, kb_name: str | None = None) -> int:
        """Reset all failed queue items back to pending.

        Returns the number of items reset.
        """
        total = self._queue_repo().reset_failed(kb_name)
        logger.info("Re-queued %d failed items", total)
        return total

    # ── Run ─────────────────────────────────────────────────────

    def run(
        self,
        scan: bool = False,
        kb_name: str | None = None,
        retry: bool = False,
        dry_run: bool = False,
    ) -> None:
        """Main entry point for the worker CLI.

        Performs scan / retry / process according to flags.
        """
        if retry:
            self.retry_failed(kb_name)

        if scan:
            queued = self.scan_watch_paths(kb_name, dry_run=dry_run)
            logger.info("Scan complete: %d files queued", queued)
            if dry_run:
                return

        worker_stats = self.process_queue(kb_name)
        logger.info(
            "Worker done: %d processed, %d failed, %d skipped, "
            "%d chunks total (%.1fs)",
            worker_stats.processed,
            worker_stats.failed,
            worker_stats.skipped,
            worker_stats.total_chunks,
            worker_stats.duration_seconds,
        )

        if self._shutdown.is_set():
            logger.info(
                "Shutdown was requested — remaining items stay in queue"
            )
