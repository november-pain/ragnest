"""Ingestion service — queue files, directories, and text for processing.

All queue and batch operations go to the SQLite state backend.
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from ragnest.db.repositories.batch import BatchRepository
from ragnest.db.repositories.queue import QueueRepository
from ragnest.exceptions import RagnestError
from ragnest.models.domain import BatchInfo, BatchStatus

if TYPE_CHECKING:
    from ragnest.db.backend import DatabaseBackend
    from ragnest.services.kb_service import KBService

logger = logging.getLogger(__name__)


class IngestService:
    """Queue items for the background worker to process."""

    def __init__(
        self,
        state_backend: DatabaseBackend,
        kb_service: KBService,
    ) -> None:
        self._state = state_backend
        self._kb_service = kb_service

    def _queue_repo(self) -> QueueRepository:
        return QueueRepository(self._state)

    def _batch_repo(self) -> BatchRepository:
        return BatchRepository(self._state)

    def queue_file(self, kb_name: str, file_path: str) -> BatchInfo:
        """Queue a single file for ingestion. Returns the batch info."""
        self._kb_service.check_writable_local(kb_name)

        path = Path(file_path)
        if not path.is_file():
            msg = f"File does not exist: {file_path}"
            raise RagnestError(msg)

        queue_repo = self._queue_repo()
        batch_repo = self._batch_repo()

        batch_id = batch_repo.create(
            kb_name, description=f"File: {path.name}"
        )
        queued = queue_repo.enqueue_file(
            kb_name, str(path.absolute()), batch_id
        )
        total = 1 if queued else 0
        batch_repo.update_stats(batch_id, total_files=total)

        logger.info(
            "Queued file '%s' for KB '%s' (batch %s)",
            path.name, kb_name, batch_id,
        )
        return BatchInfo(
            id=batch_id,
            kb_name=kb_name,
            description=f"File: {path.name}",
            status=BatchStatus.PENDING,
            total_files=total,
            processed_files=0,
            failed_files=0,
            skipped_files=0,
            total_chunks=0,
        )

    def queue_directory(
        self,
        kb_name: str,
        dir_path: str,
        recursive: bool = True,
        file_patterns: str = "*",
    ) -> BatchInfo:
        """Queue all matching files in a directory. Returns the batch info."""
        self._kb_service.check_writable_local(kb_name)

        path = Path(dir_path)
        if not path.is_dir():
            msg = f"Not a directory: {dir_path}"
            raise RagnestError(msg)

        queue_repo = self._queue_repo()
        batch_repo = self._batch_repo()

        batch_id = batch_repo.create(
            kb_name, description=f"Directory: {dir_path}"
        )
        queued = queue_repo.enqueue_directory(
            kb_name, dir_path, batch_id,
            recursive=recursive, file_patterns=file_patterns,
        )

        logger.info(
            "Queued %d files from '%s' for KB '%s' (batch %s)",
            queued, dir_path, kb_name, batch_id,
        )
        return BatchInfo(
            id=batch_id,
            kb_name=kb_name,
            description=f"Directory: {dir_path}",
            status=BatchStatus.PENDING,
            total_files=queued,
            processed_files=0,
            failed_files=0,
            skipped_files=0,
            total_chunks=0,
        )

    def queue_text(
        self,
        kb_name: str,
        text: str,
        source_name: str = "manual_entry",
    ) -> BatchInfo:
        """Write text to a temp file and queue it for ingestion."""
        self._kb_service.check_writable(kb_name)

        tmp_dir = Path(tempfile.gettempdir()) / "ragnest"
        tmp_dir.mkdir(exist_ok=True)
        tmp_file = tmp_dir / f"{source_name}_{uuid.uuid4().hex[:8]}.txt"
        tmp_file.write_text(text, encoding="utf-8")

        queue_repo = self._queue_repo()
        batch_repo = self._batch_repo()

        batch_id = batch_repo.create(
            kb_name, description=f"Text: {source_name}"
        )
        queue_repo.enqueue_file(
            kb_name, str(tmp_file), batch_id
        )
        batch_repo.update_stats(batch_id, total_files=1)

        logger.info(
            "Queued text '%s' for KB '%s' (batch %s, temp=%s)",
            source_name, kb_name, batch_id, tmp_file,
        )
        return BatchInfo(
            id=batch_id,
            kb_name=kb_name,
            description=f"Text: {source_name}",
            status=BatchStatus.PENDING,
            total_files=1,
            processed_files=0,
            failed_files=0,
            skipped_files=0,
            total_chunks=0,
        )
