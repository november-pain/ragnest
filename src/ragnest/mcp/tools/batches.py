"""Batch and worker tools — status, undo, and worker management."""

# pyright: reportUnusedFunction=false

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated

from fastmcp.exceptions import ToolError
from pydantic import Field

from ragnest.exceptions import (
    BatchAlreadyUndoneError,
    BatchNotFoundError,
    KBNotFoundError,
    RagnestError,
)
from ragnest.mcp.formatting import (
    format_batch_list,
    format_batch_status,
    format_worker_status,
)

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from ragnest.services.kb_service import KBService
    from ragnest.services.worker_service import WorkerService

logger = logging.getLogger(__name__)


def register_batch_tools(
    mcp: FastMCP,
    kb_service: KBService,
    worker_service: WorkerService,
) -> None:
    """Register batch and worker MCP tools."""

    @mcp.tool
    def batch_status(
        batch_id: Annotated[str, Field(description="Batch ID to check")],
    ) -> str:
        """Check the progress and details of an ingestion batch.

        Shows file counts, failures, and chunk totals.
        """
        try:
            detail = kb_service.batch_status(batch_id)
        except BatchNotFoundError as e:
            raise ToolError(f"Batch '{e.batch_id}' not found.") from e
        except RagnestError as e:
            raise ToolError(str(e)) from e
        else:
            return format_batch_status(detail)

    @mcp.tool
    def list_batches(
        kb_name: Annotated[str, Field(description="Knowledge base name")],
    ) -> str:
        """List recent ingestion batches for a knowledge base."""
        try:
            batches = kb_service.list_batches(kb_name)
        except KBNotFoundError as e:
            raise ToolError(f"KB '{e.kb_name}' not found. Use list_kbs() to see available.") from e
        except RagnestError as e:
            raise ToolError(str(e)) from e
        else:
            return format_batch_list(batches)

    @mcp.tool
    def undo_batch(
        batch_id: Annotated[str, Field(description="Batch ID to undo")],
    ) -> str:
        """Undo an ingestion batch — removes all documents and chunks it created.

        This is useful for rolling back a bad ingestion. The batch record
        is kept for audit purposes but marked as undone.
        """
        try:
            kb_service.undo_batch(batch_id)
        except BatchNotFoundError as e:
            raise ToolError(f"Batch '{e.batch_id}' not found.") from e
        except BatchAlreadyUndoneError as e:
            raise ToolError(f"Batch '{e.batch_id}' was already undone.") from e
        except RagnestError as e:
            raise ToolError(str(e)) from e
        else:
            return (
                f"Batch '{batch_id}' has been undone. "
                f"All associated documents and chunks have been removed."
            )

    @mcp.tool
    def worker_status() -> str:
        """Check the background worker's current state.

        Shows queue depth, processing status, and last run time.
        """
        try:
            status = worker_service.get_status()
        except RagnestError as e:
            raise ToolError(str(e)) from e
        else:
            return format_worker_status(status)

    @mcp.tool
    def trigger_scan(
        kb_name: Annotated[
            str | None,
            Field(description="KB to scan (omit to scan all)"),
        ] = None,
    ) -> str:
        """Trigger an immediate scan of watch paths for new/modified files.

        Discovered files are queued for the worker to process.
        """
        try:
            queued = worker_service.scan_watch_paths(kb_name)
        except KBNotFoundError as e:
            raise ToolError(f"KB '{e.kb_name}' not found. Use list_kbs() to see available.") from e
        except RagnestError as e:
            raise ToolError(str(e)) from e
        else:
            target = f"KB '{kb_name}'" if kb_name else "all KBs"
            if queued == 0:
                return f"Scan complete for {target}. No new files found."
            return (
                f"Scan complete for {target}. "
                f"Queued {queued} file(s) for processing.\n\n"
                f"Run the worker to process: ragnest-worker"
            )
