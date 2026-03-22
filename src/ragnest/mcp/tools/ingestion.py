"""Ingestion tools — queue files, directories, and text for processing."""

# pyright: reportUnusedFunction=false

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated

from fastmcp.exceptions import ToolError
from pydantic import Field

from ragnest.exceptions import RagnestError

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from ragnest.services.ingest_service import IngestService

logger = logging.getLogger(__name__)


def register_ingestion_tools(mcp: FastMCP, ingest_service: IngestService) -> None:
    """Register ingestion MCP tools."""

    @mcp.tool
    def add_file(
        kb_name: Annotated[str, Field(description="Knowledge base name")],
        file_path: Annotated[str, Field(description="Absolute path to the file")],
    ) -> str:
        """Queue a single file for ingestion into a knowledge base.

        The file is queued for the background worker to process.
        Queued != done — the worker embeds asynchronously.
        Use batch_status() to check progress.
        """
        try:
            batch = ingest_service.queue_file(kb_name, file_path)
        except RagnestError as e:
            raise ToolError(str(e)) from e
        else:
            return (
                f"File queued for ingestion.\n\n"
                f"Batch ID: {batch.id}\n"
                f"KB: {batch.kb_name}\n"
                f"Files: {batch.total_files}\n\n"
                f"Run the worker to process: ragnest-worker --scan --kb {kb_name}"
            )

    @mcp.tool
    def add_directory(
        kb_name: Annotated[str, Field(description="Knowledge base name")],
        dir_path: Annotated[str, Field(description="Directory path to ingest")],
        recursive: Annotated[bool, Field(description="Include subdirectories")] = True,
        file_patterns: Annotated[
            str,
            Field(description="Comma-separated file patterns (e.g. '*.md,*.txt')"),
        ] = "*",
    ) -> str:
        """Queue all matching files in a directory for one-time ingestion.

        For ongoing directories, prefer add_watch_path() instead.
        The worker processes files asynchronously.
        """
        try:
            batch = ingest_service.queue_directory(
                kb_name, dir_path, recursive=recursive, file_patterns=file_patterns
            )
        except RagnestError as e:
            raise ToolError(str(e)) from e
        else:
            return (
                f"Directory queued for ingestion.\n\n"
                f"Batch ID: {batch.id}\n"
                f"KB: {batch.kb_name}\n"
                f"Files queued: {batch.total_files}\n\n"
                f"Run the worker to process: ragnest-worker --scan --kb {kb_name}"
            )

    @mcp.tool
    def add_text(
        kb_name: Annotated[str, Field(description="Knowledge base name")],
        text: Annotated[str, Field(description="Raw text content to ingest")],
        source_name: Annotated[
            str,
            Field(description="Label for this text entry"),
        ] = "manual_entry",
    ) -> str:
        """Queue raw text for ingestion into a knowledge base.

        The text is saved to a temp file and queued for the worker.
        Useful for ingesting content that isn't in a file.
        """
        try:
            batch = ingest_service.queue_text(kb_name, text, source_name=source_name)
        except RagnestError as e:
            raise ToolError(str(e)) from e
        else:
            return (
                f"Text queued for ingestion.\n\n"
                f"Batch ID: {batch.id}\n"
                f"KB: {batch.kb_name}\n"
                f"Source: {source_name}\n\n"
                f"Run the worker to process: ragnest-worker --scan --kb {kb_name}"
            )
