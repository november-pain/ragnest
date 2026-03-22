"""Watch path tools — manage auto-ingestion directories for knowledge bases."""

# pyright: reportUnusedFunction=false

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated

from fastmcp.exceptions import ToolError
from pydantic import Field

from ragnest.exceptions import KBNotFoundError, RagnestError
from ragnest.mcp.formatting import format_watch_paths

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from ragnest.services.kb_service import KBService

logger = logging.getLogger(__name__)


def register_watch_path_tools(mcp: FastMCP, kb_service: KBService) -> None:
    """Register watch path management MCP tools."""

    @mcp.tool
    def add_watch_path(
        kb_name: Annotated[str, Field(description="Knowledge base name")],
        dir_path: Annotated[str, Field(description="Directory path to watch")],
        recursive: Annotated[bool, Field(description="Watch subdirectories")] = True,
        file_patterns: Annotated[
            str,
            Field(description="Comma-separated file patterns (e.g. '*.md,*.txt')"),
        ] = "*",
    ) -> str:
        """Add a directory to be watched for automatic ingestion.

        The worker will scan this path and queue matching files.
        Prefer watch paths over one-time ingestion for ongoing directories.
        """
        try:
            wp = kb_service.add_watch_path(
                kb_name, dir_path, recursive=recursive, file_patterns=file_patterns
            )
        except KBNotFoundError as e:
            raise ToolError(f"KB '{e.kb_name}' not found. Use list_kbs() to see available.") from e
        except RagnestError as e:
            raise ToolError(str(e)) from e
        else:
            return (
                f"Watch path added.\n\n"
                f"Path: {wp.dir_path}\n"
                f"KB: {wp.kb_name}\n"
                f"Recursive: {wp.recursive}\n"
                f"Patterns: {wp.file_patterns}\n\n"
                f"Run the worker to scan: ragnest-worker --scan --kb {kb_name}"
            )

    @mcp.tool
    def remove_watch_path(
        kb_name: Annotated[str, Field(description="Knowledge base name")],
        dir_path: Annotated[str, Field(description="Directory path to remove")],
    ) -> str:
        """Remove a watch path. Already-ingested documents are not affected."""
        try:
            kb_service.remove_watch_path(kb_name, dir_path)
        except RagnestError as e:
            raise ToolError(str(e)) from e
        else:
            return f"Watch path '{dir_path}' removed from KB '{kb_name}'."

    @mcp.tool
    def list_watch_paths(
        kb_name: Annotated[
            str | None,
            Field(description="Filter by KB name (omit to list all)"),
        ] = None,
    ) -> str:
        """List configured watch paths, optionally filtered by knowledge base."""
        try:
            paths = kb_service.list_watch_paths(kb_name)
        except RagnestError as e:
            raise ToolError(str(e)) from e
        else:
            return format_watch_paths(paths)

    @mcp.tool
    def pause_watch_path(
        kb_name: Annotated[str, Field(description="Knowledge base name")],
        dir_path: Annotated[str, Field(description="Directory path to pause")],
    ) -> str:
        """Pause a watch path — the worker will skip it during scans.

        Use resume_watch_path() to re-enable.
        """
        try:
            kb_service.pause_watch_path(kb_name, dir_path)
        except RagnestError as e:
            raise ToolError(str(e)) from e
        else:
            return (
                f"Watch path '{dir_path}' paused for KB '{kb_name}'. "
                f"Use resume_watch_path() to re-enable."
            )

    @mcp.tool
    def resume_watch_path(
        kb_name: Annotated[str, Field(description="Knowledge base name")],
        dir_path: Annotated[str, Field(description="Directory path to resume")],
    ) -> str:
        """Resume a previously paused watch path."""
        try:
            kb_service.resume_watch_path(kb_name, dir_path)
        except RagnestError as e:
            raise ToolError(str(e)) from e
        else:
            return f"Watch path '{dir_path}' resumed for KB '{kb_name}'."
