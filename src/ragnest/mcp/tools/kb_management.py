"""Knowledge base management tools — CRUD and initialization."""

# pyright: reportUnusedFunction=false

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, Literal

from fastmcp.exceptions import ToolError
from pydantic import Field

from ragnest.exceptions import (
    KBAlreadyExistsError,
    KBNotFoundError,
    RagnestError,
)
from ragnest.mcp.formatting import format_kb_detail, format_kb_list
from ragnest.models.domain import KBConfig

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from ragnest.services.kb_service import KBService

logger = logging.getLogger(__name__)


def register_kb_tools(mcp: FastMCP, kb_service: KBService) -> None:
    """Register knowledge base management MCP tools."""

    @mcp.tool
    def list_kbs() -> str:
        """List all knowledge bases with document and chunk counts.

        Run this early in a conversation to see what's available.
        """
        try:
            kbs = kb_service.list_kbs()
        except RagnestError as e:
            raise ToolError(str(e)) from e
        else:
            return format_kb_list(kbs)

    @mcp.tool
    def create_kb(
        name: Annotated[
            str,
            Field(description="KB name (lowercase, underscores, e.g. 'my_docs')"),
        ],
        model: Annotated[
            str,
            Field(description="Embedding model (e.g. 'bge-m3' or 'gte-qwen2:7b')"),
        ],
        description: Annotated[str, Field(description="Human-readable description")] = "",
        dimensions: Annotated[
            int, Field(description="Embedding dimensions (must match model)")
        ] = 1024,
        chunk_size: Annotated[int, Field(description="Characters per chunk")] = 1000,
        chunk_overlap: Annotated[int, Field(description="Overlap between chunks")] = 200,
        backend: Annotated[
            str,
            Field(description="Database backend name (default: 'default')"),
        ] = "default",
        external: Annotated[
            bool,
            Field(description="True for external/shared KBs (no watch paths)"),
        ] = False,
        mode: Annotated[
            str,
            Field(description="Access mode: 'read_write' or 'read_only'"),
        ] = "read_write",
    ) -> str:
        """Create a new knowledge base.

        Choose 'bge-m3' (1024d) for multilingual/Ukrainian content,
        'gte-qwen2:7b' (3584d) for English-only with higher quality.
        Use the 'backend' parameter to select which database backend to use.
        Set 'external=True' for shared/remote KBs that don't need watch paths.
        """
        if mode not in ("read_write", "read_only"):
            raise ToolError(f"Invalid mode: {mode!r}. Must be 'read_write' or 'read_only'.")
        kb_mode: Literal["read_write", "read_only"] = mode  # type: ignore[assignment]
        try:
            config = KBConfig(
                name=name,
                description=description,
                model=model,
                dimensions=dimensions,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                backend=backend,
                external=external,
                mode=kb_mode,
            )
            kb = kb_service.create_kb(config)
        except KBAlreadyExistsError as e:
            raise ToolError(f"KB '{e.kb_name}' already exists. Use update_kb() to modify.") from e
        except RagnestError as e:
            raise ToolError(str(e)) from e
        else:
            return f"Created successfully.\n\n{format_kb_detail(kb)}"

    @mcp.tool
    def update_kb(
        kb_name: Annotated[str, Field(description="Knowledge base name")],
        description: Annotated[
            str | None, Field(description="New description (omit to keep current)")
        ] = None,
        chunk_size: Annotated[
            int | None, Field(description="New chunk size (omit to keep current)")
        ] = None,
        chunk_overlap: Annotated[
            int | None, Field(description="New chunk overlap (omit to keep current)")
        ] = None,
    ) -> str:
        """Update mutable settings of a knowledge base.

        Note: changing chunk_size/chunk_overlap only affects future ingestions.
        Existing chunks are not re-split.
        """
        try:
            kb = kb_service.update_kb(
                kb_name,
                description=description,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        except KBNotFoundError as e:
            raise ToolError(f"KB '{e.kb_name}' not found. Use list_kbs() to see available.") from e
        except RagnestError as e:
            raise ToolError(str(e)) from e
        else:
            return f"Updated successfully.\n\n{format_kb_detail(kb)}"

    @mcp.tool
    def delete_kb(
        kb_name: Annotated[str, Field(description="Knowledge base to delete")],
    ) -> str:
        """Permanently delete a knowledge base and all its data.

        This is irreversible. All documents, chunks, watch paths,
        and batches associated with this KB will be removed.
        """
        try:
            kb_service.delete_kb(kb_name)
        except KBNotFoundError as e:
            raise ToolError(f"KB '{e.kb_name}' not found. Use list_kbs() to see available.") from e
        except RagnestError as e:
            raise ToolError(str(e)) from e
        else:
            return f"Knowledge base '{kb_name}' has been permanently deleted."

    @mcp.tool
    def init_kb(
        name: Annotated[
            str,
            Field(description="KB name (lowercase, underscores)"),
        ],
        folder_path: Annotated[
            str,
            Field(description="Directory to watch for documents"),
        ],
        model: Annotated[
            str,
            Field(description="Embedding model (e.g. 'bge-m3')"),
        ],
        description: Annotated[str, Field(description="Human-readable description")] = "",
        dimensions: Annotated[int, Field(description="Embedding dimensions")] = 1024,
        chunk_size: Annotated[int, Field(description="Characters per chunk")] = 1000,
        chunk_overlap: Annotated[int, Field(description="Overlap between chunks")] = 200,
        backend: Annotated[
            str,
            Field(description="Database backend name (default: 'default')"),
        ] = "default",
        file_patterns: Annotated[
            str,
            Field(
                description=(
                    "Comma-separated glob patterns for files to ingest "
                    "(e.g. '*.py', '*.md,*.txt'). Default '*' matches all."
                )
            ),
        ] = "*",
    ) -> str:
        """Convenience: create a KB and set up a watch path in one step.

        Cannot be used with external KBs. After calling this, run the
        worker to scan and ingest: ``ragnest-worker --scan --kb <name>``
        """
        try:
            kb = kb_service.init_kb(
                name=name,
                folder_path=folder_path,
                model=model,
                description=description,
                dimensions=dimensions,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                backend=backend,
                file_patterns=file_patterns,
            )
        except KBAlreadyExistsError as e:
            raise ToolError(f"KB '{e.kb_name}' already exists. Use update_kb() to modify.") from e
        except RagnestError as e:
            raise ToolError(str(e)) from e
        else:
            return (
                f"Initialized KB with watch path.\n\n{format_kb_detail(kb)}\n\n"
                f"Watch path: {folder_path}\n"
                f"Run the worker to scan and process: "
                f"ragnest-worker --scan --kb {name}"
            )
