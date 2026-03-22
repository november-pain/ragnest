"""Export tools — export knowledge bases to portable formats."""

# pyright: reportUnusedFunction=false

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated

from fastmcp.exceptions import ToolError
from pydantic import Field

from ragnest.exceptions import KBNotFoundError, RagnestError

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from ragnest.services.export_service import ExportService

logger = logging.getLogger(__name__)


def register_export_tools(mcp: FastMCP, export_service: ExportService) -> None:
    """Register export MCP tools."""

    @mcp.tool
    def export_knowledge_base(
        kb_name: Annotated[str, Field(description="Knowledge base to export")],
        output_dir: Annotated[str, Field(description="Directory for output files")],
        fmt: Annotated[
            str,
            Field(description="Export format: 'parquet' or 'json'"),
        ] = "parquet",
    ) -> str:
        """Export a knowledge base to Parquet or JSON format.

        Creates the data file and a metadata sidecar (.meta.json)
        in the specified output directory.
        """
        try:
            file_path = export_service.export_kb(kb_name, output_dir, fmt=fmt)
        except KBNotFoundError as e:
            raise ToolError(f"KB '{e.kb_name}' not found. Use list_kbs() to see available.") from e
        except RagnestError as e:
            raise ToolError(str(e)) from e
        else:
            return (
                f"Export complete.\n\n"
                f"File: {file_path}\n"
                f"Format: {fmt}\n"
                f"A metadata sidecar ({kb_name}.meta.json) was also created."
            )
