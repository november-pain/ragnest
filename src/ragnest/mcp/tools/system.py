"""System tools — database health, model listing, and system information."""

# pyright: reportUnusedFunction=false

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastmcp.exceptions import ToolError

from ragnest.exceptions import RagnestError
from ragnest.mcp.formatting import format_db_status, format_system_info

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from ragnest.services.system_service import SystemService

logger = logging.getLogger(__name__)


def register_system_tools(mcp: FastMCP, system_service: SystemService) -> None:
    """Register system information MCP tools."""

    @mcp.tool
    def db_status() -> str:
        """Check database connectivity and show aggregate statistics.

        Returns connection status, row counts, and table sizes.
        """
        try:
            status = system_service.db_status()
        except RagnestError as e:
            raise ToolError(str(e)) from e
        else:
            return format_db_status(status)

    @mcp.tool
    def list_models() -> str:
        """List available Ollama embedding models.

        Shows models that can be used when creating knowledge bases.
        Requires Ollama to be running.
        """
        try:
            models = system_service.list_models()
        except RagnestError as e:
            raise ToolError(str(e)) from e
        else:
            if not models:
                return "No embedding models found. Is Ollama running? Try: ollama pull bge-m3"
            lines = [f"**{len(models)} available model(s):**\n"]
            lines.extend(f"- {m}" for m in models)
            return "\n".join(lines)

    @mcp.tool
    def system_info() -> str:
        """Show overall system information.

        Includes version, database status, Ollama URL, configured KBs,
        and supported file formats.
        """
        try:
            info = system_service.system_info()
        except RagnestError as e:
            raise ToolError(str(e)) from e
        else:
            return format_system_info(info)
