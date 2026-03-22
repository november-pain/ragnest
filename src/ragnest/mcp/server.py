"""FastMCP app factory — creates the MCP server with all registered tools."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastmcp import FastMCP

from ragnest.app import Application
from ragnest.config import load_settings
from ragnest.mcp.tools.batches import register_batch_tools
from ragnest.mcp.tools.documents import register_document_tools
from ragnest.mcp.tools.export import register_export_tools
from ragnest.mcp.tools.ingestion import register_ingestion_tools
from ragnest.mcp.tools.kb_management import register_kb_tools
from ragnest.mcp.tools.search import register_search_tools
from ragnest.mcp.tools.system import register_system_tools
from ragnest.mcp.tools.watch_paths import register_watch_path_tools

if TYPE_CHECKING:
    from ragnest.config import AppSettings

logger = logging.getLogger(__name__)

_INSTRUCTIONS = """\
You have access to Ragnest, a multi-knowledge-base RAG system.

## Workflow
- Run list_kbs() early to see available knowledge bases.
- Search before answering — use search_kb or search_all_kbs.
- Cite sources with filename and score. Flag weak matches (score < 0.3).
- Prefer watch paths over one-time ingestion for ongoing directories.
- Queued files are processed asynchronously by the background worker.
- Use list_models() to discover available embedding models before creating a KB.

## Configuration
- Call ragnest_help() for full configuration reference (config file, env vars, all settings).
- Config file: ~/.ragnest/config.yaml (or ./config.yaml, or RAGNEST_CONFIG env var).
- Secrets: ~/.ragnest/.env (RAGNEST_DATABASE__USER, RAGNEST_DATABASE__PASSWORD).
- Ollama URL is configurable — default http://localhost:11434, supports remote servers.
- All settings can be overridden via env vars with RAGNEST_ prefix (e.g. RAGNEST_OLLAMA__BASE_URL).

## Dependencies (manage via CLI, not MCP)
- Ollama must be running for embeddings. Install: brew install ollama
- Pull models via CLI: ollama pull <model_name>
- Worker processes queued files: ragnest-worker --scan
- PostgreSQL with pgvector required for vector storage."""


def create_mcp_server(settings: AppSettings | None = None) -> FastMCP:
    """Create and configure the Ragnest MCP server with all tools registered.

    Args:
        settings: Application settings. If ``None``, loads from config file / env.

    Returns:
        A fully configured ``FastMCP`` instance ready to run.
    """
    if settings is None:
        settings = load_settings()

    app = Application(settings)

    mcp = FastMCP("Ragnest", instructions=_INSTRUCTIONS)

    register_search_tools(mcp, app.kb_service)
    register_kb_tools(mcp, app.kb_service)
    register_watch_path_tools(mcp, app.kb_service)
    register_ingestion_tools(mcp, app.ingest_service)
    register_batch_tools(mcp, app.kb_service, app.worker_service)
    register_document_tools(mcp, app.kb_service)
    register_system_tools(mcp, app.system_service)
    register_export_tools(mcp, app.export_service)

    logger.info("MCP server created with all tools registered")
    return mcp


def main() -> None:
    """Entry point for ``python -m ragnest.mcp.server``."""
    server = create_mcp_server()
    server.run()


if __name__ == "__main__":
    main()
