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


_HELP_TEXT = """\
# Ragnest Configuration Reference

## Config file locations (searched in order)
1. `./config.yaml` (current working directory)
2. `~/.ragnest/config.yaml` (user home)
3. Override with env var: `RAGNEST_CONFIG=/path/to/config.yaml`

## config.yaml

```yaml
database:
  host: localhost       # PostgreSQL host
  port: 5433            # PostgreSQL port
  name: ragnest         # Database name

ollama:
  base_url: http://localhost:11434   # Ollama API URL (local or remote)

defaults:
  chunk_size: 1000      # Characters per chunk
  chunk_overlap: 200    # Overlap between chunks
  separator: "\\n\\n"    # Text split separator

state:
  path: ~/.ragnest/state.db   # SQLite state database path
```

## Secrets (.env file, same directory as config.yaml)

```
RAGNEST_DATABASE__USER=ragnest
RAGNEST_DATABASE__PASSWORD=yourpassword
```

## Environment variable overrides

All settings can be overridden via env vars with `RAGNEST_` prefix and `__` as separator:

| Setting | Env var |
|---------|---------|
| Database host | `RAGNEST_DATABASE__HOST` |
| Database port | `RAGNEST_DATABASE__PORT` |
| Database name | `RAGNEST_DATABASE__NAME` |
| Database user | `RAGNEST_DATABASE__USER` |
| Database password | `RAGNEST_DATABASE__PASSWORD` |
| Ollama URL | `RAGNEST_OLLAMA__BASE_URL` |
| State DB path | `RAGNEST_STATE__PATH` |
| Config file | `RAGNEST_CONFIG` |

## Multiple database backends

```yaml
databases:
  local:
    host: localhost
    port: 5433
    name: ragnest
  cloud:
    host: xyz.supabase.co
    port: 5432
    name: postgres
```

Use `backend="cloud"` when creating a KB to route it to a specific database.

## Installation

```
pip install ragnest
claude mcp add ragnest -- uvx ragnest
```

## Worker commands

```
ragnest-worker --scan                 # Scan watch paths + process queue
ragnest-worker --scan --kb <name>     # Specific KB only
ragnest-worker --retry                # Retry failed files
ragnest-worker --scan --dry-run       # Preview what would be queued
```

## Prerequisites

- Python 3.12+
- PostgreSQL with pgvector extension
- Ollama for embeddings (local or remote)
"""


def register_system_tools(mcp: FastMCP, system_service: SystemService) -> None:
    """Register system information MCP tools."""

    @mcp.tool
    def ragnest_help() -> str:
        """Show Ragnest configuration reference and usage guide.

        Returns config file format, environment variables, installation
        instructions, worker commands, and all available settings.
        Call this when users ask how to configure ragnest.
        """
        return _HELP_TEXT

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
