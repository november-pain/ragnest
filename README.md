# Ragnest

[![PyPI](https://img.shields.io/pypi/v/ragnest)](https://pypi.org/project/ragnest/)
[![Python](https://img.shields.io/pypi/pyversions/ragnest)](https://pypi.org/project/ragnest/)
[![CI](https://github.com/november-pain/ragnest/actions/workflows/ci.yml/badge.svg)](https://github.com/november-pain/ragnest/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/november-pain/ragnest/blob/main/LICENSE)

Multi-knowledge-base RAG system with [MCP](https://modelcontextprotocol.io/) integration for [Claude Code](https://docs.anthropic.com/en/docs/claude-code).

Create multiple knowledge bases, each with its own embedding model, chunk settings, and vector backend. Search them from Claude Code via 29 MCP tools.

## Quick Start

### 1. Add to Claude Code

```bash
claude mcp add ragnest -- uvx ragnest
```

On first run, Ragnest creates `~/.ragnest/config.yaml` and `~/.ragnest/.env` with defaults. The MCP server starts immediately — no manual config needed.

### 2. Check setup

Ask Claude: *"check ragnest setup status"* — it will call `ragnest_setup_status()` and tell you what's missing.

### 3. Prerequisites

You'll need these before creating knowledge bases:

- **Ollama** for embeddings:
  ```bash
  brew install ollama && brew services start ollama
  ollama pull bge-m3
  ```

- **PostgreSQL 15+** with [pgvector](https://github.com/pgvector/pgvector) for vector storage:
  ```bash
  pip install ragnest  # if not using uvx
  cd $(pip show ragnest | grep Location | cut -d' ' -f2)/../..
  docker compose up -d
  ```
  Or use any PostgreSQL 15+ instance with pgvector installed.

### 4. Configure

Edit `~/.ragnest/config.yaml` with your database settings:
```yaml
database:
  host: localhost
  port: 5432
  name: ragnest
```

Edit `~/.ragnest/.env` with credentials:
```bash
RAGNEST_DATABASE__USER=ragnest
RAGNEST_DATABASE__PASSWORD=yourpassword
```

Claude can help you with this — just ask.

## Installation

```bash
pip install ragnest
```

## Usage

### Initialize a knowledge base from a folder

```
init_kb("my_docs", "/path/to/docs", "bge-m3", file_patterns="*.py,*.md")
```

Creates the KB, sets a watch path with file filtering, and queues files for embedding.

### Run the worker

```bash
ragnest-worker --scan --kb my_docs
```

The worker processes the queue: reads files, chunks text, generates embeddings via Ollama, and stores vectors in PostgreSQL.

### Search

```
search_kb("my_docs", "how does authentication work", top_k=5)
search_all_kbs("deployment process", top_k_per_kb=3)
```

Returns ranked results with source filenames, scores, and text chunks.

## Architecture

```
Claude Code ──MCP──▶ MCP Server (29 tools)
                         │
              ┌──────────┼──────────┐
              ▼                     ▼
         SQLite                Vector Backend
       (local state)           (pgvector)
              │                     ▲
              └──────▶ Worker ──────┘
                         │
                      Ollama
                    (embeddings)
```

**Two-layer storage:**

| Layer | Engine | Stores | Purpose |
|-------|--------|--------|---------|
| State | SQLite | KBs, documents, batches, queue, watch paths | Local, zero-config, works offline |
| Vectors | PostgreSQL + pgvector | Chunks with embeddings + inline metadata | Portable, queryable by external systems |

## MCP Tools

| Category | Tools |
|----------|-------|
| **Search** | `search_kb`, `search_all_kbs`, `get_similar_documents` |
| **KB Management** | `list_kbs`, `create_kb`, `update_kb`, `delete_kb`, `init_kb` |
| **Watch Paths** | `add_watch_path`, `remove_watch_path`, `list_watch_paths`, `pause_watch_path`, `resume_watch_path` |
| **Ingestion** | `add_file`, `add_directory`, `add_text` |
| **Batches** | `batch_status`, `list_batches`, `undo_batch`, `worker_status`, `trigger_scan` |
| **Documents** | `list_documents`, `delete_document` |
| **System** | `ragnest_setup_status`, `ragnest_help`, `db_status`, `list_models`, `system_info` |
| **Export** | `export_knowledge_base` |

## Features

- **Zero-config startup** — MCP server starts immediately, scaffolds config on first run
- **Setup wizard** — `ragnest_setup_status` checks all prerequisites and guides through fixes
- **Multiple knowledge bases** — each with its own embedding model, dimensions, and chunk settings
- **Per-KB backend routing** — route different KBs to different PostgreSQL databases
- **External KB support** — connect to remote vector stores in read-only or read-write mode
- **Watch paths with file filtering** — glob patterns like `*.py,*.md` to control what gets indexed
- **Batch tracking** — view progress, retry failures, undo entire batches
- **Resilient worker** — per-file commits, SIGINT/SIGTERM handling, resume on restart
- **Content deduplication** — SHA-256 hashing skips unchanged files
- **Cross-KB search** — search all knowledge bases in one call
- **Remote Ollama** — configure any Ollama-compatible API endpoint
- **Export** — Parquet or JSON with model metadata sidecar

## Configuration

Config is auto-created at `~/.ragnest/config.yaml` on first run. All settings can also be overridden via environment variables with `RAGNEST_` prefix:

| Setting | Env var |
|---------|---------|
| Database host | `RAGNEST_DATABASE__HOST` |
| Database port | `RAGNEST_DATABASE__PORT` |
| Database name | `RAGNEST_DATABASE__NAME` |
| Database user | `RAGNEST_DATABASE__USER` |
| Database password | `RAGNEST_DATABASE__PASSWORD` |
| Ollama URL | `RAGNEST_OLLAMA__BASE_URL` |
| Config file | `RAGNEST_CONFIG` |

### Multiple backends

```yaml
databases:
  local:
    host: localhost
    port: 5432
    name: ragnest
  cloud:
    host: xyz.supabase.co
    port: 5432
    name: postgres
```

Then specify `backend="cloud"` when creating a KB.

## Worker

```bash
ragnest-worker --scan                 # Scan watch paths + process queue
ragnest-worker --scan --kb my_docs    # Specific KB only
ragnest-worker --retry                # Retry failed files
ragnest-worker --scan --dry-run       # Preview what would be queued
```

## Development

```bash
git clone https://github.com/november-pain/ragnest.git
cd ragnest
pip install -e ".[dev]"

make lint             # ruff check + format
make typecheck        # mypy + basedpyright (strict)
make test             # pytest
```

## License

[MIT](LICENSE)
