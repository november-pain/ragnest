# Ragnest

[![PyPI](https://img.shields.io/pypi/v/ragnest)](https://pypi.org/project/ragnest/)
[![Python](https://img.shields.io/pypi/pyversions/ragnest)](https://pypi.org/project/ragnest/)
[![CI](https://github.com/november-pain/ragnest/actions/workflows/ci.yml/badge.svg)](https://github.com/november-pain/ragnest/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/november-pain/ragnest/blob/main/LICENSE)

Multi-knowledge-base RAG system with [MCP](https://modelcontextprotocol.io/) integration for [Claude Code](https://docs.anthropic.com/en/docs/claude-code).

Create multiple knowledge bases, each with its own embedding model, chunk settings, and vector backend. Search them from Claude Code via 27 MCP tools.

## Installation

```bash
pip install ragnest
```

### Prerequisites

- **Python 3.12+**
- **PostgreSQL** with [pgvector](https://github.com/pgvector/pgvector) extension
- **Ollama** for local embeddings

## Quick Start

### 1. Start PostgreSQL with pgvector

```bash
# Using the included Docker Compose
git clone https://github.com/november-pain/ragnest.git
cd ragnest
docker compose up -d
```

Or use any PostgreSQL 15+ instance with pgvector installed.

### 2. Configure

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml`:
```yaml
database:
  host: localhost
  port: 5433
  name: ragnest

ollama:
  base_url: http://localhost:11434
```

Create `.env` with database credentials:
```bash
RAGNEST_DATABASE__USER=ragnest
RAGNEST_DATABASE__PASSWORD=yourpassword
```

### 3. Install Ollama and pull an embedding model

```bash
brew install ollama && brew services start ollama
ollama pull bge-m3
```

### 4. Add to Claude Code

```bash
claude mcp add ragnest -- uvx --from ragnest ragnest-server
```

Database and schema are auto-initialized when the MCP server starts.

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
Claude Code ──MCP──▶ MCP Server (27 tools)
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
| **System** | `db_status`, `list_models`, `system_info` |
| **Export** | `export_knowledge_base` |

## Features

- **Multiple knowledge bases** — each with its own embedding model, dimensions, and chunk settings
- **Per-KB backend routing** — route different KBs to different PostgreSQL databases
- **External KB support** — connect to remote vector stores in read-only or read-write mode
- **Watch paths with file filtering** — glob patterns like `*.py,*.md` to control what gets indexed
- **Batch tracking** — view progress, retry failures, undo entire batches
- **Resilient worker** — per-file commits, SIGINT/SIGTERM handling, resume on restart
- **Content deduplication** — SHA-256 hashing skips unchanged files
- **Cross-KB search** — search all knowledge bases in one call
- **Export** — Parquet or JSON with model metadata sidecar

## Configuration

### Basic

```yaml
# config.yaml
database:
  host: localhost
  port: 5433
  name: ragnest

ollama:
  base_url: http://localhost:11434

defaults:
  chunk_size: 1000
  chunk_overlap: 200
```

```bash
# .env
RAGNEST_DATABASE__USER=ragnest
RAGNEST_DATABASE__PASSWORD=yourpassword
```

### Multiple backends

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

Then specify `backend="cloud"` when creating a KB.

## Worker

```bash
ragnest-worker --scan                 # Scan watch paths + process queue
ragnest-worker --scan --kb my_docs    # Specific KB only
ragnest-worker --retry                # Retry failed files
ragnest-worker --scan --dry-run       # Preview what would be queued
```

Deploy files for [launchd](deploy/com.raghub.worker.plist) and [systemd](deploy/rag-hub-worker.service) are included for scheduled runs.

## Development

```bash
git clone https://github.com/november-pain/ragnest.git
cd ragnest
pip install -e ".[dev]"

make lint             # ruff check + format
make typecheck        # mypy + basedpyright (strict)
make test             # pytest (113 tests)
```

## Project Structure

```
src/ragnest/
├── app.py                  # Application container + wiring
├── config.py               # Pydantic Settings + YAML
├── exceptions.py           # Exception hierarchy
├── models/                 # Pydantic domain + DB row models
├── db/
│   ├── backends/           # PostgreSQL, SQLite implementations
│   ├── repositories/       # 6 data access repositories
│   ├── schema.py           # Vector DDL + index management
│   └── sqlite_schema.py    # Local state DDL
├── services/               # Business logic (7 services)
├── mcp/
│   ├── server.py           # FastMCP app factory
│   └── tools/              # 8 tool modules (27 tools)
└── cli/                    # Worker + DB setup entrypoints
```

## License

[MIT](LICENSE)
