"""Response formatting helpers — convert domain models to human-readable strings."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragnest.models.domain import (
        BatchDetail,
        BatchInfo,
        DBStatus,
        DocumentInfo,
        KBStats,
        SearchResult,
        SystemInfo,
        WatchPathInfo,
        WorkerStatus,
    )

_SCORE_WEAK = 0.3
_SCORE_GOOD = 0.6
_CONTENT_PREVIEW_LEN = 300


def format_search_results(results: list[SearchResult]) -> str:
    """Format search results with scores, filenames, and content snippets."""
    if not results:
        return "No results found."

    lines: list[str] = [f"Found {len(results)} result(s):\n"]
    for i, r in enumerate(results, 1):
        score_label = (
            "weak" if r.score < _SCORE_WEAK else "good" if r.score < _SCORE_GOOD else "strong"
        )
        preview = r.content[:_CONTENT_PREVIEW_LEN]
        ellipsis = "..." if len(r.content) > _CONTENT_PREVIEW_LEN else ""
        lines.append(
            f"**{i}. [{r.filename}]** (score: {r.score:.3f} — {score_label})\n"
            f"   KB: {r.kb_name} | Chunk #{r.chunk_index} | Doc ID: {r.document_id}\n"
            f"   {preview}{ellipsis}\n"
        )
    return "\n".join(lines)


def format_kb_list(kbs: list[KBStats]) -> str:
    """Format a list of knowledge bases as a summary table."""
    if not kbs:
        return "No knowledge bases found. Use create_kb() to create one."

    lines: list[str] = [f"**{len(kbs)} knowledge base(s):**\n"]
    for kb in kbs:
        ext_label = ""
        if kb.external:
            ext_label = f" [{kb.mode}]"
        lines.append(
            f"- **{kb.name}**: {kb.description or '(no description)'}{ext_label}\n"
            f"  Model: {kb.model} ({kb.dimensions}d) | "
            f"Docs: {kb.document_count} | Chunks: {kb.chunk_count} | "
            f"Backend: {kb.backend}"
        )
    return "\n".join(lines)


def format_kb_detail(kb: KBStats) -> str:
    """Format a single knowledge base with full detail."""
    lines = [
        f"**{kb.name}**",
        f"Description: {kb.description or '(none)'}",
        f"Model: {kb.model} ({kb.dimensions} dimensions)",
        f"Documents: {kb.document_count}",
        f"Chunks: {kb.chunk_count}",
        f"Backend: {kb.backend}",
    ]
    if kb.external:
        lines.append(f"External: yes (mode: {kb.mode})")
    return "\n".join(lines)


def format_batch_status(batch: BatchDetail) -> str:
    """Format detailed batch status with progress and failure info."""
    lines: list[str] = [
        f"**Batch {batch.id}** — {batch.status.value}\n"
        f"KB: {batch.kb_name}\n"
        f"Description: {batch.description or '(none)'}\n"
        f"Progress: {batch.processed_files}/{batch.total_files} processed, "
        f"{batch.failed_files} failed, {batch.skipped_files} skipped\n"
        f"Chunks created: {batch.total_chunks}\n"
        f"Pending: {batch.pending_count}",
    ]
    if batch.created_at:
        lines.append(f"Created: {batch.created_at.isoformat()}")
    if batch.completed_at:
        lines.append(f"Completed: {batch.completed_at.isoformat()}")
    if batch.failed_details:
        lines.append("\nFailed files:")
        lines.extend(
            f"  - {detail.get('file', 'unknown')}: {detail.get('error', '?')}"
            for detail in batch.failed_details
        )
    return "\n".join(lines)


def format_batch_list(batches: list[BatchInfo]) -> str:
    """Format a list of batches as a compact summary."""
    if not batches:
        return "No batches found."

    lines: list[str] = [f"**{len(batches)} batch(es):**\n"]
    for b in batches:
        created = b.created_at.isoformat() if b.created_at else "?"
        lines.append(
            f"- **{b.id}** [{b.status.value}] — {b.description or '(none)'}\n"
            f"  Files: {b.processed_files}/{b.total_files} | "
            f"Failed: {b.failed_files} | Chunks: {b.total_chunks} | "
            f"Created: {created}"
        )
    return "\n".join(lines)


def format_watch_paths(paths: list[WatchPathInfo]) -> str:
    """Format watch paths with their configuration and scan status."""
    if not paths:
        return "No watch paths configured."

    lines: list[str] = [f"**{len(paths)} watch path(s):**\n"]
    for wp in paths:
        status = "enabled" if wp.enabled else "PAUSED"
        scanned = wp.last_scanned_at.isoformat() if wp.last_scanned_at else "never"
        lines.append(
            f"- **{wp.dir_path}** -> KB '{wp.kb_name}' [{status}]\n"
            f"  Recursive: {wp.recursive} | Patterns: {wp.file_patterns} | "
            f"Last scanned: {scanned}"
        )
    return "\n".join(lines)


def format_document_list(docs: list[DocumentInfo]) -> str:
    """Format a list of documents with metadata."""
    if not docs:
        return "No documents found."

    lines: list[str] = [f"**{len(docs)} document(s):**\n"]
    for d in docs:
        ingested = d.ingested_at.isoformat() if d.ingested_at else "?"
        lines.append(
            f"- **{d.filename or '(unknown)'}** (ID: {d.id})\n"
            f"  Type: {d.file_type or '?'} | Chunks: {d.chunk_count} | "
            f"Ingested: {ingested}"
        )
    return "\n".join(lines)


def format_worker_status(status: WorkerStatus) -> str:
    """Format worker status with queue depth and current state."""
    state = "processing" if status.is_processing else "idle"
    last_run = status.last_run_at.isoformat() if status.last_run_at else "never"
    lines: list[str] = [
        f"**Worker Status:** {state}\nQueue depth: {status.queue_depth}\nLast run: {last_run}",
    ]
    if status.current_kb:
        lines.append(f"Current KB: {status.current_kb}")
    return "\n".join(lines)


def format_db_status(status: DBStatus) -> str:
    """Format database status with connection health and table sizes."""
    if not status.connected:
        return f"**Database:** DISCONNECTED (backend: {status.backend})"

    lines: list[str] = [
        f"**Database:** connected (backend: {status.backend})\n"
        f"Knowledge bases: {status.total_kbs}\n"
        f"Documents: {status.total_documents}\n"
        f"Chunks: {status.total_chunks}",
    ]
    if status.table_sizes:
        lines.append("\nTable row counts:")
        lines.extend(f"  {table}: {count}" for table, count in sorted(status.table_sizes.items()))
    return "\n".join(lines)


def format_system_info(info: SystemInfo) -> str:
    """Format overall system information."""
    lines: list[str] = [
        f"**Ragnest v{info.version}**\n"
        f"Ollama URL: {info.ollama_url}\n"
        f"DB: {'connected' if info.db_status.connected else 'DISCONNECTED'} "
        f"({info.db_status.backend})\n"
        f"Backends: {', '.join(info.configured_backends) or '(none)'}\n"
        f"Configured KBs: {', '.join(info.configured_kbs) or '(none)'}\n"
        f"Supported formats: {', '.join(info.supported_formats) or '(none)'}",
    ]
    return "\n".join(lines)
