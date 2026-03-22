"""File reading utilities — dispatch by extension, PDF and text support."""

from __future__ import annotations

import logging
from pathlib import Path  # noqa: TC003 — used in function signatures at runtime

from ragnest.exceptions import FileReadError

logger = logging.getLogger(__name__)


SUPPORTED_EXTENSIONS: set[str] = {
    # Text / markup
    ".txt",
    ".md",
    ".html",
    ".htm",
    ".xml",
    ".csv",
    # Data / config
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    # Code
    ".py",
    ".js",
    ".ts",
    ".rs",
    ".go",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".sql",
    ".sh",
    ".rb",
    ".jsx",
    ".tsx",
    ".vue",
    ".svelte",
    # Logs
    ".log",
    # PDF
    ".pdf",
}


def read_text(path: Path) -> str:
    """Read a text-based file with lenient encoding."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise FileReadError(str(path), str(exc)) from exc


def read_pdf(path: Path) -> str:
    """Read a PDF file using PyMuPDF (fitz). Lazy-imported."""
    try:
        import fitz  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise FileReadError(str(path), "pymupdf not installed — cannot read PDFs") from exc

    try:
        doc = fitz.open(str(path))  # pyright: ignore[reportUnknownMemberType]
        pages: list[str] = []
        for page in doc:  # pyright: ignore[reportUnknownVariableType]
            page_text: str = str(
                page.get_text()  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
            )
            pages.append(page_text)
        doc.close()  # pyright: ignore[reportUnknownMemberType]
        text = "".join(pages)
    except Exception as exc:
        raise FileReadError(str(path), f"PDF read error: {exc}") from exc
    else:
        return text


def read_file(path: Path) -> str:
    """Dispatch to the correct reader based on file extension.

    Returns the file content as a string.
    Raises ``FileReadError`` for unsupported or unreadable files.
    """
    ext = path.suffix.lower()
    if ext == ".pdf":
        return read_pdf(path)
    if ext in SUPPORTED_EXTENSIONS:
        return read_text(path)
    # Attempt text read for unknown extensions
    try:
        return read_text(path)
    except FileReadError:
        raise
    except Exception as exc:
        raise FileReadError(str(path), f"Unsupported or unreadable file: {exc}") from exc


def list_supported_formats() -> list[str]:
    """Return sorted list of supported file extensions."""
    return sorted(SUPPORTED_EXTENSIONS)
