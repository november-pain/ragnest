"""Unit tests for file_reader — dispatch, supported formats, text reading."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragnest.exceptions import FileReadError
from ragnest.services.file_reader import (
    SUPPORTED_EXTENSIONS,
    list_supported_formats,
    read_file,
    read_text,
)

SAMPLE_DOCS_DIR = Path(__file__).parent.parent / "fixtures" / "sample_docs"


# -- read_text --


class TestReadText:
    """read_text reads text-based files with UTF-8 encoding."""

    def test_reads_txt_file(self) -> None:
        content = read_text(SAMPLE_DOCS_DIR / "sample.txt")
        assert "sample text document" in content

    def test_reads_md_file(self) -> None:
        content = read_text(SAMPLE_DOCS_DIR / "sample.md")
        assert "# Sample Markdown" in content

    def test_raises_on_nonexistent_file(self) -> None:
        with pytest.raises(FileReadError, match="nonexistent"):
            read_text(Path("/nonexistent/file.txt"))


# -- read_file (dispatch) --


class TestReadFileDispatch:
    """read_file dispatches to the correct reader based on extension."""

    def test_dispatches_txt(self) -> None:
        content = read_file(SAMPLE_DOCS_DIR / "sample.txt")
        assert "sample text document" in content

    def test_dispatches_md(self) -> None:
        content = read_file(SAMPLE_DOCS_DIR / "sample.md")
        assert "Sample Markdown" in content

    def test_reads_file_with_supported_extension(self, tmp_path: Path) -> None:
        py_file = tmp_path / "test.py"
        py_file.write_text("print('hello')", encoding="utf-8")

        content = read_file(py_file)

        assert "print('hello')" in content

    def test_attempts_text_read_for_unknown_extension(self, tmp_path: Path) -> None:
        """Unknown extensions still attempt text read as fallback."""
        unknown_file = tmp_path / "data.xyz"
        unknown_file.write_text("some data", encoding="utf-8")

        content = read_file(unknown_file)

        assert content == "some data"


# -- list_supported_formats --


class TestListSupportedFormats:
    """list_supported_formats returns a sorted list of extensions."""

    def test_returns_sorted_list(self) -> None:
        formats = list_supported_formats()
        assert formats == sorted(formats)

    def test_contains_common_extensions(self) -> None:
        formats = list_supported_formats()
        for ext in [".txt", ".md", ".py", ".json", ".pdf"]:
            assert ext in formats

    def test_count_matches_set(self) -> None:
        formats = list_supported_formats()
        assert len(formats) == len(SUPPORTED_EXTENSIONS)


# -- SUPPORTED_EXTENSIONS --


class TestSupportedExtensions:
    """SUPPORTED_EXTENSIONS set contains expected categories."""

    def test_text_extensions(self) -> None:
        for ext in [".txt", ".md", ".html", ".htm", ".xml", ".csv"]:
            assert ext in SUPPORTED_EXTENSIONS

    def test_code_extensions(self) -> None:
        for ext in [".py", ".js", ".ts", ".rs", ".go", ".java"]:
            assert ext in SUPPORTED_EXTENSIONS

    def test_config_extensions(self) -> None:
        for ext in [".json", ".yaml", ".yml", ".toml", ".ini"]:
            assert ext in SUPPORTED_EXTENSIONS

    def test_pdf_included(self) -> None:
        assert ".pdf" in SUPPORTED_EXTENSIONS
