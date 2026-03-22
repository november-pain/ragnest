"""Export service — export knowledge bases to portable formats.

Reads chunk data from the vector backend (which now has inline metadata).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragnest.exceptions import RagnestError

if TYPE_CHECKING:
    from ragnest.app import BackendRegistry

logger = logging.getLogger(__name__)


class ExportService:
    """Export knowledge base data to Parquet or JSON."""

    def __init__(self, registry: BackendRegistry) -> None:
        self._registry = registry

    def export_kb(
        self,
        kb_name: str,
        output_dir: str,
        fmt: str = "parquet",
        backend_name: str = "default",
    ) -> str:
        """Export a knowledge base to the specified format.

        Args:
            kb_name: Knowledge base to export.
            output_dir: Directory for output files.
            fmt: Output format — ``"parquet"`` or ``"json"``.
            backend_name: Which vector backend holds the chunks.

        Returns:
            Path to the exported file.

        Raises:
            KBNotFoundError: If the knowledge base has no chunks.
            RagnestError: For empty KBs or unsupported formats.
        """
        backend = self._registry.get(backend_name)

        with backend.cursor() as cur:
            # Read chunks with inline metadata (no JOIN needed)
            cur.execute(
                "SELECT c.id, c.content, c.chunk_index, c.metadata, "
                "c.embedding::text, c.filename, c.source_path "
                "FROM chunks c "
                "WHERE c.kb_name = %s ORDER BY c.filename, c.chunk_index",
                (kb_name,),
            )
            rows = cur.fetchall()

        if not rows:
            msg = f"Knowledge base '{kb_name}' is empty or not found"
            raise RagnestError(msg)

        data: list[dict[str, Any]] = []
        for row in rows:
            embedding_text: str = row[4]
            embedding = [float(x) for x in embedding_text.strip("[]").split(",")]
            data.append(
                {
                    "chunk_id": row[0],
                    "content": row[1],
                    "chunk_index": row[2],
                    "metadata": str(row[3]),
                    "embedding": embedding,
                    "filename": row[5],
                    "source_path": row[6],
                }
            )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if fmt == "parquet":
            file_path = self._write_parquet(output_path, kb_name, data)
        elif fmt == "json":
            file_path = self._write_json(output_path, kb_name, data)
        else:
            msg = f"Unsupported export format: {fmt}"
            raise RagnestError(msg)

        # Write metadata sidecar
        meta_path = output_path / f"{kb_name}.meta.json"
        meta_path.write_text(
            json.dumps(
                {
                    "kb_name": kb_name,
                    "chunk_count": len(data),
                    "format": fmt,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        logger.info(
            "Exported %d chunks to %s",
            len(data),
            file_path,
        )
        return str(file_path)

    def _write_parquet(self, output_path: Path, kb_name: str, data: list[dict[str, Any]]) -> Path:
        """Write data as Parquet — lazy-imports pandas/pyarrow."""
        import pandas as pd  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

        file_path = output_path / f"{kb_name}.parquet"
        df = pd.DataFrame(data)  # pyright: ignore[reportUnknownMemberType]
        df.to_parquet(str(file_path), index=False)  # pyright: ignore[reportUnknownMemberType]
        return file_path

    def _write_json(self, output_path: Path, kb_name: str, data: list[dict[str, Any]]) -> Path:
        """Write data as JSON — lazy-imports pandas."""
        import pandas as pd  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

        file_path = output_path / f"{kb_name}.json"
        df = pd.DataFrame(data)  # pyright: ignore[reportUnknownMemberType]
        df.to_json(  # pyright: ignore[reportUnknownMemberType]
            str(file_path), orient="records", indent=2
        )
        return file_path
