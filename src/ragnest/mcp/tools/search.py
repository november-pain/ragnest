"""Search tools — vector similarity search across knowledge bases."""

# pyright: reportUnusedFunction=false

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated

from fastmcp.exceptions import ToolError
from pydantic import Field

from ragnest.exceptions import (
    DocumentNotFoundError,
    EmbeddingError,
    KBNotFoundError,
    RagnestError,
)
from ragnest.mcp.formatting import format_search_results

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from ragnest.services.kb_service import KBService

logger = logging.getLogger(__name__)


def register_search_tools(mcp: FastMCP, kb_service: KBService) -> None:
    """Register search-related MCP tools."""

    @mcp.tool
    def search_kb(
        kb_name: Annotated[str, Field(description="Name of the knowledge base to search")],
        query: Annotated[str, Field(description="Search query text")],
        top_k: Annotated[int, Field(description="Max results to return")] = 5,
        threshold: Annotated[
            float | None,
            Field(description="Minimum similarity score (0-1). Omit to return all."),
        ] = None,
    ) -> str:
        """Search a single knowledge base using vector similarity.

        Returns ranked results with content snippets and similarity scores.
        Scores below 0.3 indicate weak matches.
        """
        try:
            results = kb_service.search(
                kb_name, query, top_k=top_k, threshold=threshold
            )
        except KBNotFoundError as e:
            raise ToolError(
                f"KB '{e.kb_name}' not found. Use list_kbs() to see available."
            ) from e
        except EmbeddingError as e:
            raise ToolError(f"Embedding error: {e}. Is Ollama running?") from e
        except RagnestError as e:
            raise ToolError(str(e)) from e
        else:
            return format_search_results(results)

    @mcp.tool
    def search_all_kbs(
        query: Annotated[str, Field(description="Search query text")],
        top_k_per_kb: Annotated[int, Field(description="Max results per KB")] = 3,
        threshold: Annotated[
            float | None,
            Field(description="Minimum similarity score (0-1). Omit to return all."),
        ] = None,
    ) -> str:
        """Search across all knowledge bases that contain data.

        Returns results grouped by KB name. Useful for broad discovery
        when you're unsure which KB has relevant content.
        """
        try:
            all_results = kb_service.search_all(
                query, top_k_per_kb=top_k_per_kb, threshold=threshold
            )
        except EmbeddingError as e:
            raise ToolError(f"Embedding error: {e}. Is Ollama running?") from e
        except RagnestError as e:
            raise ToolError(str(e)) from e
        else:
            if not all_results:
                return "No results found in any knowledge base."

            sections: list[str] = []
            for kb_name_key, results in all_results.items():
                sections.append(f"## {kb_name_key}\n{format_search_results(results)}")
            return "\n\n".join(sections)

    @mcp.tool
    def get_similar_documents(
        kb_name: Annotated[str, Field(description="Knowledge base name")],
        document_id: Annotated[str, Field(description="ID of the reference document")],
        top_k: Annotated[int, Field(description="Max similar documents to return")] = 5,
    ) -> str:
        """Find documents similar to a given document within a KB.

        Useful for discovering related content or finding duplicates.
        """
        try:
            results = kb_service.get_similar_documents(
                kb_name, document_id, top_k=top_k
            )
        except KBNotFoundError as e:
            raise ToolError(
                f"KB '{e.kb_name}' not found. Use list_kbs() to see available."
            ) from e
        except DocumentNotFoundError as e:
            raise ToolError(
                f"Document '{e.document_id}' not found."
            ) from e
        except EmbeddingError as e:
            raise ToolError(f"Embedding error: {e}. Is Ollama running?") from e
        except RagnestError as e:
            raise ToolError(str(e)) from e
        else:
            return format_search_results(results)
