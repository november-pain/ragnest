"""Document tools — list and delete documents within knowledge bases."""

# pyright: reportUnusedFunction=false

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated

from fastmcp.exceptions import ToolError
from pydantic import Field

from ragnest.exceptions import (
    DocumentNotFoundError,
    KBNotFoundError,
    RagnestError,
)
from ragnest.mcp.formatting import format_document_list

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from ragnest.services.kb_service import KBService

logger = logging.getLogger(__name__)


def register_document_tools(mcp: FastMCP, kb_service: KBService) -> None:
    """Register document management MCP tools."""

    @mcp.tool
    def list_documents(
        kb_name: Annotated[str, Field(description="Knowledge base name")],
    ) -> str:
        """List all documents in a knowledge base with metadata.

        Shows filename, type, chunk count, and ingestion date.
        """
        try:
            docs = kb_service.list_documents(kb_name)
        except KBNotFoundError as e:
            raise ToolError(
                f"KB '{e.kb_name}' not found. Use list_kbs() to see available."
            ) from e
        except RagnestError as e:
            raise ToolError(str(e)) from e
        else:
            return format_document_list(docs)

    @mcp.tool
    def delete_document(
        document_id: Annotated[str, Field(description="Document ID to delete")],
    ) -> str:
        """Delete a document and all its chunks from the knowledge base.

        Use list_documents() to find document IDs. This is irreversible.
        """
        try:
            kb_service.delete_document(document_id)
        except DocumentNotFoundError as e:
            raise ToolError(
                f"Document '{e.document_id}' not found. "
                f"Use list_documents() to see available documents."
            ) from e
        except RagnestError as e:
            raise ToolError(str(e)) from e
        else:
            return f"Document '{document_id}' and its chunks have been deleted."
