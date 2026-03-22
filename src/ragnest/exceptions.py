"""Custom exception hierarchy for Ragnest."""


class RagnestError(Exception):
    """Base exception for all Ragnest errors."""


class ConfigError(RagnestError):
    """Configuration is invalid or missing."""


class KBNotFoundError(RagnestError):
    """Knowledge base does not exist."""

    def __init__(self, kb_name: str) -> None:
        self.kb_name = kb_name
        super().__init__(f"Knowledge base '{kb_name}' not found")


class KBAlreadyExistsError(RagnestError):
    """Knowledge base with this name already exists."""

    def __init__(self, kb_name: str) -> None:
        self.kb_name = kb_name
        super().__init__(f"Knowledge base '{kb_name}' already exists")


class DocumentNotFoundError(RagnestError):
    """Document does not exist."""

    def __init__(self, document_id: str) -> None:
        self.document_id = document_id
        super().__init__(f"Document '{document_id}' not found")


class BatchNotFoundError(RagnestError):
    """Batch does not exist."""

    def __init__(self, batch_id: str) -> None:
        self.batch_id = batch_id
        super().__init__(f"Batch '{batch_id}' not found")


class BatchAlreadyUndoneError(RagnestError):
    """Batch was already undone."""

    def __init__(self, batch_id: str) -> None:
        self.batch_id = batch_id
        super().__init__(f"Batch '{batch_id}' was already undone")


class EmbeddingError(RagnestError):
    """Failed to generate embeddings."""


class FileReadError(RagnestError):
    """Failed to read a file for ingestion."""

    def __init__(self, file_path: str, reason: str) -> None:
        self.file_path = file_path
        self.reason = reason
        super().__init__(f"Failed to read '{file_path}': {reason}")


class DatabaseError(RagnestError):
    """Database operation failed."""
