"""Base repository — common dependency for all repositories."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragnest.db.backend import DatabaseBackend


class BaseRepository:
    """Base class providing the shared database backend reference."""

    def __init__(self, backend: DatabaseBackend) -> None:
        self._backend = backend
