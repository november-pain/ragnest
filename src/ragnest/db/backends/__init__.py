"""Backend factory — create the configured DatabaseBackend."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragnest.db.backends.postgres import PostgresBackend
from ragnest.db.backends.sqlite import SQLiteBackend
from ragnest.exceptions import ConfigError

if TYPE_CHECKING:
    from ragnest.config import DBSettings, StateSettings
    from ragnest.db.backend import DatabaseBackend


def create_backend(settings: DBSettings) -> DatabaseBackend:
    """Instantiate the backend selected by ``settings.backend``."""
    if settings.backend == "postgres":
        return PostgresBackend(settings.connection_string)
    if settings.backend == "supabase":
        msg = "Supabase backend not yet implemented"
        raise NotImplementedError(msg)
    # Unreachable for typed Literal values, but defensive fallback
    msg = f"Unknown database backend: {settings.backend}"  # pyright: ignore[reportUnreachable]
    raise ConfigError(msg)


def create_state_backend(settings: StateSettings) -> SQLiteBackend:
    """Create the SQLite backend for local state storage."""
    return SQLiteBackend(settings.path)
