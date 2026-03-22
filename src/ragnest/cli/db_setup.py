"""Database setup CLI — create database, init schema, register KBs.

Usage::

    python -m ragnest.cli.db_setup
    python -m ragnest.cli.db_setup --config /path/to/config.yaml
"""

from __future__ import annotations

import argparse
import logging
from typing import TYPE_CHECKING

from ragnest.config import load_settings
from ragnest.db.backends import create_backend
from ragnest.db.repositories.knowledge_base import KBRepository
from ragnest.db.schema import (
    create_database_if_not_exists,
    create_vector_index,
    init_schema,
)
from ragnest.log import setup_logging

if TYPE_CHECKING:
    from ragnest.config import AppSettings

logger = logging.getLogger(__name__)


def _register_knowledge_bases(
    settings: AppSettings,
) -> None:
    """Register all configured KBs in the database."""
    backend = create_backend(settings.database)
    try:
        kb_repo = KBRepository(backend)
        for kb_name, kb_config in settings.knowledge_bases.items():
            created = kb_repo.create(kb_config)
            if created:
                logger.info(
                    "Registered KB: %s (model=%s, dim=%d)",
                    kb_name, kb_config.model, kb_config.dimensions,
                )
            else:
                # Update existing KB settings
                kb_repo.update(
                    kb_name,
                    description=kb_config.description,
                    chunk_size=kb_config.chunk_size,
                    chunk_overlap=kb_config.chunk_overlap,
                )
                logger.info("Updated KB: %s", kb_name)
    finally:
        backend.close()


def main() -> None:
    """Entry point for ``ragnest-db`` / ``python -m ragnest.cli.db_setup``."""
    parser = argparse.ArgumentParser(
        description="Ragnest DB Setup — initialize database and schema",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    setup_logging(level=args.log_level)

    settings = load_settings(args.config)

    # Step 1: Create database if it does not exist
    logger.info("Step 1: Checking/creating database '%s'", settings.database.name)
    create_database_if_not_exists(settings.database)

    # Step 2: Initialize schema (tables, indexes)
    logger.info("Step 2: Initializing schema")
    backend = create_backend(settings.database)
    try:
        init_schema(backend)
    finally:
        backend.close()

    # Step 3: Register configured knowledge bases
    if settings.knowledge_bases:
        logger.info(
            "Step 3: Registering %d knowledge base(s)",
            len(settings.knowledge_bases),
        )
        _register_knowledge_bases(settings)

        # Step 4: Create vector indexes for each KB
        logger.info("Step 4: Creating vector indexes")
        backend = create_backend(settings.database)
        try:
            for kb_name, kb_config in settings.knowledge_bases.items():
                create_vector_index(backend, kb_name, kb_config.dimensions)
        finally:
            backend.close()
    else:
        logger.info(
            "No knowledge bases configured — skipping KB registration"
        )

    logger.info("Database setup complete!")


if __name__ == "__main__":
    main()
