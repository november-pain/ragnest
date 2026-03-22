"""Worker CLI — background ingestion entrypoint.

Usage::

    python -m ragnest.cli.worker --scan
    python -m ragnest.cli.worker --scan --kb legal_ua
    python -m ragnest.cli.worker --retry
    python -m ragnest.cli.worker --scan --dry-run
"""

from __future__ import annotations

import argparse
import signal
from typing import TYPE_CHECKING

from ragnest.app import Application
from ragnest.config import load_settings
from ragnest.log import setup_logging

if TYPE_CHECKING:
    import types


def main() -> None:
    """Entry point for ``ragnest-worker`` / ``python -m ragnest.cli.worker``."""
    parser = argparse.ArgumentParser(
        description="Ragnest Worker — background ingestion",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan watch paths for new/changed files",
    )
    parser.add_argument(
        "--kb",
        default=None,
        help="Process only this knowledge base",
    )
    parser.add_argument(
        "--retry",
        action="store_true",
        help="Re-queue failed files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--json-log",
        action="store_true",
        help="Emit JSON log lines instead of human-readable",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config YAML file",
    )

    args = parser.parse_args()

    setup_logging(level=args.log_level, json_format=args.json_log)

    settings = load_settings(args.config)
    app = Application(settings)

    # Graceful shutdown on SIGINT / SIGTERM
    def _handle_signal(
        _signum: int,
        _frame: types.FrameType | None,
    ) -> None:
        app.worker_service.request_shutdown()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        app.worker_service.run(
            scan=args.scan,
            kb_name=args.kb,
            retry=args.retry,
            dry_run=args.dry_run,
        )
    finally:
        app.close()


if __name__ == "__main__":
    main()
