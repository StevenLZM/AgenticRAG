"""Run the local API with bounded graceful SIGTERM handling."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from collections.abc import Sequence
from pathlib import Path

import uvicorn  # type: ignore[import-not-found]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from agentic_rag.api.app import create_app  # noqa: E402
from agentic_rag.config import Settings  # noqa: E402


LOGGER = logging.getLogger(__name__)


async def serve(*, host: str, port: int, grace_seconds: int) -> None:
    """Serve until SIGTERM/SIGINT drains in-flight HTTP work or grace expires.

    Uvicorn stops accepting new HTTP requests when ``should_exit`` is set. Query
    and ingestion workers persist their current graph node in the existing
    SQLite checkpoints, so interrupted work is resumed by their next process.
    """
    config = uvicorn.Config(
        create_app(Settings()),  # type: ignore[call-arg]
        host=host,
        port=port,
        log_level="info",
        timeout_graceful_shutdown=grace_seconds,
    )
    server = uvicorn.Server(config)

    def request_shutdown() -> None:
        if not server.should_exit:
            LOGGER.info(
                "SIGTERM received: refusing new API work and draining current requests"
            )
            server.should_exit = True

    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(signum, request_shutdown)
        except NotImplementedError:  # pragma: no cover - Windows only
            signal.signal(signum, lambda _signal, _frame: request_shutdown())
    await server.serve()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--grace-seconds", type=int, default=30)
    args = parser.parse_args(argv)
    if args.grace_seconds <= 0:
        parser.error("--grace-seconds must be positive")
    asyncio.run(serve(host=args.host, port=args.port, grace_seconds=args.grace_seconds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
