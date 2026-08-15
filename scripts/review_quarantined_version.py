#!/usr/bin/env python3
"""Approve or reject one quarantined Version from a trusted local terminal."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from agentic_rag.config import Settings  # noqa: E402
from agentic_rag.ingestion.worker import SqlAlchemyIngestionJobStore  # noqa: E402
from agentic_rag.persistence.mysql import (  # noqa: E402
    create_mysql_engine,
    create_session_factory,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Trusted local review of one quarantined document version"
    )
    value.add_argument("--version-id", required=True)
    decision = value.add_mutually_exclusive_group(required=True)
    decision.add_argument("--approve", action="store_true")
    decision.add_argument("--reject", action="store_true")
    return value


async def review(version_id: str, *, approve: bool, settings: Settings) -> str:
    engine = create_mysql_engine(settings.mysql_dsn, pool_pre_ping=True)
    try:
        store = SqlAlchemyIngestionJobStore(create_session_factory(engine))
        return await store.review_quarantine(version_id, approve=approve)
    finally:
        await engine.dispose()


def main() -> int:
    args = parser().parse_args()
    try:
        job_id = asyncio.run(
            review(
                args.version_id,
                approve=bool(args.approve),
                settings=Settings(),  # type: ignore[call-arg]
            )
        )
    except KeyError:
        print("version is not quarantined or does not exist", file=sys.stderr)
        return 2
    print(f"job_id={job_id} decision={'approved' if args.approve else 'rejected'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
