"""Opt-in, disposable deterministic backend for chat browser acceptance.

Supply explicit local test DSNs (the same contract as real_query_fixture).
The MySQL database must be a dedicated agentic_rag_chat_test_* database.
Only localhost is served; fixture-owned rows, index and stream are cleaned on exit.
This utility never calls real model or Mem0 providers.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.engine import make_url
import uvicorn


async def serve(output: Path, port: int) -> None:
    from agentic_rag.persistence.repositories import documents
    from tests.fixtures.query_services import real_query_fixture

    dsn = os.environ.get("AGENTIC_RAG_TEST_MYSQL_DSN", "")
    if not dsn or not (make_url(dsn).database or "").startswith(
        "agentic_rag_chat_test_"
    ):
        raise ValueError(
            "an explicit disposable agentic_rag_chat_test_* database is required"
        )
    os.environ["AGENTIC_RAG_RUN_REAL_QUERY_E2E"] = "1"
    os.environ["AGENTIC_RAG_MEM0_ENABLED"] = "false"
    output.mkdir(parents=True, exist_ok=False)
    iterator = real_query_fixture.__wrapped__(output)
    fixture = await anext(iterator)
    try:
        generate = fixture._dependencies.generator.generate

        async def delayed_generation(*args, **kwargs):
            await asyncio.sleep(3)
            return await generate(*args, **kwargs)

        fixture._dependencies.generator.generate = delayed_generation
        async with fixture.container.repositories.session_factory() as db:
            document_id = (
                await db.execute(
                    select(documents.c.id).where(
                        documents.c.user_id == fixture.settings.default_user_id,
                    )
                )
            ).scalar_one()
        (output / "seeded-document-id").write_text(document_id, encoding="utf-8")
        print(f"Disposable preview ready: http://127.0.0.1:{port}", flush=True)
        server = uvicorn.Server(
            uvicorn.Config(
                fixture.app,
                host="127.0.0.1",
                port=port,
                lifespan="off",
                log_level="warning",
            )
        )
        await server.serve()
    finally:
        await iterator.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="new private fixture directory"
    )
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    asyncio.run(serve(args.output, args.port))
