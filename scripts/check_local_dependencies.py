#!/usr/bin/env python3
"""Check configured local dependencies without invoking paid model APIs."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pydantic import ValidationError  # noqa: E402

from agentic_rag.bootstrap import build_container  # noqa: E402
from agentic_rag.config import Settings  # noqa: E402


async def _check(settings: Settings) -> int:
    try:
        container = build_container(settings)
    except Exception as error:
        print(f"bootstrap: unavailable ({type(error).__name__})")
        return 1

    try:
        dependencies = await container.readiness_checks.run()
    finally:
        await container.close()

    for name, state in dependencies.items():
        print(f"{name}: {state}")
    return 0 if all(state == "available" for state in dependencies.values()) else 1


def main() -> int:
    try:
        settings = Settings()  # type: ignore[call-arg]
    except ValidationError as error:
        fields = sorted(
            ".".join(str(part) for part in item["loc"]) for item in error.errors()
        )
        print(f"configuration: unavailable (missing or invalid: {', '.join(fields)})")
        return 1
    return asyncio.run(_check(settings))


if __name__ == "__main__":
    raise SystemExit(main())
