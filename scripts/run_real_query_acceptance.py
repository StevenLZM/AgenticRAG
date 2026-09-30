"""Retired compatibility entry point for the former seeded Query acceptance run.

The historical command created a directly seeded document and used a fixed
embedding.  That makes it a protocol fixture, not a business-quality RAG
evaluation, so it is deliberately unable to produce a passing report.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path


_MIGRATION = (
    "scripts.run_real_query_acceptance is retired because its directly seeded "
    "fixture and fixed embedding are not real-RAG quality evidence. Run "
    "scripts/run_real_rag_evaluation.py --run-dir <isolated-run-dir> "
    "--stage evaluate instead."
)


class LegacyAcceptanceRetiredError(RuntimeError):
    """Raised whenever a caller attempts to use the retired quality entry point."""


async def run(output: Path | None = None) -> dict[str, object]:
    """Fail closed without creating output, resources, Queries, or judge calls."""
    del output
    raise LegacyAcceptanceRetiredError(_MIGRATION)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="legacy compatibility argument; no output is created",
    )
    args = parser.parse_args(argv)
    try:
        asyncio.run(run(args.output))
    except LegacyAcceptanceRetiredError as error:
        print(f"LEGACY QUERY ACCEPTANCE RETIRED: {error}", file=sys.stderr)
        return 2
    raise AssertionError("retired acceptance entry point unexpectedly returned")


if __name__ == "__main__":
    raise SystemExit(main())
