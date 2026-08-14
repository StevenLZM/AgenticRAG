"""Command-line validator for the fixed Phase 5 JSONL datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evals.models import validate_dataset_directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Agentic RAG evaluation datasets")
    parser.add_argument("directory", type=Path, help="directory containing the three JSONL datasets")
    args = parser.parse_args(argv)
    try:
        counts = validate_dataset_directory(args.directory)
    except ValueError as error:
        print(f"dataset validation failed: {error}", file=sys.stderr)
        return 1
    for name, count in counts.items():
        print(f"{name}: {count} cases")
    print("dataset validation passed")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI command
    raise SystemExit(main())
