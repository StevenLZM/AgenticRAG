"""Runnable Query Worker entrypoint contracts."""

from __future__ import annotations

import inspect

from scripts.run_query_worker import run
from agentic_rag.runtime.query_composition import build_query_dependencies


def test_query_worker_run_uses_production_composition_by_default() -> None:
    parameter = inspect.signature(run).parameters["dependencies_factory"]
    assert parameter.default is build_query_dependencies

