"""Local mem0 adapter contract tests (enabled only with local infrastructure)."""

from __future__ import annotations

import os

import pytest


@pytest.mark.integration
async def test_local_mem0_and_elasticsearch_contract_is_opt_in() -> None:
    """No network or service is reached unless the local integration suite enables it."""
    if os.environ.get("AGENTIC_RAG_TEST_MEM0_ENABLED") != "1":
        pytest.skip("set AGENTIC_RAG_TEST_MEM0_ENABLED=1 with local mem0/ES configured")
    pytest.fail("configure a local AsyncMemory fixture before enabling this contract")
