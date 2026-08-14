"""Opt-in Mem0 provider E2E entry point."""

from __future__ import annotations

import pytest

from tests.integration.memory.test_mem0_adapter import _run_real_mem0_contract


@pytest.mark.e2e
@pytest.mark.integration
async def test_mem0_real_services_preserve_scope_and_delete() -> None:
    await _run_real_mem0_contract()
