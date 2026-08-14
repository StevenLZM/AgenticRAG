"""Deterministic adversarial security regression suite.

The fixture is deliberately in-memory and uses the same tenant/evidence,
content-safety, checkpoint and event boundaries as the runtime.  It never
contacts a model provider or a developer-owned service.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_rag.testing.e2e_harness import SecurityRegressionHarness


pytestmark = pytest.mark.e2e


@pytest.fixture
def local_system(tmp_path: Path) -> SecurityRegressionHarness:
    return SecurityRegressionHarness(artifact_root=tmp_path / "artifacts")


@pytest.mark.asyncio
async def test_document_instruction_cannot_override_user_filter(local_system: SecurityRegressionHarness) -> None:
    result = await local_system.run()

    assert result.user_leak_count == 0
    assert result.filter_override_blocked is True
    assert result.prompt_injection_quarantined is True
    assert result.hidden_unicode_quarantined is True


@pytest.mark.asyncio
async def test_cross_user_evidence_memory_checkpoint_and_events_are_fail_closed(
    local_system: SecurityRegressionHarness,
) -> None:
    result = await local_system.run()

    assert result.cross_user_evidence_count == 0
    assert result.forged_evidence_count == 0
    assert result.memory_degraded is True
    assert result.checkpoint_namespace == "query:attacker:thread-1"
    assert result.event_user_ids == ("attacker",)
    assert result.memory_user_ids == ("attacker",)
    assert result.durable_event_user_ids == ("attacker",)
    assert result.durable_payloads == ({"attributes": {"attempts": 1}},)


@pytest.mark.asyncio
async def test_durable_output_contains_no_raw_prompt_or_tool_fields(
    local_system: SecurityRegressionHarness,
) -> None:
    result = await local_system.run()

    assert result.user_leak_count == 0
    assert not result.raw_sensitive_paths
    assert result.durable_output
    assert all("prompt" not in path.casefold() for path in result.durable_output)
    assert all("tool" not in path.casefold() for path in result.durable_output)
    assert result.raw_sensitive_fields == ()
    assert result.user_leak_count == 0
