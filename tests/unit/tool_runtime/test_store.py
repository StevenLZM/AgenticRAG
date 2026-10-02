"""SQLite persistence must keep budgets and results across process lifetimes."""

from __future__ import annotations

import sqlite3
import asyncio
from dataclasses import replace

import pytest

from agentic_rag.tool_runtime.models import ToolError, ToolResult
from agentic_rag.tool_runtime.runtime import RuntimeLimits, ToolRuntime
from agentic_rag.tool_runtime.store import InvocationStore
from tests.unit.tool_runtime.test_runtime import EchoAdapter, context, definition


async def test_restart_preserves_success_results_and_physical_call_budget(tmp_path):
    path = tmp_path / "tools.sqlite"
    ctx = context()
    first = ToolRuntime([EchoAdapter()], InvocationStore(path), RuntimeLimits(max_calls=1))
    result = await first.call("local.echo", {"message": "hello"}, ctx, "first")
    await first.aclose()
    adapter = EchoAdapter()
    resumed = ToolRuntime([adapter], InvocationStore(path), RuntimeLimits(max_calls=50))
    try:
        assert await resumed.call("local.echo", {"message": "hello"}, ctx, "first") == result
        assert (await resumed.call("local.echo", {"message": "hello"}, ctx, "new")).error_code == "tool_budget_exhausted"
        assert adapter.calls == 0
    finally:
        await resumed.aclose()


async def test_discovery_budget_survives_restart(tmp_path):
    path = tmp_path / "tools.sqlite"
    ctx = context()
    first = ToolRuntime([EchoAdapter()], InvocationStore(path), RuntimeLimits(max_discoveries=1))
    await first.discover("echo", ctx)
    await first.aclose()
    resumed = ToolRuntime([EchoAdapter()], InvocationStore(path))
    try:
        with pytest.raises(ToolError, match="discovery_budget_exhausted"):
            await resumed.discover("echo", ctx)
    finally:
        await resumed.aclose()


async def test_deadline_cannot_be_extended_by_new_context(tmp_path):
    path = tmp_path / "tools.sqlite"
    ctx = context()
    rt = ToolRuntime([EchoAdapter()], InvocationStore(path))
    try:
        await rt.discover("echo", ctx)
        await rt.discover("echo", replace(ctx, deadline=ctx.deadline + 1000))
        usage = await rt.store.get_usage(ctx)
        assert usage["deadline"] == ctx.deadline
        assert usage["discoveries"] == 2 and usage["calls"] == 0
    finally:
        await rt.aclose()


async def test_journal_does_not_store_raw_call_arguments(tmp_path):
    path = tmp_path / "tools.sqlite"
    adapter = EchoAdapter()
    adapter.result = {"ok": True}
    rt = ToolRuntime([adapter], InvocationStore(path))
    await rt.call("local.echo", {"message": "sensitive-payload-never-store-raw"}, context(), "one")
    await rt.aclose()
    with sqlite3.connect(path) as connection:
        dump = "\n".join(connection.iterdump())
    assert "sensitive-payload-never-store-raw" not in dump


async def test_expired_attempt_cannot_publish_success(tmp_path):
    store = InvocationStore(tmp_path / "tools.sqlite")
    ctx = context()
    try:
        claim = await store.reserve(ctx, definition(), "arguments-digest", "one", RuntimeLimits(call_timeout_seconds=0.01))
        await asyncio.sleep(0.02)
        result = ToolResult(call_id="one", tool_id="local.echo", source_kind="calculation",
                            status="success", data={"message": "late"}, observed_at="2026-10-02T00:00:00+00:00")
        assert not await store.finish(ctx, "one", claim.token, result)
    finally:
        await store.aclose()


async def test_uncertain_read_only_call_retries_after_lease_with_fencing(tmp_path):
    store = InvocationStore(tmp_path / "tools.sqlite")
    ctx = context()
    limits = RuntimeLimits(call_timeout_seconds=0.01)
    try:
        first = await store.reserve(ctx, definition(), "digest", "same", limits)
        with pytest.raises(ToolError, match="call_in_progress"):
            await store.reserve(ctx, definition(), "digest", "same", limits)
        await asyncio.sleep(0.02)
        second = await store.reserve(ctx, definition(), "digest", "same", limits)
        assert second.token != first.token
        result = ToolResult(call_id="same", tool_id="local.echo", source_kind="calculation",
                            status="success", data={"message": "ok"}, observed_at="2026-10-02T00:00:00+00:00")
        assert not await store.finish(ctx, "same", first.token, result)
        assert await store.finish(ctx, "same", second.token, result)
        assert (await store.get_usage(ctx))["calls"] == 2
    finally:
        await store.aclose()


async def test_separate_store_connections_cannot_overspend_run_budget(tmp_path):
    path = tmp_path / "tools.sqlite"
    stores = [InvocationStore(path), InvocationStore(path)]
    ctx = context()
    try:
        claims = await asyncio.gather(*(store.reserve(ctx, definition(), "digest", str(i), RuntimeLimits(max_calls=1))
                                        for i, store in enumerate(stores)), return_exceptions=True)
        assert len([result for result in claims if isinstance(result, ToolError) and result.code == "tool_budget_exhausted"]) == 1
        assert (await stores[0].get_usage(ctx))["calls"] == 1
    finally:
        for store in stores:
            await store.aclose()


async def test_interruption_revokes_only_owned_attempt_and_preserves_budget(tmp_path):
    store = InvocationStore(tmp_path / "tools.sqlite")
    ctx = context()
    limits = RuntimeLimits()
    try:
        interrupted = await store.reserve(ctx, definition(), "digest", "interrupted", limits)
        sibling = await store.reserve(ctx, definition(), "digest", "sibling", limits)
        assert not await store.interrupt(ctx, "interrupted", "not-the-owner")
        assert not await store.interrupt(replace(ctx, session_id="other"), "interrupted", interrupted.token)
        assert await store.interrupt(ctx, "interrupted", interrupted.token)
        result = ToolResult(call_id="interrupted", tool_id="local.echo", source_kind="calculation",
                            status="success", data={"message": "late"}, observed_at="2026-10-02T00:00:00+00:00")
        assert not await store.finish(ctx, "interrupted", interrupted.token, result)
        assert await store.finish(ctx, "sibling", sibling.token, result.model_copy(update={"call_id": "sibling"}))
        recovered = await store.reserve(ctx, definition(), "digest", "interrupted", limits)
        assert recovered.token != interrupted.token
        assert not await store.interrupt(ctx, "interrupted", interrupted.token)
        assert await store.finish(ctx, "interrupted", recovered.token, result)
        assert await store.get_usage(ctx) == {"calls": 3, "discoveries": 0, "deadline": ctx.deadline}
    finally:
        await store.aclose()
