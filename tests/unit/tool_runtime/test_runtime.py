"""Behavior at the shared tool execution and recovery boundary."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from agentic_rag.domain.models import UserScope
from agentic_rag.runtime.models import RuntimeConfigSnapshot
from agentic_rag.tool_runtime.models import ToolContext, ToolDefinition, ToolError
from agentic_rag.tool_runtime.runtime import RuntimeLimits, ToolRuntime
from agentic_rag.tool_runtime.store import InvocationStore


SNAPSHOT = RuntimeConfigSnapshot(
    app_version="test", graph_version="test", prompt_version="test",
    main_model_id="main", light_model_id="light", embedding_model="embedding",
    embedding_dimensions=1024, reranker_version="test", retrieval_config_version="test",
    index_generation="test", memory_config_version="test",
)


def context(**overrides):
    return replace(ToolContext(
        scope=UserScope(user_id="alice"), run_id="run-1", session_id="session-1",
        snapshot=SNAPSHOT, deadline=time.time() + 60,
    ), **overrides)


def definition(**overrides):
    return ToolDefinition(**{
        "tool_id": "local.echo", "adapter_id": "local", "name": "echo",
        "description": "Echo a message", "source_kind": "calculation", "version": "1",
        "input_schema": {"type": "object", "properties": {"message": {"type": "string"}},
                         "required": ["message"], "additionalProperties": False},
        **overrides,
    })


class EchoAdapter:
    adapter_id = "local"

    def __init__(self):
        self.tools = (definition(),)
        self.calls = 0
        self.closed = False
        self.error = None
        self.result = None
        self.entered = asyncio.Event()
        self.release = None

    async def list_tools(self, ctx):
        return self.tools

    async def call_tool(self, tool, args, ctx):
        self.calls += 1
        self.entered.set()
        if self.release is not None:
            await self.release.wait()
        if self.error is not None:
            raise self.error
        return self.result if self.result is not None else {"message": args["message"], "user": ctx.scope.user_id}

    async def aclose(self):
        self.closed = True


@pytest.fixture
async def runtime(tmp_path):
    adapter = EchoAdapter()
    runtime = ToolRuntime([adapter], InvocationStore(tmp_path / "tools.sqlite"))
    yield runtime, adapter
    await runtime.aclose()


async def test_discovery_filters_capabilities_and_bounds_results(runtime):
    rt, adapter = runtime
    adapter.tools = tuple(definition(tool_id=f"local.route{i}", name=f"route{i}",
                                     description="地图路线查询", capabilities=("maps", "route")) for i in range(8))
    assert len(await rt.discover("地图路线", context(), limit=100)) == 5
    assert await rt.discover("unrelated astronomy", context()) == ()


async def test_discovery_isolates_unavailable_remote_service(tmp_path):
    class Broken:
        adapter_id = "mcp.broken"

        async def list_tools(self, ctx):
            raise RuntimeError("Authorization: Bearer secret")

    rt = ToolRuntime([Broken(), EchoAdapter()], InvocationStore(tmp_path / "tools.sqlite"))
    try:
        assert (await rt.discover("echo", context()))[0].tool_id == "local.echo"
    finally:
        await rt.aclose()


@pytest.mark.parametrize("code,retryable", [("authentication_failed", False), ("tool_unavailable", True)])
async def test_discovery_surfaces_relevant_failed_service_without_breaking_native_tools(tmp_path, code, retryable):
    class BrokenMaps:
        adapter_id = "mcp.amap_maps"
        config = SimpleNamespace(capabilities=("maps.route", "maps.places"))

        async def list_tools(self, ctx):
            raise ToolError(code, retryable=retryable)

    rt = ToolRuntime([BrokenMaps(), EchoAdapter()], InvocationStore(tmp_path / "tools.sqlite"))
    try:
        with pytest.raises(ToolError) as failure:
            await rt.discover("maps.route", context())
        assert failure.value.code == code and failure.value.retryable == retryable
        assert (await rt.discover("echo", context()))[0].tool_id == "local.echo"
        assert await rt.discover("unrelated astronomy", context()) == ()
    finally:
        await rt.aclose()


async def test_discovery_error_relevance_uses_configured_capabilities_not_only_service_name(tmp_path):
    class Broken:
        adapter_id = "mcp.vendor1"
        config = SimpleNamespace(capabilities=("weather.current",))

        async def list_tools(self, ctx):
            raise ToolError("authentication_failed")

    rt = ToolRuntime([Broken(), EchoAdapter()], InvocationStore(tmp_path / "tools.sqlite"))
    try:
        with pytest.raises(ToolError, match="authentication_failed"):
            await rt.discover("weather.current", context())
        assert await rt.discover("maps.route", context()) == ()
    finally:
        await rt.aclose()


async def test_sole_unavailable_service_does_not_claim_unrelated_capabilities(tmp_path):
    class Broken:
        adapter_id = "mcp.weather"

        async def list_tools(self, ctx):
            raise ToolError("authentication_failed")

    rt = ToolRuntime([Broken()], InvocationStore(tmp_path / "tools.sqlite"))
    try:
        assert await rt.discover("astronomy", context()) == ()
        with pytest.raises(ToolError, match="authentication_failed"):
            await rt.discover("weather", context())
    finally:
        await rt.aclose()


@pytest.mark.parametrize("arguments", [{"message": 2}, {"message": "ok", "user_id": "bob"}, {"message": "ok", "api_key": "secret"}])
async def test_invalid_arguments_never_execute_upstream(runtime, arguments):
    rt, adapter = runtime
    result = await rt.call("local.echo", arguments, context(), "invalid")
    assert result.error_code == "invalid_arguments"
    assert adapter.calls == 0


async def test_known_tool_does_not_require_discovery_and_duplicate_replays(runtime):
    rt, adapter = runtime
    ctx = context()
    first = await rt.call("local.echo", {"message": "hello"}, ctx, "stable-id")
    replay = await rt.call("local.echo", {"message": "hello"}, ctx, "stable-id")
    assert first.status == "success" and first.data == {"message": "hello", "user": "alice"}
    assert replay == first
    assert adapter.calls == 1


async def test_changed_call_arguments_are_rejected(runtime):
    rt, adapter = runtime
    ctx = context()
    await rt.call("local.echo", {"message": "first"}, ctx, "stable-id")
    result = await rt.call("local.echo", {"message": "second"}, ctx, "stable-id")
    assert result.error_code == "call_id_conflict"
    assert adapter.calls == 1


async def test_cached_result_is_not_replayed_after_tool_revocation(runtime):
    rt, adapter = runtime
    ctx = context()
    await rt.call("local.echo", {"message": "hello"}, ctx, "stable-id")
    adapter.tools = ()
    result = await rt.call("local.echo", {"message": "hello"}, ctx, "stable-id")
    assert result.error_code == "tool_not_found"


async def test_changed_definition_cannot_replay_old_result(runtime):
    rt, adapter = runtime
    ctx = context()
    await rt.call("local.echo", {"message": "hello"}, ctx, "stable-id")
    adapter.tools = (definition(version="2"),)
    result = await rt.call("local.echo", {"message": "hello"}, ctx, "stable-id")
    assert result.error_code == "definition_changed"
    assert adapter.calls == 1


async def test_user_scope_is_part_of_idempotency_key(runtime):
    rt, adapter = runtime
    alice = await rt.call("local.echo", {"message": "hello"}, context(), "same")
    bob = await rt.call("local.echo", {"message": "hello"}, context(scope=UserScope(user_id="bob")), "same")
    assert alice.data["user"] == "alice" and bob.data["user"] == "bob"
    assert adapter.calls == 2


async def test_run_cannot_cross_session_boundaries(runtime):
    rt, adapter = runtime
    await rt.call("local.echo", {"message": "hello"}, context(), "first")
    result = await rt.call("local.echo", {"message": "hello"}, context(session_id="other"), "second")
    assert result.error_code == "context_mismatch"
    assert adapter.calls == 1


async def test_simultaneous_duplicate_is_not_physically_replayed(runtime):
    rt, adapter = runtime
    adapter.release = asyncio.Event()
    ctx = context()
    task = asyncio.create_task(rt.call("local.echo", {"message": "hello"}, ctx, "same"))
    await adapter.entered.wait()
    repeated = await rt.call("local.echo", {"message": "hello"}, ctx, "same")
    adapter.release.set()
    assert (await task).status == "success"
    assert repeated.error_code == "call_in_progress" and repeated.retryable
    assert adapter.calls == 1


async def test_interrupted_call_resumes_same_id_without_resetting_budget(runtime):
    rt, adapter = runtime
    adapter.release = asyncio.Event()
    ctx = context()
    task = asyncio.create_task(rt.call("local.echo", {"message": "hello"}, ctx, "same"))
    await adapter.entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await rt.store.get_usage(ctx) == {"calls": 1, "discoveries": 0, "deadline": ctx.deadline}
    adapter.release.set()
    result = await rt.call("local.echo", {"message": "hello"}, ctx, "same")
    assert result.status == "success" and result.data["message"] == "hello"
    assert adapter.calls == 2
    assert await rt.store.get_usage(ctx) == {"calls": 2, "discoveries": 0, "deadline": ctx.deadline}
    assert await rt.call("local.echo", {"message": "hello"}, ctx, "same") == result
    assert adapter.calls == 2


async def test_timeout_is_bounded_and_structured(tmp_path):
    adapter = EchoAdapter()
    adapter.release = asyncio.Event()
    rt = ToolRuntime([adapter], InvocationStore(tmp_path / "tools.sqlite"), RuntimeLimits(call_timeout_seconds=0.02))
    try:
        result = await rt.call("local.echo", {"message": "hello"}, context(), "slow")
        assert result.error_code == "tool_timeout" and result.retryable
    finally:
        await rt.aclose()


async def test_expired_run_never_executes(runtime):
    rt, adapter = runtime
    result = await rt.call("local.echo", {"message": "hello"}, context(deadline=time.time() - 1), "late")
    assert result.error_code == "deadline_exceeded"
    assert adapter.calls == 0


@pytest.mark.parametrize("output", [{"value": float("nan")}, {"value": object()}, {1: "invalid"}, {"value": "x" * 70_000}])
async def test_non_json_or_oversized_results_fail_closed(runtime, output):
    rt, adapter = runtime
    adapter.result = output
    result = await rt.call("local.echo", {"message": "hello"}, context(), "bad-result")
    assert result.status == "error" and result.error_code in {"invalid_tool_result", "result_too_large"}
    assert result.data == {}
    json.dumps(result.model_dump(), allow_nan=False)


async def test_raw_adapter_errors_never_enter_observations(runtime):
    rt, adapter = runtime
    adapter.error = RuntimeError("request failed: Authorization: Bearer private-secret")
    result = await rt.call("local.echo", {"message": "hello"}, context(), "error")
    assert result.error_code == "tool_unavailable"
    assert "private-secret" not in result.model_dump_json()


async def test_structured_adapter_errors_are_preserved(runtime):
    rt, adapter = runtime
    adapter.error = ToolError("rate_limited", retryable=True)
    result = await rt.call("local.echo", {"message": "hello"}, context(), "error")
    assert result.error_code == "rate_limited" and result.retryable


async def test_remote_schema_refs_are_rejected_without_fetch(runtime):
    rt, adapter = runtime
    adapter.tools = (definition(input_schema={"type": "object", "$ref": "https://attacker.invalid/schema"}),)
    result = await rt.call("local.echo", {"message": "hello"}, context(), "unsafe-schema")
    assert result.error_code == "invalid_tool_schema"
    assert adapter.calls == 0


async def test_close_closes_adapters_and_rejects_future_calls(tmp_path):
    adapter = EchoAdapter()
    rt = ToolRuntime([adapter], InvocationStore(tmp_path / "tools.sqlite"))
    await rt.aclose()
    assert adapter.closed
    result = await rt.call("local.echo", {"message": "hello"}, context(), "closed")
    assert result.error_code == "runtime_closed"


async def test_document_payload_is_bounded_after_allowing_existing_retrieval_batches(runtime):
    rt, adapter = runtime
    adapter.tools = (definition(source_kind="document"),)
    adapter.result = {"batch": {"content": "文" * 100_000}}
    good = await rt.call("local.echo", {"message": "hello"}, context(), "large-document")
    assert good.status == "success"
    adapter.result = {"batch": {"content": "x" * 1_100_000}}
    bad = await rt.call("local.echo", {"message": "hello"}, context(), "oversize-document")
    assert bad.error_code == "result_too_large"


async def test_local_description_never_lists_remote_services(tmp_path):
    class Unrelated:
        adapter_id = "mcp.other"
        calls = 0

        async def list_tools(self, ctx):
            self.calls += 1
            raise AssertionError("unrelated remote catalog was accessed")

    unrelated = Unrelated()
    rt = ToolRuntime([unrelated, EchoAdapter()], InvocationStore(tmp_path / "tools.sqlite"))
    try:
        assert (await rt.describe("local.echo", context())).name == "echo"
        assert unrelated.calls == 0
    finally:
        await rt.aclose()


async def test_reserved_fields_are_blocked_even_when_remote_schema_allows_them(runtime):
    rt, adapter = runtime
    adapter.tools = (definition(input_schema={"type": "object"}),)
    result = await rt.call("local.echo", {"message": "hello", "credentials": {"token": "secret"}}, context(), "inject")
    assert result.error_code == "invalid_arguments" and adapter.calls == 0


async def test_invalid_schema_type_is_not_silently_treated_as_valid(runtime):
    rt, adapter = runtime
    adapter.tools = (definition(input_schema={"type": "object", "properties": {"message": {"type": "nonsense"}}}),)
    result = await rt.call("local.echo", {"message": "hello"}, context(), "bad-schema")
    assert result.error_code == "invalid_tool_schema" and adapter.calls == 0


async def test_catalog_resolution_and_execution_share_single_call_timeout(tmp_path):
    class Slow(EchoAdapter):
        async def list_tools(self, ctx):
            await asyncio.sleep(0.04)
            return self.tools

        async def call_tool(self, tool, args, ctx):
            await asyncio.sleep(0.04)
            return {"message": "completed too late"}

    rt = ToolRuntime([Slow()], InvocationStore(tmp_path / "tools.sqlite"), RuntimeLimits(call_timeout_seconds=0.06))
    try:
        result = await rt.call("local.echo", {"message": "hello"}, context(), "slow-all")
        assert result.error_code == "tool_timeout"
    finally:
        await rt.aclose()


async def test_cancelled_adapter_cannot_publish_suppressed_cancellation(runtime):
    rt, adapter = runtime
    original_call = adapter.call_tool

    async def suppress(tool, args, ctx):
        adapter.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return {"message": "late result"}

    adapter.call_tool = suppress
    ctx = context()
    task = asyncio.create_task(rt.call("local.echo", {"message": "hello"}, ctx, "cancel"))
    await adapter.entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    adapter.call_tool = original_call
    recovered = await rt.call("local.echo", {"message": "hello"}, ctx, "cancel")
    assert recovered.status == "success" and recovered.data["message"] == "hello"
    assert (await rt.store.get_usage(ctx))["calls"] == 2


async def test_catalog_cancellation_cannot_be_swallowed_before_execution(runtime):
    rt, adapter = runtime
    original_list = adapter.list_tools

    async def suppress(ctx):
        adapter.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return adapter.tools

    adapter.list_tools = suppress
    ctx = context()
    task = asyncio.create_task(rt.call("local.echo", {"message": "hello"}, ctx, "cancel-list"))
    await adapter.entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert adapter.calls == 0
    adapter.list_tools = original_list
    assert (await rt.call("local.echo", {"message": "hello"}, ctx, "cancel-list")).status == "success"
    assert (await rt.store.get_usage(ctx))["calls"] == 1


@pytest.mark.parametrize("operation", ["discover", "describe"])
async def test_catalog_interruption_does_not_cancel_resumable_run(runtime, operation):
    rt, adapter = runtime
    original_list = adapter.list_tools

    async def blocked(ctx):
        adapter.entered.set()
        await asyncio.Event().wait()

    adapter.list_tools = blocked
    ctx = context()
    invocation = rt.discover("echo", ctx) if operation == "discover" else rt.describe("local.echo", ctx)
    task = asyncio.create_task(invocation)
    await adapter.entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    adapter.list_tools = original_list
    assert (await rt.call("local.echo", {"message": "hello"}, ctx, "resume")).status == "success"
    assert await rt.store.get_usage(ctx) == {
        "calls": 1, "discoveries": 1 if operation == "discover" else 0, "deadline": ctx.deadline,
    }


async def test_child_interruption_does_not_poison_parent_or_sibling_attempt(runtime):
    rt, adapter = runtime
    entered = {name: asyncio.Event() for name in ("parent", "child")}
    released = {name: asyncio.Event() for name in ("parent", "child")}

    async def independent(tool, args, ctx):
        adapter.calls += 1
        entered[args["message"]].set()
        await released[args["message"]].wait()
        return {"message": args["message"]}

    adapter.call_tool = independent
    ctx = context()
    parent = asyncio.create_task(rt.call("local.echo", {"message": "parent"}, ctx, "parent-call"))
    await entered["parent"].wait()
    child = asyncio.create_task(rt.call("local.echo", {"message": "child"}, ctx, "child-call"))
    await entered["child"].wait()
    child.cancel()
    with pytest.raises(asyncio.CancelledError):
        await child
    released["parent"].set()
    assert (await parent).status == "success"
    released["child"].set()
    assert (await rt.call("local.echo", {"message": "child"}, ctx, "child-call")).status == "success"
    assert (await rt.store.get_usage(ctx))["calls"] == 3


async def test_explicit_durable_cancellation_still_blocks_inflight_publication(runtime):
    rt, adapter = runtime
    adapter.release = asyncio.Event()
    ctx = context()
    task = asyncio.create_task(rt.call("local.echo", {"message": "hello"}, ctx, "cancelled"))
    await adapter.entered.wait()
    await rt.store.cancel_run(ctx)
    adapter.release.set()
    result = await task
    assert result.status == "error" and result.data == {}
    assert (await rt.call("local.echo", {"message": "hello"}, ctx, "cancelled")).error_code == "run_cancelled"
    assert adapter.calls == 1


async def test_shutdown_interruption_can_resume_after_runtime_restart(tmp_path):
    adapter = EchoAdapter()
    adapter.release = asyncio.Event()
    path = tmp_path / "tools.sqlite"
    ctx = context()
    first = ToolRuntime([adapter], InvocationStore(path))
    task = asyncio.create_task(first.call("local.echo", {"message": "hello"}, ctx, "restart"))
    await adapter.entered.wait()
    await first.aclose()
    with pytest.raises(asyncio.CancelledError):
        await task
    resumed = ToolRuntime([EchoAdapter()], InvocationStore(path))
    try:
        result = await resumed.call("local.echo", {"message": "hello"}, ctx, "restart")
        assert result.status == "success"
        assert await resumed.store.get_usage(ctx) == {"calls": 2, "discoveries": 0, "deadline": ctx.deadline}
    finally:
        await resumed.aclose()
