"""Exercise the pinned MCP client over local HTTP/SSE protocol fixtures."""

import asyncio
import json
import socket
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest
import uvicorn
from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from agentic_rag.domain.models import UserScope
from agentic_rag.tool_runtime.credentials import EnvironmentCredentialProvider
from agentic_rag.tool_runtime.mcp import McpAdapter, McpServerConfig
from agentic_rag.tool_runtime.models import ToolError


SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
    "additionalProperties": False,
}


def context(user="alice", timeout=5):
    return SimpleNamespace(
        scope=UserScope(user_id=user),
        run_id="run",
        session_id="chat",
        snapshot=None,
        deadline=time.time() + timeout,
    )


class ProtocolFixture:
    def __init__(self):
        self.requests = []
        self.queues = {}
        self.tools = [
            {
                "name": "search",
                "description": "Find a place",
                "inputSchema": SCHEMA,
                "annotations": {"readOnlyHint": True},
            },
            {"name": "unapproved", "inputSchema": SCHEMA},
            {
                "name": "write",
                "inputSchema": SCHEMA,
                "annotations": {"readOnlyHint": False},
            },
        ]
        self.result = {
            "content": [{"type": "text", "text": "place found"}],
            "structuredContent": {"places": [{"name": "Park"}]},
            "isError": False,
        }
        self.status = 200
        self.redirect = None
        self.sse_endpoint = None
        self.delay = 0

    async def rpc(self, request: Request):
        self.requests.append((request.method, request.url.path, dict(request.headers)))
        if self.redirect:
            return Response(status_code=307, headers={"location": self.redirect})
        if self.status != 200:
            return JSONResponse({"error": "fixture-secret"}, status_code=self.status)
        if request.method == "DELETE":
            return Response(status_code=200)
        if request.method == "GET":
            return Response(status_code=405)
        message = await request.json()
        if "id" not in message:
            return Response(status_code=202)
        method = message["method"]
        if method == "initialize":
            result = {
                "protocolVersion": message["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fixture", "version": "1"},
            }
        elif method == "tools/list":
            result = {"tools": self.tools}
        elif method == "tools/call":
            await asyncio.sleep(self.delay)
            result = self.result
        else:
            result = {}
        payload = {"jsonrpc": "2.0", "id": message["id"], "result": result}
        session_id = request.query_params.get("session_id")
        if session_id:
            await self.queues[session_id].put(payload)
            return Response(status_code=202)
        return JSONResponse(
            payload,
            headers={
                "mcp-session-id": request.headers.get("mcp-session-id", str(uuid4()))
            },
        )

    async def sse(self, request):
        self.requests.append(("GET", "/sse", dict(request.headers)))
        session_id = str(uuid4())
        queue = self.queues[session_id] = asyncio.Queue()

        async def events():
            endpoint = self.sse_endpoint or f"/messages?session_id={session_id}"
            yield f"event: endpoint\ndata: {endpoint}\n\n"
            while True:
                payload = await queue.get()
                yield f"event: message\ndata: {json.dumps(payload)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")


@asynccontextmanager
async def serve_protocol():
    fixture = ProtocolFixture()
    app = Starlette(
        routes=[
            Route("/mcp", fixture.rpc, methods=["POST", "GET", "DELETE"]),
            Route("/sse", fixture.sse),
            Route("/messages", fixture.rpc, methods=["POST"]),
        ]
    )
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="critical", lifespan="off"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        while not server.started:
            await asyncio.sleep(0.005)
        yield fixture, f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 3)
        sock.close()


def config(url, **kwargs):
    return McpServerConfig(
        enabled=True,
        id="maps",
        url=url,
        allow_loopback=True,
        allowed_tools=("search", "write"),
        capabilities=("maps",),
        **kwargs,
    )


@pytest.mark.parametrize(
    "transport,path", [("streamable_http", "/mcp"), ("sse", "/sse")]
)
async def test_sdk_discovery_call_credentials_and_session_isolation(transport, path):
    provider = EnvironmentCredentialProvider(
        bindings={"maps": "env:TOKEN"}, overrides={"TOKEN": "fixture-secret"}
    )
    async with serve_protocol() as (fixture, base):
        adapter = McpAdapter(
            config(
                base + path,
                transport=transport,
                auth_type="bearer",
                credential_ref="env:TOKEN",
            ),
            provider,
        )
        tools = await adapter.list_tools(context())
        assert [tool.tool_id for tool in tools] == ["mcp.maps.search"]
        result = await adapter.call_tool(tools[0], {"query": "park"}, context())
        assert result == {
            "structured": {"places": [{"name": "Park"}]},
            "text": "place found",
            "arguments": {"query": "park"},
        }
        await adapter.list_tools(context("bob"))
        assert all(
            headers["authorization"] == "Bearer fixture-secret"
            for _, _, headers in fixture.requests
        )
        if transport == "streamable_http":
            sessions = {
                headers["mcp-session-id"]
                for _, _, headers in fixture.requests
                if "mcp-session-id" in headers
            }
            assert len(sessions) == 3
        await adapter.aclose()
        with pytest.raises(ToolError, match="adapter_closed"):
            await adapter.list_tools(context())


@pytest.mark.parametrize(
    "url",
    [
        "http://public.example/mcp",
        "https://127.0.0.1/mcp",
        "https://169.254.169.254/mcp",
        "https://user:pass@example.com/mcp",
        "https://example.com/mcp?key=secret",
        "file:///tmp/mcp",
    ],
)
def test_config_rejects_unsafe_endpoints(url):
    with pytest.raises(ValidationError):
        McpServerConfig(id="maps", url=url)


def test_config_rejects_unimplemented_auth_and_header_override():
    for kwargs in [
        {"auth_type": "oauth"},
        {"auth_type": "header", "credential_ref": "env:TOKEN", "header_name": "Host"},
        {"auth_type": "bearer", "credential_ref": "raw-secret"},
    ]:
        with pytest.raises(ValidationError):
            McpServerConfig(id="maps", url="https://example.com/mcp", **kwargs)


async def test_scope_rejection_happens_before_network_access():
    provider = EnvironmentCredentialProvider(
        bindings={"maps": "env:TOKEN"},
        overrides={"TOKEN": "fixture-secret"},
        allowed_users={"maps": {"alice"}},
    )
    async with serve_protocol() as (fixture, base):
        adapter = McpAdapter(
            config(
                base + "/mcp",
                transport="streamable_http",
                auth_type="bearer",
                credential_ref="env:TOKEN",
            ),
            provider,
        )
        with pytest.raises(ToolError, match="credential_forbidden"):
            await adapter.list_tools(context("bob"))
        assert fixture.requests == []


@pytest.mark.parametrize(
    "status,code,retryable",
    [
        (401, "authentication_failed", False),
        (403, "authorization_denied", False),
        (429, "rate_limited", True),
        (503, "upstream_unavailable", True),
    ],
)
async def test_http_errors_are_structured_and_do_not_expose_body(
    status, code, retryable
):
    async with serve_protocol() as (fixture, base):
        fixture.status = status
        adapter = McpAdapter(config(base + "/mcp", transport="streamable_http"))
        with pytest.raises(ToolError) as error:
            await adapter.list_tools(context())
        assert error.value.code == code
        assert error.value.retryable is retryable
        assert "fixture-secret" not in str(error.value)


async def test_redirects_and_sse_cross_origin_endpoints_are_rejected():
    async with serve_protocol() as (fixture, base):
        fixture.redirect = "http://169.254.169.254/credentials"
        adapter = McpAdapter(config(base + "/mcp", transport="streamable_http"))
        with pytest.raises(ToolError, match="endpoint_forbidden"):
            await adapter.list_tools(context())
        fixture.sse_endpoint = "http://169.254.169.254/credentials"
        adapter = McpAdapter(config(base + "/sse", transport="sse"))
        with pytest.raises(ToolError):
            await adapter.list_tools(context())
        assert not any(path == "/credentials" for _, path, _ in fixture.requests)


async def test_argument_validation_and_allowlist_precede_tool_execution():
    async with serve_protocol() as (fixture, base):
        adapter = McpAdapter(config(base + "/mcp", transport="streamable_http"))
        tool = (await adapter.list_tools(context()))[0]
        requests_before = len(fixture.requests)
        for args in [
            {"query": 123},
            {"query": "park", "user_id": "bob"},
            {"query": "park", "authorization": "secret"},
        ]:
            with pytest.raises(ToolError, match="invalid_arguments"):
                await adapter.call_tool(tool, args, context())
        unapproved = tool.model_copy(
            update={"name": "unapproved", "tool_id": "mcp.maps.unapproved"}
        )
        with pytest.raises(ToolError, match="tool_forbidden"):
            await adapter.call_tool(unapproved, {"query": "park"}, context())
        assert len(fixture.requests) == requests_before


async def test_result_limits_redaction_and_remote_errors(caplog):
    provider = EnvironmentCredentialProvider(
        bindings={"maps": "env:TOKEN"}, overrides={"TOKEN": "fixture-secret"}
    )
    async with serve_protocol() as (fixture, base):
        adapter = McpAdapter(
            config(
                base + "/mcp",
                transport="streamable_http",
                auth_type="header",
                credential_ref="env:TOKEN",
                header_name="X-Api-Key",
                max_result_bytes=1024,
            ),
            provider,
        )
        tool = (await adapter.list_tools(context()))[0]
        fixture.result = {
            "content": [{"type": "text", "text": "echo fixture-secret"}],
            "structuredContent": {"token": "fixture-secret"},
        }
        with caplog.at_level("DEBUG"):
            result = await adapter.call_tool(tool, {"query": "park"}, context())
        assert "fixture-secret" not in json.dumps(result)
        assert "fixture-secret" not in caplog.text
        fixture.result = {"content": [{"type": "text", "text": "x" * 2000}]}
        with pytest.raises(ToolError, match="payload_too_large"):
            await adapter.call_tool(tool, {"query": "park"}, context())
        fixture.result = {
            "content": [{"type": "text", "text": "fixture-secret"}],
            "isError": True,
        }
        with pytest.raises(ToolError, match="remote_tool_error"):
            await adapter.call_tool(tool, {"query": "park"}, context())


async def test_timeout_and_cancellation_close_the_operation():
    async with serve_protocol() as (fixture, base):
        adapter = McpAdapter(config(base + "/mcp", transport="streamable_http"))
        tool = (await adapter.list_tools(context()))[0]
        fixture.delay = 1
        with pytest.raises(ToolError, match="timeout"):
            await adapter.call_tool(tool, {"query": "park"}, context(timeout=0.05))
        task = asyncio.create_task(
            adapter.call_tool(tool, {"query": "park"}, context())
        )
        await asyncio.sleep(0.03)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await adapter.aclose()


def test_dotted_capabilities_and_protocol_tool_names_are_accepted():
    value = McpServerConfig(
        id="amap_maps",
        url="https://example.com/mcp",
        allowed_tools=("maps.search", "maps-route"),
        capabilities=("maps.search", "maps.route", "maps.geocode"),
    )
    assert value.allowed_tools == ("maps.search", "maps-route")
    assert value.capabilities == ("maps.search", "maps.route", "maps.geocode")


async def test_private_dns_answers_are_rejected_before_connect(monkeypatch):
    async def private_resolution(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 443))]

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", private_resolution)
    adapter = McpAdapter(
        McpServerConfig(
            enabled=True,
            id="maps",
            url="https://metadata.example/mcp",
            transport="streamable_http",
        )
    )
    with pytest.raises(ToolError, match="endpoint_forbidden"):
        await adapter.list_tools(context())


async def test_catalog_bounds_reject_untrusted_schema_and_bound_description():
    async with serve_protocol() as (fixture, base):
        fixture.tools = [
            {"name": "search", "description": "long " * 300, "inputSchema": SCHEMA}
        ]
        adapter = McpAdapter(
            config(base + "/mcp", transport="streamable_http", max_description_chars=64)
        )
        assert len((await adapter.list_tools(context()))[0].description) == 64
        fixture.tools[0]["inputSchema"] = {
            "type": "object",
            "properties": {"q": {"$ref": "https://169.254.169.254/secret"}},
        }
        assert await adapter.list_tools(context()) == ()
        fixture.tools[0]["inputSchema"] = {
            "type": "object",
            "description": "x" * 20_000,
        }
        assert await adapter.list_tools(context()) == ()


async def test_oversized_wire_payload_is_rejected_before_sdk_json_decode():
    async with serve_protocol() as (fixture, base):
        fixture.tools[0]["description"] = "x" * 3000
        adapter = McpAdapter(
            config(base + "/mcp", transport="streamable_http", max_response_bytes=1024)
        )
        with pytest.raises(ToolError, match="payload_too_large"):
            await adapter.list_tools(context())


async def test_secret_redaction_does_not_modify_json_primitives():
    provider = EnvironmentCredentialProvider(
        bindings={"maps": "env:TOKEN"}, overrides={"TOKEN": "true"}
    )
    async with serve_protocol() as (fixture, base):
        fixture.result = {
            "content": [{"type": "text", "text": "true"}],
            "structuredContent": {"ok": True},
        }
        adapter = McpAdapter(
            config(
                base + "/mcp",
                transport="streamable_http",
                auth_type="bearer",
                credential_ref="env:TOKEN",
            ),
            provider,
        )
        tool = (await adapter.list_tools(context()))[0]
        assert await adapter.call_tool(tool, {"query": "park"}, context()) == {
            "structured": {"ok": True},
            "text": "[redacted]",
            "arguments": {"query": "park"},
        }


async def test_deadline_covers_credential_resolution_and_close_cancels_it():
    from pydantic import SecretStr

    entered = asyncio.Event()

    class SlowProvider:
        async def resolve(self, **kwargs):
            entered.set()
            await asyncio.sleep(1)
            return SecretStr("fixture-secret")

    async with serve_protocol() as (fixture, base):
        adapter = McpAdapter(
            config(
                base + "/mcp",
                transport="streamable_http",
                auth_type="bearer",
                credential_ref="env:TOKEN",
            ),
            SlowProvider(),
        )
        with pytest.raises(ToolError, match="timeout"):
            await asyncio.wait_for(adapter.list_tools(context(timeout=0.02)), 0.2)
        entered.clear()
        task = asyncio.create_task(adapter.list_tools(context()))
        await entered.wait()
        await adapter.aclose()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert fixture.requests == []
