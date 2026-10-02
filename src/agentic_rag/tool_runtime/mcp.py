"""Read-only MCP SDK adapter with bounded, origin-pinned network operations.

Each discovery/call owns its client, SDK session and task scopes. This deliberately
trades connection reuse for isolation between users, credential rotations and MCP
protocol sessions. No SDK object or credential enters a serialized ToolResult.
"""

import asyncio
import hashlib
import ipaddress
import json
import logging
import re
import socket
import time
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from contextvars import ContextVar
from datetime import timedelta
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
from jsonschema import Draft202012Validator
from mcp import ClientSession, types
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .credentials import CredentialProvider, EnvironmentCredentialProvider
from .models import ToolContext, ToolDefinition, ToolError

_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_REF = re.compile(r"^env:[A-Za-z_][A-Za-z0-9_]*$")
_HEADER = re.compile(r"^[A-Za-z][A-Za-z0-9-]{0,63}$")
_RESERVED_HEADERS = {
    "host",
    "content-length",
    "content-type",
    "connection",
    "transfer-encoding",
    "accept",
    "accept-encoding",
    "cookie",
    "proxy-authorization",
    "mcp-session-id",
    "mcp-protocol-version",
}
_RESERVED_ARGUMENTS = {
    "user_id",
    "scope",
    "session_id",
    "run_id",
    "credential",
    "credential_ref",
    "authorization",
    "api_key",
    "access_token",
    "password",
    "secret",
    "headers",
    "budget",
    "deadline",
}
_IN_OPERATION: ContextVar[bool] = ContextVar("mcp_adapter_operation", default=False)


class _NoPayloadLogging(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # The SDK logs raw remote messages (including errors) at debug/warning.
        # Suppress only our operation context, never unrelated clients' logs.
        return not _IN_OPERATION.get()


for _logger in (
    "mcp.client.sse",
    "mcp.client.streamable_http",
    "mcp.client.session",
    "mcp.shared.session",
    "httpcore.http11",
    "httpcore.connection",
    "httpx",
):
    logging.getLogger(_logger).addFilter(_NoPayloadLogging())


def _literal_address(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        address = ipaddress.ip_address(host)
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            return address.ipv4_mapped
        return address
    except ValueError:
        return None


def _allowed_address(host: str, allow_loopback: bool) -> bool:
    address = _literal_address(host)
    return address is not None and (
        address.is_global or (allow_loopback and address.is_loopback)
    )


def _endpoint_parts(url: str, *, allow_loopback: bool, initial: bool = False):
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
        raise ValueError("invalid MCP endpoint") from None
    literal = _literal_address(host)
    local = bool(literal and literal.is_loopback and allow_loopback)
    if (
        not host
        or parts.username is not None
        or parts.password is not None
        or parts.fragment
        or (initial and parts.query)
        or "\\" in url
        or any(ord(c) < 33 or ord(c) == 127 for c in url)
        or parts.scheme not in (("https", "http") if local else ("https",))
        or (literal is not None and not _allowed_address(host, allow_loopback))
        or (allow_loopback and parts.scheme == "http" and not local)
    ):
        raise ValueError("unsafe MCP endpoint")
    return parts.scheme, host.lower(), port


class McpServerConfig(BaseModel):
    """Trusted configuration: allowlist entries are operator-approved read-only tools."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    enabled: bool = False
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    url: str = Field(max_length=2048)
    transport: Literal["sse", "streamable_http"] = "sse"
    auth_type: Literal["none", "bearer", "header"] = "none"
    credential_ref: str | None = None
    header_name: str | None = None
    allowed_tools: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    timeout_seconds: float = Field(default=20, gt=0, le=60)
    max_concurrency: int = Field(default=4, ge=1, le=16)
    max_response_bytes: int = Field(default=1_048_576, ge=1024, le=4_194_304)
    max_result_bytes: int = Field(default=131_072, ge=1024, le=1_048_576)
    max_schema_bytes: int = Field(default=16_384, ge=256, le=65_536)
    max_description_chars: int = Field(default=1024, ge=64, le=4096)
    max_tools: int = Field(default=128, ge=1, le=256)
    # Explicit trusted development opt-in; LAN/link-local targets remain denied.
    allow_loopback: bool = False

    @model_validator(mode="after")
    def valid_config(self):
        _endpoint_parts(self.url, allow_loopback=self.allow_loopback, initial=True)
        if len(self.allowed_tools) > self.max_tools or any(
            not _NAME.fullmatch(name) for name in self.allowed_tools
        ):
            raise ValueError("invalid approved tool names")
        if len(self.capabilities) > 32 or any(
            not _NAME.fullmatch(name) for name in self.capabilities
        ):
            raise ValueError("invalid capability names")
        if self.auth_type == "none":
            if self.credential_ref or self.header_name:
                raise ValueError("none authentication cannot carry credentials")
        elif not self.credential_ref or not _REF.fullmatch(self.credential_ref):
            raise ValueError(
                "authentication requires an environment credential reference"
            )
        if self.auth_type == "header":
            if (
                not self.header_name
                or not _HEADER.fullmatch(self.header_name)
                or self.header_name.lower() in _RESERVED_HEADERS
            ):
                raise ValueError("unsafe authentication header")
        elif self.header_name is not None:
            raise ValueError("header name requires header authentication")
        return self


class _BoundedStream(httpx.AsyncByteStream):
    def __init__(
        self,
        stream: httpx.AsyncByteStream,
        limit: int,
        fail: Callable[[ToolError], None],
    ) -> None:
        self._stream, self._limit, self._fail = stream, limit, fail

    async def __aiter__(self) -> AsyncIterator[bytes]:
        size = 0
        async for chunk in self._stream:
            size += len(chunk)
            if size > self._limit:
                error = ToolError("payload_too_large")
                self._fail(error)
                raise error
            yield chunk

    async def aclose(self) -> None:
        await self._stream.aclose()


class _GuardedTransport(httpx.AsyncBaseTransport):
    def __init__(self, config: McpServerConfig) -> None:
        self._config = config
        self._origin = _endpoint_parts(config.url, allow_loopback=config.allow_loopback)
        self._inner = httpx.AsyncHTTPTransport(retries=0)
        self.error: ToolError | None = None

    def _fail(self, error: ToolError) -> None:
        if self.error is None:
            self.error = error

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        try:
            origin = _endpoint_parts(
                str(request.url), allow_loopback=self._config.allow_loopback
            )
            if origin != self._origin:
                raise ToolError("endpoint_forbidden")
            host, port = origin[1:]
            addresses = await asyncio.get_running_loop().getaddrinfo(
                host, port, type=socket.SOCK_STREAM
            )
            ips = tuple(dict.fromkeys(item[4][0] for item in addresses))
            if not ips or not all(
                _allowed_address(ip, self._config.allow_loopback) for ip in ips
            ):
                raise ToolError("endpoint_forbidden")
            # Connect to the checked IP, retaining the original HTTP Host and TLS
            # SNI. A second DNS lookup cannot rebind the destination to a LAN IP.
            pinned = httpx.Request(
                request.method,
                request.url.copy_with(host=ips[0]),
                headers=request.headers,
                stream=request.stream,
                extensions={**request.extensions, "sni_hostname": host},
            )
            response = await self._inner.handle_async_request(pinned)
            status = response.status_code
            if 300 <= status < 400:
                await response.aclose()
                raise ToolError("endpoint_forbidden")
            if status in (401, 403, 429) or status >= 500:
                await response.aclose()
                raise ToolError(
                    {
                        401: "authentication_failed",
                        403: "authorization_denied",
                        429: "rate_limited",
                    }.get(status, "upstream_unavailable"),
                    retryable=status == 429 or status >= 500,
                )
            encoding = response.headers.get("content-encoding", "identity").lower()
            length = response.headers.get("content-length")
            if encoding != "identity" or (
                length and int(length) > self._config.max_response_bytes
            ):
                await response.aclose()
                raise ToolError("payload_too_large")
            assert isinstance(response.stream, httpx.AsyncByteStream)
            response.stream = _BoundedStream(
                response.stream, self._config.max_response_bytes, self._fail
            )
            return response
        except ToolError as error:
            self._fail(error)
            raise
        except ValueError:
            invalid_endpoint = ToolError("endpoint_forbidden")
            self._fail(invalid_endpoint)
            raise invalid_endpoint from None

    async def aclose(self) -> None:
        await self._inner.aclose()


def _bounded_json(value: Any, limit: int) -> str:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, RecursionError):
        raise ToolError("invalid_payload") from None
    if len(text.encode()) > limit:
        raise ToolError("payload_too_large")
    return text


def _schema_safe(value: Any, depth: int = 0) -> bool:
    if depth > 20:
        return False
    if isinstance(value, dict):
        return all(
            (
                key not in {"$ref", "$dynamicRef", "$recursiveRef"}
                or (isinstance(item, str) and item.startswith("#"))
            )
            and _schema_safe(item, depth + 1)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return all(_schema_safe(item, depth + 1) for item in value)
    return True


def _arguments_safe(value: Any, depth: int = 0) -> bool:
    if depth > 20:
        return False
    if isinstance(value, dict):
        return all(
            key.lower() not in _RESERVED_ARGUMENTS and _arguments_safe(item, depth + 1)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return all(_arguments_safe(item, depth + 1) for item in value)
    return True


def _redact(value: Any, secret: str) -> Any:
    if isinstance(value, str):
        return value.replace(secret, "[redacted]")
    if isinstance(value, dict):
        return {
            _redact(key, secret): _redact(item, secret) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    return value


def _safe_exception(error: BaseException) -> ToolError:
    if isinstance(error, ToolError):
        return error
    if isinstance(error, BaseExceptionGroup):
        for child in error.exceptions:
            classified = _safe_exception(child)
            if classified.code != "upstream_unavailable":
                return classified
    if isinstance(error, (TimeoutError, httpx.TimeoutException)):
        return ToolError("timeout", retryable=True)
    return ToolError("upstream_unavailable", retryable=True)


class McpAdapter:
    def __init__(
        self, config: McpServerConfig, credentials: CredentialProvider | None = None
    ) -> None:
        self.config = config
        self.adapter_id = f"mcp.{config.id}"
        self.credentials = credentials or EnvironmentCredentialProvider(
            bindings={config.id: config.credential_ref} if config.credential_ref else {}
        )
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
        self._closed = False
        self._tasks: set[asyncio.Task[Any]] = set()

    @asynccontextmanager
    async def _session(self, context: ToolContext):
        if self._closed:
            raise ToolError("adapter_closed")
        if not self.config.enabled:
            raise ToolError("service_disabled")
        remaining = min(self.config.timeout_seconds, context.deadline - time.time())
        if remaining <= 0:
            raise ToolError("timeout")
        transport: _GuardedTransport | None = None
        task = asyncio.current_task()
        if task:
            self._tasks.add(task)
        token = _IN_OPERATION.set(True)
        try:
            async with (
                asyncio.timeout(remaining),
                self._semaphore,
                AsyncExitStack() as stack,
            ):
                secret = ""
                headers = {"Accept-Encoding": "identity"}
                if self.config.credential_ref:
                    credential = await self.credentials.resolve(
                        service_id=self.config.id,
                        credential_ref=self.config.credential_ref,
                        scope=context.scope,
                    )
                    secret = credential.get_secret_value()
                    if self.config.auth_type == "bearer":
                        headers["Authorization"] = "Bearer " + secret
                    elif self.config.header_name:
                        headers[self.config.header_name] = secret
                transport = _GuardedTransport(self.config)
                client = httpx.AsyncClient(
                    transport=transport,
                    headers=headers,
                    timeout=remaining,
                    follow_redirects=False,
                    trust_env=False,
                )

                def client_factory(
                    headers: dict[str, str] | None = None,
                    timeout: httpx.Timeout | None = None,
                    auth: httpx.Auth | None = None,
                ) -> httpx.AsyncClient:
                    return client

                if self.config.transport == "sse":
                    # SSE enters/closes the client returned by its factory.
                    streams = await stack.enter_async_context(
                        sse_client(
                            self.config.url,
                            headers=headers,
                            timeout=remaining,
                            sse_read_timeout=remaining,
                            httpx_client_factory=client_factory,
                        )
                    )
                else:
                    await stack.enter_async_context(client)
                    streams = await stack.enter_async_context(
                        streamable_http_client(self.config.url, http_client=client)
                    )
                session = await stack.enter_async_context(
                    ClientSession(
                        streams[0],
                        streams[1],
                        read_timeout_seconds=timedelta(seconds=remaining),
                    )
                )
                await session.initialize()
                yield session, secret
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise (transport.error if transport else None) or _safe_exception(
                error
            ) from None
        finally:
            _IN_OPERATION.reset(token)
            if task:
                self._tasks.discard(task)

    def _definition(self, tool: types.Tool, secret: str) -> ToolDefinition | None:
        if tool.name not in self.config.allowed_tools or not _NAME.fullmatch(tool.name):
            return None
        annotations = tool.annotations
        if annotations and (
            annotations.readOnlyHint is False or annotations.destructiveHint is True
        ):
            return None
        schema = tool.inputSchema
        try:
            raw = _bounded_json(schema, self.config.max_schema_bytes)
            if (
                schema.get("type") != "object"
                or not _schema_safe(schema)
                or (secret and secret in raw)
            ):
                return None
            Draft202012Validator.check_schema(schema)
        except Exception:
            return None
        description = (tool.description or "")[: self.config.max_description_chars]
        if secret:
            description = description.replace(secret, "[redacted]")
        version = hashlib.sha256((raw + description).encode()).hexdigest()[:16]
        return ToolDefinition(
            tool_id=f"{self.adapter_id}.{tool.name}",
            adapter_id=self.adapter_id,
            name=tool.name,
            description=description,
            input_schema=schema,
            version=version,
            source_kind="external",
            capabilities=self.config.capabilities,
        )

    async def list_tools(self, context: ToolContext) -> tuple[ToolDefinition, ...]:
        if not self.config.enabled:
            return ()
        async with self._session(context) as (session, secret):
            definitions = []
            cursor = None
            seen = set()
            count = 0
            for _ in range(4):
                page = await session.list_tools(cursor=cursor)
                for tool in page.tools:
                    count += 1
                    if count > self.config.max_tools:
                        raise ToolError("catalog_too_large")
                    definition = self._definition(tool, secret)
                    if definition is not None and definition.tool_id not in seen:
                        definitions.append(definition)
                        seen.add(definition.tool_id)
                cursor = page.nextCursor
                if not cursor:
                    return tuple(definitions)
            raise ToolError("catalog_too_large")

    async def call_tool(
        self,
        definition: ToolDefinition,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> dict[str, Any]:
        if (
            definition.adapter_id != self.adapter_id
            or definition.name not in self.config.allowed_tools
            or definition.tool_id != f"{self.adapter_id}.{definition.name}"
        ):
            raise ToolError("tool_forbidden")
        try:
            _bounded_json(arguments, 16_384)
            if not _arguments_safe(arguments) or not _schema_safe(
                definition.input_schema
            ):
                raise ValueError
            Draft202012Validator(definition.input_schema).validate(arguments)
        except Exception:
            raise ToolError("invalid_arguments") from None
        async with self._session(context) as (session, secret):
            # Use the SDK's typed request directly. ClientSession.call_tool would
            # automatically fetch and validate untrusted remote output schemas.
            result = await session.send_request(
                types.ClientRequest(
                    types.CallToolRequest(
                        params=types.CallToolRequestParams(
                            name=definition.name, arguments=arguments
                        )
                    )
                ),
                types.CallToolResult,
            )
            if result.isError:
                raise ToolError("remote_tool_error")
            data = {
                "structured": result.structuredContent,
                "text": "\n".join(
                    block.text
                    for block in result.content
                    if isinstance(block, types.TextContent)
                ),
                "arguments": arguments,
            }
            raw = _bounded_json(data, self.config.max_result_bytes)
            bounded = json.loads(raw)
            return _redact(bounded, secret) if secret else bounded

    async def aclose(self) -> None:
        self._closed = True
        current = asyncio.current_task()
        tasks = [task for task in self._tasks if task is not current]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
