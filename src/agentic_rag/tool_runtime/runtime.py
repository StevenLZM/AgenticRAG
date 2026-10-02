"""One scoped permission, schema, budget and observation boundary for tools."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any, cast

from jsonschema import FormatChecker
from jsonschema.validators import validator_for
from pydantic import BaseModel, ConfigDict, Field

from agentic_rag.tool_runtime.models import (
    SourceKind, ToolAdapter, ToolContext, ToolDefinition, ToolError, ToolResult, json_payload,
)
from agentic_rag.tool_runtime.store import InvocationStore


class RuntimeLimits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    max_calls: int = Field(default=12, ge=1, le=1000)
    max_discoveries: int = Field(default=4, ge=1, le=100)
    call_timeout_seconds: float = Field(default=20, gt=0, le=300)
    max_result_bytes: int = Field(default=65_536, ge=128, le=1_048_576)
    max_document_result_bytes: int = Field(default=1_048_576, ge=128, le=4_194_304)
    max_arguments_bytes: int = Field(default=16_384, ge=128, le=65_536)
    max_schema_bytes: int = Field(default=16_384, ge=128, le=65_536)
    max_description_chars: int = Field(default=2048, ge=64, le=4096)
    max_catalog_tools: int = Field(default=100, ge=1, le=1000)


_SERVER_FIELDS = frozenset({
    "user_id", "session_id", "run_id", "scope", "snapshot", "deadline", "credentials",
    "credential", "credential_ref", "authorization", "api_key", "access_token", "budget",
    "max_calls", "max_discoveries",
})


def _has_server_fields(value: Any) -> bool:
    if isinstance(value, dict):
        return any(key.lower() in _SERVER_FIELDS or _has_server_fields(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_has_server_fields(item) for item in value)
    return False


class ToolRuntime:
    """Adapters are process-owned; callers supply only server-created context."""

    def __init__(
        self, adapters: Sequence[ToolAdapter], store: InvocationStore, limits: RuntimeLimits | None = None,
    ) -> None:
        self.adapters = tuple(adapters)
        self.store = store
        self.limits = limits or RuntimeLimits()
        self._by_id = {adapter.adapter_id: adapter for adapter in self.adapters}
        if len(self._by_id) != len(self.adapters):
            raise ValueError("adapter IDs must be unique")
        self._closed = False
        self._active: set[asyncio.Task[Any]] = set()

    def _open(self) -> None:
        if self._closed:
            raise ToolError("runtime_closed")

    async def _catalog(
        self, context: ToolContext, deadline: float,
    ) -> tuple[dict[str, ToolDefinition], dict[str, ToolError]]:
        async def listing(adapter: ToolAdapter) -> tuple[str, tuple[ToolDefinition, ...] | ToolError]:
            try:
                timeout = min(self.limits.call_timeout_seconds, deadline - time.time())
                if timeout <= 0:
                    raise ToolError("deadline_exceeded")
                async with asyncio.timeout(timeout):
                    definitions = await adapter.list_tools(context)
                if not isinstance(definitions, tuple) or len(definitions) > self.limits.max_catalog_tools:
                    raise ToolError("invalid_tool_catalog")
                if any(not isinstance(tool, ToolDefinition) or tool.adapter_id != adapter.adapter_id
                       or not tool.tool_id.startswith(adapter.adapter_id + ".") for tool in definitions):
                    raise ToolError("invalid_tool_catalog")
                if len({tool.tool_id for tool in definitions}) != len(definitions):
                    raise ToolError("invalid_tool_catalog")
                return adapter.adapter_id, definitions
            except asyncio.CancelledError:
                raise
            except ToolError as error:
                return adapter.adapter_id, error
            except TimeoutError:
                return adapter.adapter_id, ToolError("tool_timeout", retryable=True)
            except Exception:
                return adapter.adapter_id, ToolError("tool_unavailable", retryable=True)

        collected = await asyncio.gather(*(listing(adapter) for adapter in self.adapters))
        catalog: dict[str, ToolDefinition] = {}
        errors: dict[str, ToolError] = {}
        for adapter_id, result in collected:
            if isinstance(result, ToolError):
                errors[adapter_id] = result
            else:
                for definition in result:
                    if definition.tool_id in catalog:
                        raise ToolError("invalid_tool_catalog")
                    catalog[definition.tool_id] = definition.model_copy(deep=True)
        return catalog, errors

    def _validator(self, definition: ToolDefinition):
        schema = definition.input_schema
        try:
            payload = json_payload(schema)
            if len(payload.encode()) > self.limits.max_schema_bytes or schema.get("type") != "object":
                raise ValueError("invalid object schema")

            def check_refs(value: Any) -> None:
                if isinstance(value, dict):
                    for key, child in value.items():
                        if key in ("$ref", "$dynamicRef", "$recursiveRef") and (
                            not isinstance(child, str) or not child.startswith("#")
                        ):
                            raise ValueError("remote schema references disabled")
                        if key == "$id" and (not isinstance(child, str) or not child.startswith("#")):
                            raise ValueError("external schema identifiers disabled")
                        check_refs(child)
                elif isinstance(value, list):
                    for child in value:
                        check_refs(child)

            check_refs(schema)
            # jsonschema accepts None to reject unknown dialects; its stubs
            # only describe validator-class defaults. Do not silently fall back.
            validator_cls = validator_for(schema, default=cast(Any, None)) if "$schema" in schema else validator_for(schema)
            if validator_cls is None:
                raise ValueError("unsupported schema version")
            validator_cls.check_schema(schema)
            return validator_cls(schema, format_checker=FormatChecker())
        except Exception:
            raise ToolError("invalid_tool_schema") from None

    async def discover(self, query: str, context: ToolContext, limit: int = 5) -> tuple[ToolDefinition, ...]:
        """Search fresh visible metadata, charging the persisted discovery budget."""
        self._open()
        try:
            deadline = await self.store.consume_discovery(context, self.limits)
            catalog, errors = await self._catalog(context, deadline)
            terms = re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]+", str(query).lower()[:2048])
            terms.extend(word[i:i + 2] for word in tuple(terms) if re.search(r"[\u3400-\u9fff]", word)
                         for i in range(len(word) - 1))
            scored = []
            for definition in catalog.values():
                try:
                    self._validator(definition)
                except ToolError:
                    continue
                haystack = " ".join((definition.tool_id, definition.name, definition.description,
                                     *definition.capabilities)).lower()
                score = sum(1 for word in set(terms) if word in haystack)
                if score or not str(query).strip():
                    scored.append((score, definition.tool_id, definition))
            if not scored:
                # A healthy native catalog must not disguise an unavailable
                # requested service as "no matching capability". Relevance
                # uses trusted registry metadata, never a failed remote reply.
                for adapter_id, error in errors.items():
                    adapter = self._by_id[adapter_id]
                    capabilities = getattr(getattr(adapter, "config", None), "capabilities", ())
                    labels = [adapter_id]
                    if isinstance(capabilities, (tuple, list)):
                        labels.extend(value for value in capabilities if isinstance(value, str))
                    available_hints = " ".join(labels).lower()
                    if not str(query).strip() or any(
                        term in available_hints for term in terms if term not in {"mcp", "local"}
                    ):
                        raise error
            scored.sort(key=lambda item: (-item[0], item[1]))
            return tuple(item[2].model_copy(update={
                "description": item[2].description[:self.limits.max_description_chars],
                "capabilities": tuple(value[:128] for value in item[2].capabilities[:32]),
            }) for item in scored[:max(0, min(5, limit))])
        except asyncio.CancelledError:
            raise
        except ToolError:
            raise
        except Exception:
            raise ToolError("tool_unavailable", retryable=True) from None

    async def describe(self, tool_id: str, context: ToolContext) -> ToolDefinition | None:
        """Resolve a known tool directly, rechecking current adapter policy."""
        self._open()
        try:
            deadline = await self.store.check_context(context, self.limits)
            # Resolve only the owning namespace so unavailable remote catalogs
            # cannot delay a native call or a different remote service.
            candidates = [adapter for adapter in self.adapters if tool_id.startswith(adapter.adapter_id + ".")]
            if not candidates:
                return None
            owner = max(candidates, key=lambda adapter: len(adapter.adapter_id))
            timeout = min(self.limits.call_timeout_seconds, deadline - time.time())
            if timeout <= 0:
                raise ToolError("deadline_exceeded")
            async with asyncio.timeout(timeout):
                definitions = await owner.list_tools(context)
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                raise asyncio.CancelledError()
            if not isinstance(definitions, tuple) or len(definitions) > self.limits.max_catalog_tools:
                raise ToolError("invalid_tool_catalog")
            matches = [definition for definition in definitions if isinstance(definition, ToolDefinition)
                       and definition.tool_id == tool_id and definition.adapter_id == owner.adapter_id]
            if len(matches) > 1:
                raise ToolError("invalid_tool_catalog")
            if not matches:
                return None
            definition = matches[0].model_copy(deep=True)
            self._validator(definition)
            return definition
        except asyncio.CancelledError:
            raise
        except ToolError:
            raise
        except TimeoutError:
            raise ToolError("tool_timeout", retryable=True) from None
        except Exception:
            raise ToolError("tool_unavailable", retryable=True) from None

    @staticmethod
    def _error(call_id: str, tool_id: str, source_kind: SourceKind, error: ToolError) -> ToolResult:
        return ToolResult(call_id=call_id, tool_id=tool_id, source_kind=source_kind, status="error",
                          observed_at=datetime.now(timezone.utc).isoformat(), error_code=error.code,
                          retryable=error.retryable)

    async def call(
        self, tool_id: str, arguments: dict[str, Any], context: ToolContext, call_id: str,
    ) -> ToolResult:
        """Validate, reserve exactly one local attempt, execute and fence its result."""
        source_kind: SourceKind = "external"
        token: str | None = None
        attempt_deadline = time.time() + self.limits.call_timeout_seconds
        task = asyncio.current_task()
        if task is not None:
            self._active.add(task)
        try:
            self._open()
            if not isinstance(call_id, str) or not call_id or len(call_id) > 256:
                raise ToolError("invalid_call_id")
            definition = await self.describe(tool_id, context)
            if definition is None:
                raise ToolError("tool_not_found")
            source_kind = definition.source_kind
            try:
                payload = json_payload(arguments)
                if type(arguments) is not dict or len(payload.encode()) > self.limits.max_arguments_bytes or _has_server_fields(arguments):
                    raise ValueError("invalid arguments")
                self._validator(definition).validate(arguments)
            except ToolError:
                raise
            except Exception:
                raise ToolError("invalid_arguments") from None
            claim = await self.store.reserve(context, definition, hashlib.sha256(payload.encode()).hexdigest(),
                                             call_id, self.limits, attempt_deadline=attempt_deadline)
            if claim.result is not None:
                return claim.result
            token = claim.token
            assert token is not None
            timeout = claim.deadline - time.time()
            if timeout <= 0:
                raise ToolError("deadline_exceeded")
            try:
                async with asyncio.timeout(timeout):
                    data = await self._by_id[definition.adapter_id].call_tool(definition, json.loads(payload), context)
            except TimeoutError:
                raise ToolError("tool_timeout", retryable=True) from None
            if task is not None and task.cancelling():
                raise asyncio.CancelledError()
            if time.time() >= claim.deadline:
                raise ToolError("tool_timeout", retryable=True)
            try:
                encoded = json_payload(data)
                if type(data) is not dict:
                    raise ValueError("data must be an object")
            except Exception:
                raise ToolError("invalid_tool_result") from None
            result_limit = (self.limits.max_document_result_bytes if source_kind == "document"
                            else self.limits.max_result_bytes)
            if len(encoded.encode()) > result_limit:
                raise ToolError("result_too_large")
            result = ToolResult(call_id=call_id, tool_id=tool_id, source_kind=source_kind, status="success",
                                data=json.loads(encoded), observed_at=datetime.now(timezone.utc).isoformat())
            if not await self.store.finish(context, call_id, token, result):
                raise ToolError("invocation_superseded")
            return result
        except asyncio.CancelledError:
            # A task can be interrupted by worker shutdown, lease loss or a
            # child timeout. Only an explicit durable cancellation may poison
            # the whole Run; this path revokes the attempt we actually own.
            if token is not None:
                await asyncio.shield(self.store.interrupt(context, call_id, token))
            raise
        except Exception as error:
            safe = error if isinstance(error, ToolError) else ToolError("tool_unavailable", retryable=True)
            result = self._error(call_id, tool_id, source_kind, safe)
            if token is not None:
                try:
                    await self.store.finish(context, call_id, token, result)
                except Exception:
                    pass
            return result
        finally:
            if task is not None:
                self._active.discard(task)

    async def aclose(self) -> None:
        """Cancel local executions, then close every owned resource independently."""
        if self._closed:
            return
        self._closed = True
        tasks = [task for task in self._active if task is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for adapter in self.adapters:
            close = getattr(adapter, "aclose", None)
            if callable(close):
                try:
                    result = close()
                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    pass
        await self.store.aclose()
