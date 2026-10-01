"""Per-SDK-request usage, including unsuccessful attempts; never log prompts."""
from __future__ import annotations

import asyncio
import time
from uuid import uuid4

from agentic_rag.observability.logging import _EMISSION_SCOPE, stable_event_key


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def usage_fields(response, kind):
    usage = _get(response, "usage")
    cache = _get(usage, "prompt_cache_hit_tokens", _get(_get(usage, "prompt_tokens_details"), "cached_tokens",
        _get(_get(usage, "input_tokens_details"), "cached_tokens")))
    return {"input_tokens": _get(usage, "prompt_tokens", _get(usage, "input_tokens")),
            "output_tokens": 0 if kind == "embedding" else _get(usage, "completion_tokens", _get(usage, "output_tokens")),
            "cached_input_tokens": 0 if kind == "embedding" else cache,
            "actual_model": _get(response, "model")}


def meter_create(original, kind):
    async def metered(*args, **kwargs):
        scope = _EMISSION_SCOPE.get()
        if scope is None or scope.user_id is None:
            return await original(*args, **kwargs)
        identity = uuid4().hex
        attributes = {"provider_request_id": identity, "operation": kind, "requested_model": kwargs.get("model")}
        async def emit(event_type, details):
            try:
                await scope.emitter.emit(run_id=scope.run_id, user_id=scope.user_id, event_type=event_type,
                    node_name=scope.operation, summary="failed" if event_type.endswith("FAILED") else "completed" if event_type.endswith("COMPLETED") else "started",
                    event_key=stable_event_key(scope.run_id, identity, event_type), attributes={**attributes, **details})
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                # A missing begin/end pair causes incomplete evaluation usage.
                pass
        await emit("PROVIDER_REQUEST_STARTED", {})
        started = time.perf_counter()
        try:
            response = await original(*args, **kwargs)
        except (Exception, asyncio.CancelledError) as error:
            await emit("PROVIDER_REQUEST_FAILED", {"error_class": type(error).__name__})
            raise
        await emit("PROVIDER_REQUEST_COMPLETED", {**usage_fields(response, kind), "latency_ms": round((time.perf_counter() - started) * 1000)})
        return response
    return metered


def install_provider_meter(client):
    for path, kind in (("chat.completions", "llm"), ("responses", "llm"), ("embeddings", "embedding")):
        resource = client
        for name in path.split("."):
            resource = getattr(resource, name, None)
        create = getattr(resource, "create", None)
        if create is not None:
            resource.create = meter_create(create, kind)
