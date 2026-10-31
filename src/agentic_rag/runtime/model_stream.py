"""Native provider deltas with explicit completion and no prefix retries."""
import asyncio
from collections.abc import AsyncIterator
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict


class ModelStreamEvent(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["delta", "completed"]
    text: str = ""
    actual_model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


async def stream_model(gateway: Any, call: Any) -> AsyncIterator[ModelStreamEvent]:
    from agentic_rag.runtime.model_gateway import ModelProtocolError, _get, _usage_value
    from agentic_rag.runtime.circuit import CircuitOpenError
    if not gateway._circuit.allow_call():
        raise CircuitOpenError("model circuit is open")
    provider = None
    try:
        async with asyncio.timeout(call.effective_timeout_seconds):
            chat = _get(_get(gateway._client, "chat"), "completions")
            responses = _get(gateway._client, "responses")
            protocol = call.effective_protocol
            kwargs = {"model": call.requested_model, "stream": True}
            if call.temperature is not None:
                kwargs["temperature"] = call.temperature
            if protocol in ("auto", "chat") and callable(_get(chat, "create")):
                protocol = "chat"
                kwargs.update(messages=list(call.messages), stream_options={"include_usage": True})
                kwargs.update(call.thinking_options("chat"))
                if call.max_output_tokens is not None:
                    kwargs["max_tokens"] = call.max_output_tokens
                provider = await cast(Any, chat).create(**kwargs)
            elif protocol in ("auto", "responses") and callable(_get(responses, "create")):
                protocol = "responses"
                kwargs["input"] = list(call.messages)
                kwargs.update(call.thinking_options("responses"))
                if call.max_output_tokens is not None:
                    kwargs["max_output_tokens"] = call.max_output_tokens
                provider = await cast(Any, responses).create(**kwargs)
            else:
                raise ModelProtocolError("provider does not support selected streaming protocol")
            completed = False
            usage: Any = {}
            model = call.requested_model
            async for event in provider:
                text = ""
                if protocol == "chat":
                    model = _get(event, "model") or model
                    if _get(event, "usage") is not None:
                        usage = event
                    choices = cast(list[Any], _get(event, "choices") or [])
                    for choice in choices:
                        delta = _get(choice, "delta")
                        value = _get(delta, "content")
                        if isinstance(value, str):
                            text += value
                        finish = _get(choice, "finish_reason")
                        if finish is not None:
                            if finish != "stop":
                                raise RuntimeError(f"provider terminal reason: {finish}")
                            completed = True
                else:
                    kind = _get(event, "type")
                    if kind == "response.output_text.delta":
                        text = str(_get(event, "delta") or "")
                    elif kind == "response.completed":
                        response = _get(event, "response")
                        if _get(response, "status") not in (None, "completed"):
                            raise RuntimeError("provider terminal response is incomplete")
                        completed, usage = True, response
                        model = _get(response, "model") or model
                    elif kind in ("response.failed", "response.incomplete", "error"):
                        raise RuntimeError("provider terminal response failed")
                if text:
                    yield ModelStreamEvent(kind="delta", text=text, actual_model=model)
            if not completed:
                raise RuntimeError("provider EOF without terminal completion")
            gateway._circuit.record_success()
            yield ModelStreamEvent(kind="completed", actual_model=model,
                                   input_tokens=_usage_value(usage, "input_tokens", "prompt_tokens"),
                                   output_tokens=_usage_value(usage, "output_tokens", "completion_tokens"))
    except Exception:
        gateway._circuit.record_failure()
        raise
    finally:
        if provider is not None:
            close = getattr(provider, "close", None) or getattr(provider, "aclose", None)
            if close is not None:
                await close()
