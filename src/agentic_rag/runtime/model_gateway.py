"""A single retry-owning boundary for OpenAI-compatible model calls.

The gateway deliberately accepts an already-created provider client.  This keeps
credentials and network configuration outside graph state and permits fully
offline tests with a deterministic fake client.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar, cast

from pydantic import BaseModel, ConfigDict, ValidationError

from agentic_rag.observability.logging import emit_degradation, emit_model_usage
from agentic_rag.runtime.circuit import CircuitOpenError, CircuitState
from agentic_rag.runtime.models import RuntimeConfigSnapshot


T = TypeVar("T")
Sleep = Callable[[float], Awaitable[None]]


class ModelCall(BaseModel):
    """An immutable request whose model is selected from the run snapshot."""

    model_config = ConfigDict(frozen=True)

    messages: tuple[dict[str, Any], ...]
    model_role: Literal["main", "light"] = "main"
    protocol: Literal["auto", "chat", "responses"] | None = None
    snapshot: RuntimeConfigSnapshot
    timeout_seconds: float | None = None
    temperature: float | None = None

    @property
    def requested_model(self) -> str:
        return (
            self.snapshot.main_model_id
            if self.model_role == "main"
            else self.snapshot.light_model_id
        )

    @property
    def effective_timeout_seconds(self) -> float:
        return self.timeout_seconds or float(self.snapshot.query_run_timeout_seconds)

    @property
    def effective_protocol(self) -> Literal["auto", "chat", "responses"]:
        return self.protocol or self.snapshot.deepseek_protocol


class ModelResponse(BaseModel, Generic[T]):
    """Provider output plus the metadata required for audit and billing."""

    model_config = ConfigDict(frozen=True)

    value: T
    requested_model: str
    actual_model: str
    input_tokens: int
    output_tokens: int
    attempts: int
    latency_ms: int


class StructuredOutputValidationError(ValueError):
    """The model failed its one permitted schema-repair attempt."""


class ModelProtocolError(TypeError):
    """The injected provider client cannot satisfy the selected protocol."""


class ModelGateway:
    """Invoke an injected OpenAI-compatible client with bounded retries.

    Nodes must call this class once per logical model operation.  Retrying it in
    a graph node would violate the single retry-owner invariant.
    """

    def __init__(
        self,
        client: object,
        *,
        max_retries: int = 2,
        backoff_base_seconds: float = 0.25,
        sleep: Sleep = asyncio.sleep,
        random_source: Callable[[], float] = random.random,
        circuit: CircuitState | None = None,
    ) -> None:
        if max_retries < 0 or max_retries > 2:
            raise ValueError("max_retries must be between 0 and 2")
        self._client = client
        self._max_retries = max_retries
        self._backoff_base_seconds = backoff_base_seconds
        self._sleep = sleep
        self._random = random_source
        self._circuit = circuit or CircuitState()

    async def complete(self, call: ModelCall) -> ModelResponse[str]:
        """Return plain text, retrying only transient provider failures."""
        started = time.perf_counter()
        response, attempts = await self._request_with_retries(call, structured=False)
        result = ModelResponse[str](
            value=_extract_text(response),
            requested_model=call.requested_model,
            actual_model=_as_string(_get(response, "model")) or call.requested_model,
            input_tokens=_usage_value(response, "input_tokens", "prompt_tokens"),
            output_tokens=_usage_value(response, "output_tokens", "completion_tokens"),
            attempts=attempts,
            latency_ms=round((time.perf_counter() - started) * 1_000),
        )
        await emit_model_usage(
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            attempts=result.attempts,
            latency_ms=result.latency_ms,
        )
        return result

    async def complete_structured(
        self, call: ModelCall, schema: type[T]
    ) -> ModelResponse[T]:
        """Parse an entire structured result or fail closed after one repair."""
        started = time.perf_counter()
        response, attempts = await self._request_with_retries(call, structured=True)
        input_tokens = _usage_value(response, "input_tokens", "prompt_tokens")
        output_tokens = _usage_value(response, "output_tokens", "completion_tokens")
        text = _extract_text(response)
        try:
            value = _validate_schema(schema, text)
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as error:
            repair_call = call.model_copy(
                update={"messages": _repair_messages(call.messages, text, str(error))}
            )
            repair_response, repair_attempts = await self._request_with_retries(
                repair_call, structured=True
            )
            attempts += repair_attempts
            response = repair_response
            input_tokens += _usage_value(response, "input_tokens", "prompt_tokens")
            output_tokens += _usage_value(response, "output_tokens", "completion_tokens")
            try:
                value = _validate_schema(schema, _extract_text(response))
            except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as repair_error:
                await emit_degradation(
                    component="llm",
                    reason="model_schema_invalid",
                    run_id=None,
                    snapshot_id=call.snapshot.snapshot_id,
                    attempt=attempts,
                    retryable=False,
                    outcome="refused",
                    event_type="MODEL_REPAIR_EXHAUSTED",
                    attributes={
                        "schema_name": _safe_schema_name(schema),
                        "protocol": call.effective_protocol,
                        "requested_model": call.requested_model,
                        "actual_model": _as_string(_get(response, "model"))
                        or call.requested_model,
                        "error_class": type(repair_error).__name__,
                        "output_length": len(_extract_text(response)),
                        "output_sha256": hashlib.sha256(
                            _extract_text(response).encode("utf-8")
                        ).hexdigest(),
                    },
                )
                raise StructuredOutputValidationError(
                    "model output did not satisfy the requested schema after repair"
                ) from repair_error
        result = ModelResponse[T](
            value=value,
            requested_model=call.requested_model,
            actual_model=_as_string(_get(response, "model")) or call.requested_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            attempts=attempts,
            latency_ms=round((time.perf_counter() - started) * 1_000),
        )
        await emit_model_usage(
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            attempts=result.attempts,
            latency_ms=result.latency_ms,
        )
        return result

    async def _request_with_retries(
        self, call: ModelCall, *, structured: bool
    ) -> tuple[object, int]:
        for attempt in range(1, self._max_retries + 2):
            if not self._circuit.allow_call():
                await emit_degradation(
                    component="llm",
                    reason="circuit_open",
                    run_id=None,
                    snapshot_id=call.snapshot.snapshot_id,
                    attempt=attempt,
                    retryable=True,
                    outcome="degraded",
                    event_type="CIRCUIT_OPEN",
                )
                raise CircuitOpenError("model provider circuit is open")
            try:
                response = await asyncio.wait_for(
                    self._create(call, structured=structured),
                    timeout=call.effective_timeout_seconds,
                )
                self._circuit.record_success()
                return response, attempt
            except CircuitOpenError:
                raise
            except BaseException as error:
                if isinstance(error, ModelProtocolError):
                    await emit_degradation(
                        component="llm",
                        reason="protocol_error",
                        run_id=None,
                        snapshot_id=call.snapshot.snapshot_id,
                        attempt=attempt,
                        retryable=False,
                        outcome="degraded",
                        event_type="MODEL_PROTOCOL_ERROR",
                    )
                    raise
                if not _is_transient(error) or attempt > self._max_retries:
                    if _is_transient(error) and attempt > self._max_retries:
                        self._circuit.record_failure()
                        await emit_degradation(
                            component="llm",
                            reason="model_unavailable",
                            run_id=None,
                            snapshot_id=call.snapshot.snapshot_id,
                            attempt=attempt,
                            retryable=True,
                            outcome="degraded",
                            event_type="MODEL_RETRY_EXHAUSTED",
                        )
                    raise
                # Jitter prevents synchronized reattempts across independent runs.
                opened = self._circuit.record_failure()
                await emit_degradation(
                    component="llm",
                    reason="provider_outage",
                    run_id=None,
                    snapshot_id=call.snapshot.snapshot_id,
                    attempt=attempt,
                    retryable=True,
                    outcome="degraded",
                    event_type="MODEL_RETRY",
                )
                if opened:
                    await emit_degradation(
                        component="llm",
                        reason="circuit_open",
                        run_id=None,
                        snapshot_id=call.snapshot.snapshot_id,
                        attempt=attempt,
                        retryable=True,
                        outcome="degraded",
                        event_type="CIRCUIT_OPEN",
                    )
                    raise CircuitOpenError("model provider circuit opened") from error
                delay = self._backoff_base_seconds * (2 ** (attempt - 1))
                await self._sleep(delay * (0.5 + self._random()))
        raise AssertionError("retry loop must either return or raise")

    async def _create(self, call: ModelCall, *, structured: bool = False) -> object:
        responses = _get(self._client, "responses")
        chat = _get(self._client, "chat")
        completions = _get(chat, "completions") if chat is not None else None
        chat_available = completions is not None and callable(_get(completions, "create"))
        responses_available = responses is not None and callable(_get(responses, "create"))
        protocol = call.effective_protocol

        if protocol in {"auto", "chat"} and chat_available:
            kwargs: dict[str, object] = {
                "model": call.requested_model,
                "messages": list(call.messages),
            }
            if structured:
                kwargs["response_format"] = {"type": "json_object"}
            if call.temperature is not None:
                kwargs["temperature"] = call.temperature
            return await cast(Any, _get(completions, "create"))(**kwargs)

        if protocol in {"auto", "responses"} and responses_available:
            kwargs = {
                "model": call.requested_model,
                "input": list(call.messages),
            }
            if structured:
                kwargs["text"] = {"format": {"type": "json_object"}}
            if call.temperature is not None:
                kwargs["temperature"] = call.temperature
            return await cast(Any, _get(responses, "create"))(**kwargs)

        raise ModelProtocolError(
            f"client cannot satisfy requested {protocol} model protocol"
        )


class PromptTemplate(BaseModel):
    """A repository-owned immutable prompt artifact."""

    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    content: str
    content_hash: str


_PROMPT_DIRECTORY = Path(__file__).parent.parent / "prompts"


def load_prompt(name: str) -> PromptTemplate:
    """Load one checked-in versioned prompt without allowing path traversal."""
    filename = name if name.endswith(".md") else f"{name}.md"
    if Path(filename).name != filename or not filename.endswith("_v1.md"):
        raise ValueError("prompt names must name a checked-in *_v1.md file")
    path = _PROMPT_DIRECTORY / filename
    content = path.read_text(encoding="utf-8")
    version = Path(filename).stem.rsplit("_", 1)[-1]
    return PromptTemplate(
        name=Path(filename).stem,
        version=version,
        content=content,
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )


def prompt_hashes(names: Sequence[str]) -> dict[str, str]:
    """Return the content hashes to capture when creating a run snapshot."""
    return {prompt.name: prompt.content_hash for prompt in map(load_prompt, names)}


def _validate_schema(schema: type[T], text: str) -> T:
    parsed = json.loads(_normalize_structured_text(text))
    validator = _get(schema, "model_validate")
    if callable(validator):
        return cast(T, validator(parsed))
    if isinstance(parsed, schema):
        return cast(T, parsed)
    return cast(T, schema(**parsed))


def _normalize_structured_text(text: str) -> str:
    """Remove only a complete Markdown JSON fence; keep schema validation strict."""
    candidate = text.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        first_newline = candidate.find("\n")
        if first_newline > 0:
            language = candidate[3:first_newline].strip().lower()
            if language in {"", "json"}:
                return candidate[first_newline + 1 : -3].strip()
    return candidate


def _safe_schema_name(schema: object) -> str:
    name = getattr(schema, "__name__", "schema")
    return name if isinstance(name, str) and name else "schema"


def _repair_messages(
    original: Sequence[dict[str, Any]], output: str, error: str
) -> tuple[dict[str, Any], ...]:
    return (*original, {
        "role": "user",
        "content": (
            "Schema validation error. Return a complete corrected JSON object only; "
            "do not explain it. Invalid output was: "
            f"{output}\nValidation error: {error}"
        ),
    })


def _is_transient(error: BaseException) -> bool:
    """Classify only known temporary provider/transport failures.

    SDK transport exceptions commonly wrap ``httpx`` errors instead of inheriting
    the standard-library ``ConnectionError``.  Traverse an exception chain by
    identity while deliberately preserving cancellation and interrupts.
    """
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    transient_names = {"APIConnectionError", "ConnectError", "ConnectTimeout", "ReadTimeout"}
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
            return False
        if isinstance(current, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
            return True
        status_code = _get(current, "status_code")
        if isinstance(status_code, int) and (status_code == 429 or 500 <= status_code <= 599):
            return True
        if type(current).__name__ in transient_names:
            return True
        cause = current.__cause__
        context = current.__context__
        if cause is not None:
            pending.append(cause)
        if context is not None:
            pending.append(context)
    return False


def _get(value: object | None, key: str) -> object | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _as_string(value: object | None) -> str | None:
    return value if isinstance(value, str) else None


def _extract_text(response: object) -> str:
    output_text = _as_string(_get(response, "output_text"))
    if output_text is not None:
        return output_text
    choices = _get(response, "choices")
    if isinstance(choices, Sequence) and not isinstance(choices, (str, bytes)) and choices:
        message = _get(choices[0], "message")
        content = _get(message, "content")
        if isinstance(content, str):
            return content
        if isinstance(content, Sequence):
            return "".join(
                text
                for item in content
                if (text := _as_string(_get(item, "text"))) is not None
            )
    output = _get(response, "output")
    if isinstance(output, Sequence) and not isinstance(output, (str, bytes)):
        parts: list[str] = []
        for item in output:
            content = _get(item, "content")
            if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
                parts.extend(
                    text
                    for block in content
                    if (text := _as_string(_get(block, "text"))) is not None
                )
        if parts:
            return "".join(parts)
    raise ValueError("provider response contained no text output")


def _usage_value(response: object, *keys: str) -> int:
    usage = _get(response, "usage")
    for key in keys:
        value = _get(usage, key)
        if isinstance(value, int):
            return value
    return 0
