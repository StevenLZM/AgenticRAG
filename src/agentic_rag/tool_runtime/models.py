"""Serializable public tool contracts; process resources remain outside context."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentic_rag.domain.models import UserScope
from agentic_rag.runtime.models import RuntimeConfigSnapshot


SourceKind = Literal["document", "external", "calculation"]


def json_payload(value: object) -> str:
    """Encode genuine JSON only, without coercing objects, keys, or NaN."""
    def check(item: object, depth: int = 0) -> None:
        if depth > 64:
            raise ValueError("JSON nesting limit exceeded")
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item:
                check(child, depth + 1)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                check(child, depth + 1)
            return
        raise ValueError("value is not JSON")

    check(value)
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


class ToolDefinition(BaseModel):
    """An adapter's approved, read-only tool definition."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    tool_id: str = Field(min_length=1, max_length=256)
    adapter_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=256)
    description: str = ""
    input_schema: dict[str, Any]
    version: str = Field(default="1", min_length=1, max_length=128)
    source_kind: SourceKind
    capabilities: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Server-owned identity, immutable configuration and absolute run deadline."""

    scope: UserScope
    run_id: str
    session_id: str
    snapshot: RuntimeConfigSnapshot
    deadline: float

    def __post_init__(self) -> None:
        if not self.run_id or not self.session_id or not math.isfinite(self.deadline):
            raise ValueError("invalid tool context")


class ToolResult(BaseModel):
    """Bounded JSON observation safe to put in a checkpoint."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    call_id: str
    tool_id: str
    source_kind: SourceKind
    status: Literal["success", "error"]
    data: dict[str, Any] = Field(default_factory=dict)
    observed_at: str
    error_code: str | None = None
    retryable: bool = False

    @field_validator("data", mode="before")
    @classmethod
    def _json_only(cls, value: Any) -> Any:
        if type(value) is not dict:
            raise ValueError("tool data must be an object")
        json_payload(value)
        return value


class ToolError(Exception):
    """Only a safe machine code crosses adapter and execution boundaries."""

    def __init__(self, code: str, retryable: bool = False) -> None:
        self.code = code if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code) else "tool_error"
        self.retryable = bool(retryable)
        super().__init__(self.code)


class ToolAdapter(Protocol):
    @property
    def adapter_id(self) -> str: ...

    async def list_tools(self, context: ToolContext) -> tuple[ToolDefinition, ...]: ...

    async def call_tool(
        self, definition: ToolDefinition, arguments: dict[str, Any], context: ToolContext,
    ) -> dict[str, Any]: ...
