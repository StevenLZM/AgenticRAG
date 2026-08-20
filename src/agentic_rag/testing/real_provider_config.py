"""Fail-closed provider configuration checks for opt-in live acceptance."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values


ProviderConfigurationIssue = tuple[Literal["missing", "invalid"], tuple[str, ...]]
_PROVIDER_ENVIRONMENT_NAMES = frozenset(
    {
        "AGENTIC_RAG_DEEPSEEK_BASE_URL",
        "AGENTIC_RAG_DEEPSEEK_API_KEY",
        "AGENTIC_RAG_QWEN_EMBEDDING_BASE_URL",
        "AGENTIC_RAG_QWEN_API_KEY",
        "AGENTIC_RAG_MEM0_ENABLED",
    }
)


def provider_configuration_issue(settings: object) -> ProviderConfigurationIssue | None:
    """Classify absent credentials separately from explicit bad configuration.

    An opt-in pytest case can skip when a developer has not supplied live
    credentials at all.  Once a value is supplied, placeholders, blank values
    and disabled Mem0 are configuration failures and must fail the acceptance
    run instead of silently becoming a skip.
    """
    missing: list[str] = []
    invalid: list[str] = []
    for field in (
        "deepseek_base_url",
        "deepseek_api_key",
        "qwen_embedding_base_url",
        "qwen_api_key",
    ):
        raw: object = getattr(settings, field, None)
        get_secret_value = getattr(raw, "get_secret_value", None)
        if callable(get_secret_value):
            raw = get_secret_value()
        if raw is None:
            missing.append(field)
        elif not isinstance(raw, str) or not raw.strip() or raw.strip().startswith("replace-with-"):
            invalid.append(field)
    if getattr(settings, "mem0_enabled", None) is not True:
        invalid.append("mem0_enabled")
    if invalid:
        return "invalid", tuple(invalid)
    if missing:
        return "missing", tuple(missing)
    return None


def explicit_provider_configuration_issue(
    environment: Mapping[str, object],
) -> ProviderConfigurationIssue | None:
    """Find explicitly supplied bad values before Settings can be constructed.

    Pydantic may report a different required field as missing first.  Inspecting
    raw environment input prevents that missing field from converting an
    explicit ``MEM0_ENABLED=false`` or ``replace-with-*`` credential into a
    misleading live-E2E skip.
    """
    invalid: list[str] = []
    for environment_name, field in (
        ("AGENTIC_RAG_DEEPSEEK_BASE_URL", "deepseek_base_url"),
        ("AGENTIC_RAG_DEEPSEEK_API_KEY", "deepseek_api_key"),
        ("AGENTIC_RAG_QWEN_EMBEDDING_BASE_URL", "qwen_embedding_base_url"),
        ("AGENTIC_RAG_QWEN_API_KEY", "qwen_api_key"),
    ):
        if environment_name not in environment:
            continue
        value = environment[environment_name]
        if not isinstance(value, str) or not value.strip() or value.strip().startswith(
            "replace-with-"
        ):
            invalid.append(field)
    enabled_name = "AGENTIC_RAG_MEM0_ENABLED"
    if enabled_name in environment:
        enabled = environment[enabled_name]
        normalized = enabled.strip().casefold() if isinstance(enabled, str) else ""
        if normalized not in {"1", "true", "yes", "on"}:
            invalid.append("mem0_enabled")
    if invalid:
        return "invalid", tuple(invalid)
    return None


def provider_environment_from_process(
    env_file: str | Path = ".env.local",
) -> dict[str, str | None]:
    """Read only provider toggles/credentials without logging their values."""
    values = {
        name: value
        for name, value in dotenv_values(env_file).items()
        if name in _PROVIDER_ENVIRONMENT_NAMES
    }
    values.update(
        {
            name: value
            for name, value in os.environ.items()
            if name in _PROVIDER_ENVIRONMENT_NAMES
        }
    )
    return values


__all__ = [
    "ProviderConfigurationIssue",
    "explicit_provider_configuration_issue",
    "provider_configuration_issue",
    "provider_environment_from_process",
]
