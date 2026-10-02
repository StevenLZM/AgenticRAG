"""Compose shared native/MCP tools without connecting to remote services at boot."""
from __future__ import annotations

import os
from typing import Any

from agentic_rag.config import Settings
from agentic_rag.tool_runtime.credentials import EnvironmentCredentialProvider
from agentic_rag.tool_runtime.mcp import McpAdapter, McpServerConfig
from agentic_rag.tool_runtime.native import NativeAdapter
from agentic_rag.tool_runtime.runtime import RuntimeLimits, ToolRuntime
from agentic_rag.tool_runtime.store import InvocationStore


AMAP_READ_TOOLS = (
    "maps_geo", "maps_regeo", "maps_text_search", "maps_around_search", "maps_search_detail",
    "maps_direction_driving", "maps_direction_walking", "maps_bicycling",
    "maps_direction_bicycling", "maps_direction_transit_integrated", "maps_distance",
)


def configured_servers(settings: Settings) -> tuple[McpServerConfig, ...]:
    servers = list(settings.mcp_servers)
    if settings.amap_mcp_enabled:
        if any(server.id == "amap_maps" for server in servers):
            raise ValueError("amap_maps is configured twice")
        servers.append(McpServerConfig(enabled=True, id="amap_maps", url=settings.amap_mcp_url,
            transport=settings.amap_mcp_transport, auth_type="bearer", credential_ref="env:DASHSCOPE_API_KEY",
            allowed_tools=AMAP_READ_TOOLS, capabilities=("maps.search", "maps.geocode", "maps.route", "maps.distance")))
    return tuple(servers)


def build_tool_runtime(settings: Settings, *, retrieval: Any) -> tuple[ToolRuntime | None, tuple[str, ...]]:
    if not settings.tool_runtime_enabled:
        return None, ()
    adapters: list[Any] = [NativeAdapter(retrieval)]
    capabilities: set[str] = set()
    overrides = {"DASHSCOPE_API_KEY": settings.dashscope_api_key} if settings.dashscope_api_key else {}
    for server in configured_servers(settings):
        if not server.enabled:
            continue
        reference = server.credential_ref or ""
        variable = reference.removeprefix("env:")
        # Missing optional credentials disable just that service. Authentication
        # validity is checked at discovery/call time, not claimed by this list.
        if server.auth_type != "none" and not (overrides.get(variable) or os.environ.get(variable)):
            continue
        credentials = (None if server.auth_type == "none" else
                       EnvironmentCredentialProvider(bindings={server.id: reference}, overrides=overrides))
        adapters.append(McpAdapter(server, credentials))
        if server.allowed_tools:
            capabilities.update(server.capabilities)
    runtime = ToolRuntime(adapters, InvocationStore(settings.tool_invocation_path), RuntimeLimits(
        max_calls=settings.tool_max_calls_per_run, max_discoveries=settings.tool_max_discoveries_per_run,
        call_timeout_seconds=settings.tool_call_timeout_seconds))
    return runtime, tuple(sorted(capabilities))
