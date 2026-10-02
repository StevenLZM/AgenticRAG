import pytest

from agentic_rag.config import Settings
from agentic_rag.runtime.evaluation_identity import evaluation_config_fingerprint


def settings(**kwargs):
    return Settings(_env_file=None, mysql_dsn="mysql+asyncmy://test:test@localhost/test",
                    deepseek_base_url="https://model.invalid/v1", qwen_embedding_base_url="https://embedding.invalid/v1",
                    **kwargs)


def test_dashscope_key_alias_is_secret_and_not_part_of_snapshot_identity(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "fixture-not-real-secret")
    first = settings(amap_mcp_enabled=True)
    assert first.dashscope_api_key.get_secret_value() == "fixture-not-real-secret"
    assert "fixture-not-real-secret" not in first.model_dump_json()
    monkeypatch.setenv("DASHSCOPE_API_KEY", "fixture-rotated-secret")
    assert evaluation_config_fingerprint(first) == evaluation_config_fingerprint(settings(amap_mcp_enabled=True))
    assert evaluation_config_fingerprint(first) != evaluation_config_fingerprint(settings(amap_mcp_enabled=False))


def test_service_configuration_binds_transport_and_capabilities_to_snapshot():
    original = settings(amap_mcp_enabled=True)
    changed = settings(amap_mcp_enabled=True, amap_mcp_transport="streamable_http")
    assert evaluation_config_fingerprint(original) != evaluation_config_fingerprint(changed)


async def test_native_runtime_works_when_mcp_is_disabled(tmp_path):
    from agentic_rag.runtime.tool_composition import build_tool_runtime
    from agentic_rag.query.tool_loop import tool_context
    from tests.unit.query.test_router_fast_path import _initial_state
    runtime, capabilities = build_tool_runtime(settings(tool_invocation_path=tmp_path / "tools.sqlite"), retrieval=None)
    try:
        result = await runtime.call("local.calculator", {"expression": "6*7"}, tool_context(_initial_state()), "call-1")
        assert result.data["value"] == 42
        assert not capabilities
    finally:
        await runtime.aclose()


def test_missing_amap_credentials_do_not_disable_native_tools(tmp_path, monkeypatch):
    from agentic_rag.runtime.tool_composition import build_tool_runtime
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    runtime, capabilities = build_tool_runtime(settings(amap_mcp_enabled=True,
        tool_invocation_path=tmp_path / "tools.sqlite"), retrieval=None)
    assert runtime is not None
    assert capabilities == ()


def test_mcp_config_rejects_unsupported_auth_and_duplicate_ids():
    from pydantic import ValidationError
    service = {"id": "example", "url": "https://mcp.example.com/sse", "auth_type": "oauth"}
    with pytest.raises(ValidationError):
        settings(mcp_servers=[service])
    service["auth_type"] = "none"
    with pytest.raises(ValidationError):
        settings(mcp_servers=[service, service])


async def test_no_auth_service_composes_without_credential_binding(tmp_path):
    from agentic_rag.runtime.tool_composition import build_tool_runtime
    service = {"enabled": True, "id": "public", "url": "https://mcp.example.com/sse",
               "auth_type": "none", "allowed_tools": ["search"], "capabilities": ["public.search"]}
    runtime, capabilities = build_tool_runtime(settings(mcp_servers=[service],
        tool_invocation_path=tmp_path / "tools.sqlite"), retrieval=None)
    try:
        assert capabilities == ("public.search",)
        assert [adapter.adapter_id for adapter in runtime.adapters] == ["local", "mcp.public"]
    finally:
        await runtime.aclose()
