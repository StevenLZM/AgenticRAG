"""Service credentials cannot be selected by model-controlled tool arguments."""

import pytest
from pydantic import SecretStr

from agentic_rag.domain.models import UserScope
from agentic_rag.tool_runtime.credentials import EnvironmentCredentialProvider
from agentic_rag.tool_runtime.models import ToolError


async def test_environment_credentials_bind_service_reference_and_user(monkeypatch):
    monkeypatch.setenv("TEST_MCP_TOKEN", "fixture-secret")
    provider = EnvironmentCredentialProvider(
        bindings={"maps": "env:TEST_MCP_TOKEN"}, allowed_users={"maps": {"alice"}}
    )
    secret = await provider.resolve(
        service_id="maps",
        credential_ref="env:TEST_MCP_TOKEN",
        scope=UserScope(user_id="alice"),
    )
    assert secret.get_secret_value() == "fixture-secret"
    assert "fixture-secret" not in repr(secret)
    for service, reference, user in [
        ("other", "env:TEST_MCP_TOKEN", "alice"),
        ("maps", "env:OTHER_TOKEN", "alice"),
        ("maps", "env:TEST_MCP_TOKEN", "bob"),
    ]:
        with pytest.raises(ToolError, match="credential_forbidden"):
            await provider.resolve(
                service_id=service,
                credential_ref=reference,
                scope=UserScope(user_id=user),
            )


async def test_overrides_use_loaded_settings_without_mutating_environment(monkeypatch):
    monkeypatch.delenv("TEST_MCP_TOKEN", raising=False)
    provider = EnvironmentCredentialProvider(
        bindings={"maps": "env:TEST_MCP_TOKEN"},
        overrides={"TEST_MCP_TOKEN": SecretStr("from-settings")},
    )
    assert (
        await provider.resolve(
            service_id="maps",
            credential_ref="env:TEST_MCP_TOKEN",
            scope=UserScope(user_id="alice"),
        )
    ).get_secret_value() == "from-settings"
    assert "from-settings" not in repr(provider)


async def test_missing_or_malformed_credential_is_safe(monkeypatch):
    monkeypatch.delenv("TEST_MCP_MISSING", raising=False)
    provider = EnvironmentCredentialProvider(bindings={"maps": "env:TEST_MCP_MISSING"})
    with pytest.raises(ToolError, match="credential_unavailable"):
        await provider.resolve(
            service_id="maps",
            credential_ref="env:TEST_MCP_MISSING",
            scope=UserScope(user_id="alice"),
        )
    with pytest.raises(ValueError, match="credential reference"):
        EnvironmentCredentialProvider(bindings={"maps": "inline:secret"})
    provider = EnvironmentCredentialProvider(
        bindings={"maps": "env:BAD"}, overrides={"BAD": "unsafe\r\nheader"}
    )
    with pytest.raises(ToolError, match="credential_unavailable") as error:
        await provider.resolve(
            service_id="maps",
            credential_ref="env:BAD",
            scope=UserScope(user_id="alice"),
        )
    assert "unsafe" not in str(error.value)
