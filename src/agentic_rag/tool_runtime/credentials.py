"""Process-local credentials selected only by trusted service and user scope."""

import os
import re
from collections.abc import Collection, Mapping
from typing import Protocol

from pydantic import SecretStr

from agentic_rag.domain.models import UserScope

from .models import ToolError

_ENV_REFERENCE = re.compile(r"^env:([A-Za-z_][A-Za-z0-9_]*)$")


class CredentialProvider(Protocol):
    """Future vault or user providers must be explicitly injected at composition."""

    async def resolve(
        self, *, service_id: str, credential_ref: str, scope: UserScope
    ) -> SecretStr: ...


class EnvironmentCredentialProvider:
    """Read env references bound to a configured service, optionally restricting users.

    ``overrides`` accepts values already loaded by Settings (including env files).
    Keys are environment variable names, never inline credential references.
    Neither values nor the process environment are mutated or logged.
    """

    def __init__(
        self,
        *,
        bindings: Mapping[str, str],
        overrides: Mapping[str, str | SecretStr] | None = None,
        allowed_users: Mapping[str, Collection[str]] | None = None,
    ) -> None:
        if any(not _ENV_REFERENCE.fullmatch(ref) for ref in bindings.values()):
            raise ValueError("unsupported credential reference")
        self._bindings = dict(bindings)
        self._overrides = {
            key: value if isinstance(value, SecretStr) else SecretStr(value)
            for key, value in (overrides or {}).items()
        }
        self._allowed_users = {
            service: frozenset(users)
            for service, users in (allowed_users or {}).items()
        }

    async def resolve(
        self, *, service_id: str, credential_ref: str, scope: UserScope
    ) -> SecretStr:
        if self._bindings.get(service_id) != credential_ref:
            raise ToolError("credential_forbidden")
        users = self._allowed_users.get(service_id)
        if users is not None and scope.user_id not in users:
            raise ToolError("credential_forbidden")
        match = _ENV_REFERENCE.fullmatch(credential_ref)
        if match is None:
            raise ToolError("credential_forbidden")
        name = match.group(1)
        secret = self._overrides.get(name)
        value = (
            secret.get_secret_value()
            if secret is not None
            else os.environ.get(name, "")
        )
        if (
            not value
            or len(value) > 8192
            or not value.isascii()
            or any(ord(c) < 33 or ord(c) == 127 for c in value)
        ):
            raise ToolError("credential_unavailable")
        return SecretStr(value)
