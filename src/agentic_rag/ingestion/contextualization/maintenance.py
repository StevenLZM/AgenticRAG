"""External deployment authority attestations, never self-issued by this CLI."""

import base64
import hashlib
import json
import time
from typing import Protocol
from dataclasses import dataclass
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import BaseModel, ConfigDict, Field, model_validator


class DeploymentInstance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    instance: str = Field(min_length=1, max_length=255)
    role: str = Field(min_length=1, max_length=64)


class MaintenancePayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    issuer: str
    nonce: str = Field(min_length=1, max_length=128)
    from_generation: str
    to_generation: str
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    deployment_id: str = Field(min_length=1, max_length=255)
    stopped_deployment_ref: str = Field(min_length=1, max_length=1024)
    issued_at: int
    expires_at: int
    previous_instances: tuple[DeploymentInstance, ...]
    instances: tuple[DeploymentInstance, ...]

    @model_validator(mode="after")
    def inventories(self):
        for inventory in (self.previous_instances, self.instances):
            if not inventory or len({v.instance for v in inventory}) != len(inventory):
                raise ValueError(
                    "complete unique deployment instance inventories required"
                )
        if (
            self.expires_at <= self.issued_at
            or self.expires_at - self.issued_at > 86400
        ):
            raise ValueError("maintenance receipt validity must be bounded to one day")
        return self


@dataclass(frozen=True)
class VerifiedMaintenanceReceipt:
    payload: MaintenancePayload
    sha256: str


class MaintenanceReceiptVerifier(Protocol):
    def verify(self, envelope: dict) -> VerifiedMaintenanceReceipt: ...


def canonical_receipt(payload):
    return json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()


class Ed25519MaintenanceReceiptVerifier:
    def __init__(self, public_key: bytes, *, issuer: str):
        self.key = Ed25519PublicKey.from_public_bytes(public_key)
        self.issuer = issuer

    def verify(self, envelope):
        if set(envelope) != {"payload", "signature"}:
            raise ValueError("invalid maintenance receipt envelope")
        encoded = canonical_receipt(envelope["payload"])
        try:
            self.key.verify(
                base64.b64decode(envelope["signature"], validate=True), encoded
            )
        except (InvalidSignature, ValueError) as error:
            raise ValueError("maintenance receipt signature invalid") from error
        payload = MaintenancePayload.model_validate(envelope["payload"])
        if payload.issuer != self.issuer:
            raise ValueError("maintenance receipt issuer untrusted")
        if payload.issued_at > time.time() or payload.expires_at <= time.time():
            raise ValueError("maintenance receipt expired or not yet valid")
        return VerifiedMaintenanceReceipt(payload, hashlib.sha256(encoded).hexdigest())
