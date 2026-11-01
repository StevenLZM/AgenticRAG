import base64
import time
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
from agentic_rag.ingestion.contextualization.maintenance import (
    Ed25519MaintenanceReceiptVerifier,
    canonical_receipt,
)


def signed_receipt(**updates):
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    payload = dict(
        issuer="test-deployer",
        nonce="nonce-1",
        from_generation="old-v1",
        to_generation="new-v1",
        config_hash="b" * 64,
        issued_at=int(time.time()) - 1,
        expires_at=int(time.time()) + 60,
        previous_instances=[{"instance": "old-api", "role": "api"}],
        instances=[{"instance": "new-api", "role": "api"}],
        stopped_deployment_ref="deployment-audit://stopped/old",
        deployment_id="new-deployment",
    )
    payload.update(updates)
    envelope = {
        "payload": payload,
        "signature": base64.b64encode(
            private.sign(canonical_receipt(payload))
        ).decode(),
    }
    return envelope, Ed25519MaintenanceReceiptVerifier(public, issuer="test-deployer")


def test_receipt_binds_inventory_hash_expiry_and_signature():
    envelope, verifier = signed_receipt()
    receipt = verifier.verify(envelope)
    assert receipt.payload.instances[0].instance == "new-api"
    envelope["payload"]["config_hash"] = "c" * 64
    with pytest.raises(ValueError, match="signature"):
        verifier.verify(envelope)
    expired, verifier = signed_receipt(
        issued_at=int(time.time()) - 100, expires_at=int(time.time()) - 1
    )
    with pytest.raises(ValueError, match="expired"):
        verifier.verify(expired)


def test_execution_recovery_requires_nonempty_stop_and_es_quiescence_references():
    fields = dict(
        journal_id="release-1",
        execution_token="a" * 32,
        maintenance_receipt_hash="b" * 64,
        effect="alias",
        stopped_executor_ref="proof://stopped",
        es_quiescence_ref="proof://settled",
    )
    for name in ("stopped_executor_ref", "es_quiescence_ref"):
        envelope, verifier = signed_receipt(execution_recovery={**fields, name: ""})
        with pytest.raises(ValueError):
            verifier.verify(envelope)
