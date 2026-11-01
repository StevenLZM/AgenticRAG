"""Maintenance admission, configuration receipts and resumable release journal.

An opted-in session locks the same gate row as maintenance transitions. The
legacy session path issues no SQL here and needs none of the additive tables.
"""

from sqlalchemy import JSON, Column, Integer, String, Table, insert, select, update
from agentic_rag.persistence.repositories import (
    metadata,
    ingestion_jobs,
    agent_runs,
    document_versions,
)

context_gate = Table(
    "context_release_gate",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("state", String(16), nullable=False),
    Column("generation", String(128), nullable=False),
    Column("config_hash", String(64), nullable=False),
    Column("epoch", Integer, nullable=False),
    Column("roles", JSON, nullable=False),
    Column("inventory", JSON, nullable=False),
    Column("receipt_hash", String(64), nullable=False),
    Column("maintenance_receipt", JSON, nullable=False),
    Column("receipts", JSON, nullable=False),
    Column("rollback_generation", String(128)),
)
release_journal = Table(
    "context_release_journal",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("stage", String(32), nullable=False),
    Column("source_generation", String(128), nullable=False),
    Column("target_generation", String(128), nullable=False),
    Column("source_config_hash", String(64), nullable=False),
    Column("target_config_hash", String(64), nullable=False),
    Column("target_snapshot", JSON, nullable=False),
    Column("before_mapping", JSON, nullable=False),
    Column("after_mapping", JSON, nullable=False),
    Column("corpus_hash", String(64), nullable=False),
    Column("receipt_hash", String(64), nullable=False),
    Column("maintenance_receipt", JSON, nullable=False),
)

maintenance_nonces = Table(
    "context_maintenance_nonces",
    metadata,
    Column("nonce", String(128), primary_key=True),
    Column("receipt_hash", String(64), nullable=False),
    Column("operation", String(128), nullable=False),
)


async def consume_receipt(
    db,
    verifier,
    envelope,
    *,
    operation,
    source,
    target,
    config_hash,
    previous=None,
    roles=None,
):
    if verifier is None or envelope is None:
        raise ReleaseNotReady(
            "trusted external maintenance receipt authority is required"
        )
    receipt = verifier.verify(envelope)
    payload = receipt.payload
    if (payload.from_generation, payload.to_generation, payload.config_hash) != (
        source,
        target,
        config_hash,
    ):
        raise ReleaseNotReady("maintenance receipt generation/configuration mismatch")
    inventory = [value.model_dump(mode="json") for value in payload.instances]
    old_inventory = [
        value.model_dump(mode="json") for value in payload.previous_instances
    ]
    if previous is not None and sorted(
        old_inventory, key=lambda v: v["instance"]
    ) != sorted(previous, key=lambda v: v["instance"]):
        raise ReleaseNotReady(
            "maintenance receipt has missing or unknown old instances"
        )
    if roles is not None and {value["role"] for value in inventory} != set(roles):
        raise ReleaseNotReady(
            "maintenance receipt does not cover every configured role"
        )
    old = (
        (
            await db.execute(
                select(maintenance_nonces).where(
                    maintenance_nonces.c.nonce == payload.nonce
                )
            )
        )
        .mappings()
        .one_or_none()
    )
    if old is not None:
        raise ReleaseNotReady("maintenance receipt nonce already consumed")
    await db.execute(
        insert(maintenance_nonces).values(
            nonce=payload.nonce, receipt_hash=receipt.sha256, operation=operation
        )
    )
    return receipt, inventory


class ReleaseNotReady(RuntimeError):
    """Traffic remains closed until durable readiness is satisfied."""


async def check_context_admission(
    db, *, allow_draining=False, lock=True, snapshot=None
):
    expected = getattr(db, "info", {}).get("context_config_hash")
    if expected is None:
        return
    if (
        snapshot is not None
        and snapshot.model_copy(update={"evaluation": None}).snapshot_id != expected
    ):
        raise ReleaseNotReady(
            "run configuration requires compatible deployment or reprepare"
        )
    statement = select(context_gate).where(context_gate.c.id == 1)
    row = (
        (await db.execute(statement.with_for_update() if lock else statement))
        .mappings()
        .one_or_none()
    )
    if row is None or row["config_hash"] != expected:
        raise ReleaseNotReady(
            "process configuration differs from durable generation policy"
        )
    if row["state"] != "open" and not (allow_draining and row["state"] == "draining"):
        raise ReleaseNotReady("maintenance admission closed")


class ReleaseStore:
    def __init__(self, factory, verifier=None):
        self.factory, self.verifier = factory, verifier

    async def install(
        self, *, generation, config_hash, roles, receipt=None, source_generation=None
    ):
        """Explicit maintenance provisioning; never called by startup."""
        if source_generation is not None and source_generation != generation:
            raise ReleaseNotReady(
                "initial arm cannot replace a generation; use a release journal"
            )
        if not roles or len(set(roles)) != len(roles):
            raise ValueError("explicit deployment roles required")
        async with self.factory.begin() as db:
            proof, inventory = await consume_receipt(
                db,
                self.verifier,
                receipt,
                operation="install",
                source=source_generation or generation,
                target=generation,
                config_hash=config_hash,
                roles=roles,
            )
            await db.execute(
                insert(context_gate).values(
                    id=1,
                    state="frozen",
                    generation=generation,
                    config_hash=config_hash,
                    epoch=0,
                    roles=list(roles),
                    inventory=inventory,
                    receipt_hash=proof.sha256,
                    maintenance_receipt=receipt,
                    receipts={},
                )
            )

    async def pause(self):
        async with self.factory.begin() as db:
            await db.execute(
                update(context_gate)
                .where(context_gate.c.id == 1, context_gate.c.state == "open")
                .values(state="draining", receipts={})
            )

    async def continue_draining(self):
        """Repair pre-journal readiness without reopening user admission."""
        async with self.factory.begin() as db:
            gate = (
                (
                    await db.execute(
                        select(context_gate)
                        .where(context_gate.c.id == 1)
                        .with_for_update()
                    )
                )
                .mappings()
                .one()
            )
            pending = await db.scalar(
                select(release_journal.c.id)
                .where(release_journal.c.stage.not_in(("complete", "aborted")))
                .limit(1)
            )
            if pending or gate["state"] not in ("frozen", "draining"):
                raise ReleaseNotReady(
                    "unfinished release must be resumed or compensated first"
                )
            await db.execute(
                update(context_gate)
                .where(context_gate.c.id == 1)
                .values(state="draining")
            )

    async def assert_drained(self):
        from agentic_rag.orchestration import schema
        from agentic_rag.persistence.context_registry import retrieval_rebuild_jobs

        async with self.factory.begin() as db:
            row = (
                (
                    await db.execute(
                        select(context_gate)
                        .where(context_gate.c.id == 1)
                        .with_for_update()
                    )
                )
                .mappings()
                .one()
            )
            if row["state"] not in ("draining", "frozen"):
                raise ReleaseNotReady("pause admission before drain")
            for name, table, states in (
                ("ingestion", ingestion_jobs, ("queued", "running")),
                ("source publication", document_versions, ("uploaded", "building")),
                ("rebuild", retrieval_rebuild_jobs, ("queued", "running")),
                ("query", agent_runs, ("queued", "running", "cancel_requested")),
                ("supervisor", schema.runs, ("active", "finalizing")),
                ("work", schema.work_items, ("queued", "running", "retry_wait")),
            ):
                if await db.scalar(
                    select(table.c.id).where(table.c.status.in_(states)).limit(1)
                ):
                    raise ReleaseNotReady(
                        f"{name} tasks have not drained or been cancelled"
                    )
            await db.execute(
                update(context_gate)
                .where(context_gate.c.id == 1)
                .values(state="frozen")
            )

    async def set_configuration(self, generation, config_hash):
        async with self.factory.begin() as db:
            row = (
                (
                    await db.execute(
                        select(context_gate)
                        .where(context_gate.c.id == 1)
                        .with_for_update()
                    )
                )
                .mappings()
                .one()
            )
            if row["state"] != "frozen":
                raise ReleaseNotReady("configuration switch requires frozen gate")
            if (row["generation"], row["config_hash"]) != (generation, config_hash):
                await db.execute(
                    update(context_gate)
                    .where(context_gate.c.id == 1)
                    .values(
                        generation=generation,
                        config_hash=config_hash,
                        epoch=row["epoch"] + 1,
                        receipts={},
                    )
                )

    async def attest(self, role, config_hash, *, instance=None):
        """Called after role dependency validation, bound to this process session.

        config_hash is a comparison input, never a substitute for the process's
        immutable session policy installed by production composition.
        """
        async with self.factory.begin() as db:
            actual_instance = db.info.get("context_instance_id")
            if instance is None:
                instance = actual_instance
            if not instance or actual_instance != instance:
                raise ReleaseNotReady(
                    "receipt must match actual process instance identity"
                )
            if db.info.get("context_config_hash") != config_hash:
                raise ReleaseNotReady("receipt must match actual process configuration")
            row = (
                (
                    await db.execute(
                        select(context_gate)
                        .where(context_gate.c.id == 1)
                        .with_for_update()
                    )
                )
                .mappings()
                .one()
            )
            if (
                row["config_hash"] != config_hash
                or {"instance": instance, "role": role} not in row["inventory"]
            ):
                raise ReleaseNotReady("role configuration is not selected")
            receipts = dict(row["receipts"])
            receipts[instance] = {
                "role": role,
                "config_hash": config_hash,
                "epoch": row["epoch"],
            }
            await db.execute(
                update(context_gate)
                .where(context_gate.c.id == 1)
                .values(receipts=receipts)
            )

    async def reopen(self):
        async with self.factory.begin() as db:
            row = (
                (
                    await db.execute(
                        select(context_gate)
                        .where(context_gate.c.id == 1)
                        .with_for_update()
                    )
                )
                .mappings()
                .one()
            )
            if row["state"] != "frozen":
                raise ReleaseNotReady("reopen requires frozen gate")
            pending = await db.scalar(
                select(release_journal.c.id)
                .where(
                    release_journal.c.stage.not_in(("verified", "complete", "aborted"))
                )
                .limit(1)
            )
            if pending:
                raise ReleaseNotReady("release journal has unfinished boundaries")
            for member in row["inventory"]:
                if row["receipts"].get(member["instance"]) != {
                    "role": member["role"],
                    "config_hash": row["config_hash"],
                    "epoch": row["epoch"],
                }:
                    raise ReleaseNotReady(
                        "deployment roles have not all restarted and validated"
                    )
            await db.execute(
                update(context_gate).where(context_gate.c.id == 1).values(state="open")
            )
