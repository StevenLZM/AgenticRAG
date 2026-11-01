"""Maintenance admission, configuration receipts and resumable release journal.

An opted-in session locks the same gate row as maintenance transitions. The
legacy session path issues no SQL here and needs none of the additive tables.
"""

from uuid import uuid4

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
    Column("active_journal_id", String(64)),
    Column("execution_token", String(32)),
    Column("execution_effect", String(32)),
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
    Column("recovery_receipts", JSON),
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
    recovery=False,
):
    if verifier is None or envelope is None:
        raise ReleaseNotReady(
            "trusted external maintenance receipt authority is required"
        )
    receipt = verifier.verify(envelope)
    payload = receipt.payload
    if (payload.execution_recovery is not None) != recovery:
        raise ReleaseNotReady("maintenance receipt recovery purpose differs")
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

    @staticmethod
    def bind_journal(gate, journal):
        payload = gate["maintenance_receipt"]["payload"]
        if (
            gate["active_journal_id"] != journal["id"]
            or gate["receipt_hash"] != journal["receipt_hash"]
            or (payload["to_generation"], payload["config_hash"])
            != (journal["target_generation"], journal["target_config_hash"])
        ):
            raise ReleaseNotReady(
                "journal superseded or maintenance authorization differs"
            )
        if journal["stage"] == "aborted":
            raise ReleaseNotReady("aborted journal cannot execute")

    async def execution(self, db, journal, token, *, allow_open=False):
        gate = (
            (
                await db.execute(
                    select(context_gate).where(context_gate.c.id == 1).with_for_update()
                )
            )
            .mappings()
            .one()
        )
        self.bind_journal(gate, journal)
        if gate["execution_token"] != token or not token:
            raise ReleaseNotReady("release execution ownership lost")
        if gate["state"] != "frozen" and not allow_open:
            raise ReleaseNotReady("release execution requires frozen admission")
        return gate

    async def claim_execution(self, journal_id, *, compensation=False):
        token = uuid4().hex
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
            journal = dict(
                (
                    await db.execute(
                        select(release_journal).where(
                            release_journal.c.id == journal_id
                        )
                    )
                )
                .mappings()
                .one()
            )
            if journal["stage"] == "aborted":
                raise ReleaseNotReady("aborted journal cannot execute")
            if journal["stage"] == "complete" and not compensation:
                return journal, None
            self.bind_journal(gate, journal)
            if gate["execution_token"] is not None:
                raise ReleaseNotReady(
                    "release execution already owned; recovery requires trusted quiescence"
                )
            if (
                gate["state"] != "frozen"
                and not (
                    gate["state"] == "open"
                    and journal["stage"] in ("verified", "complete")
                )
                and not (compensation and gate["state"] == "draining")
            ):
                raise ReleaseNotReady("release execution requires frozen admission")
            changed = await db.execute(
                update(context_gate)
                .where(
                    context_gate.c.id == 1,
                    context_gate.c.execution_token.is_(None),
                    context_gate.c.active_journal_id == journal_id,
                    context_gate.c.receipt_hash == journal["receipt_hash"],
                )
                .values(execution_token=token, execution_effect=None)
            )
            if changed.rowcount != 1:
                raise ReleaseNotReady("release execution already owned")
        return journal, token

    async def release_execution(self, journal_id, token):
        # A timed-out/cancelled ES request may still run remotely. Never expire
        # or clear its ownership merely because this coroutine returned.
        async with self.factory.begin() as db:
            await db.execute(
                update(context_gate)
                .where(
                    context_gate.c.id == 1,
                    context_gate.c.active_journal_id == journal_id,
                    context_gate.c.execution_token == token,
                    context_gate.c.execution_effect.is_(None),
                )
                .values(execution_token=None)
            )

    async def mark_effect(self, journal, token, effect):
        async with self.factory.begin() as db:
            gate = await self.execution(db, journal, token)
            if effect is not None and gate["execution_effect"] is not None:
                raise ReleaseNotReady(
                    "release execution has an uncertain external effect"
                )
            await db.execute(
                update(context_gate)
                .where(
                    context_gate.c.id == 1,
                    context_gate.c.execution_token == token,
                )
                .values(execution_effect=effect)
            )

    async def execution_status(self, journal_id):
        """Read the exact recovery binding; no proof issuance or state mutation."""
        async with self.factory() as db:
            gate = (
                (await db.execute(select(context_gate).where(context_gate.c.id == 1)))
                .mappings()
                .one()
            )
            journal = (
                (
                    await db.execute(
                        select(release_journal).where(
                            release_journal.c.id == journal_id
                        )
                    )
                )
                .mappings()
                .one()
            )
            self.bind_journal(gate, journal)
            return dict(
                journal_id=journal_id,
                stage=journal["stage"],
                state=gate["state"],
                execution_token=gate["execution_token"],
                effect=gate["execution_effect"],
                maintenance_receipt_hash=gate["receipt_hash"],
                from_generation=gate["generation"],
                to_generation=journal["target_generation"],
                config_hash=journal["target_config_hash"],
                instances=gate["inventory"],
                previous_instances=gate["inventory"],
            )

    async def recover_execution(self, journal_id, envelope):
        """Consume trusted stopped-executor + remote-write-quiescence evidence.

        No timer, local PID probe or caller flag can clear execution ownership.
        The external authority must stop the executor and settle ALL its ES
        requests before issuing this receipt. A late local continuation still
        loses its token and cannot commit or dispatch the next write.
        """
        if self.verifier is None or envelope is None:
            raise ReleaseNotReady("trusted execution recovery authority is required")
        verified = self.verifier.verify(envelope)
        proof = verified.payload.execution_recovery
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
            journal = (
                (
                    await db.execute(
                        select(release_journal).where(
                            release_journal.c.id == journal_id
                        )
                    )
                )
                .mappings()
                .one()
            )
            self.bind_journal(gate, journal)
            if (
                proof is None
                or not gate["execution_token"]
                or (
                    proof.journal_id,
                    proof.execution_token,
                    proof.maintenance_receipt_hash,
                    proof.effect,
                )
                != (
                    journal_id,
                    gate["execution_token"],
                    gate["receipt_hash"],
                    gate["execution_effect"],
                )
            ):
                raise ReleaseNotReady("execution recovery does not match current fence")
            if gate["execution_effect"] is not None and gate["state"] != "frozen":
                raise ReleaseNotReady(
                    "uncertain execution recovery requires frozen admission"
                )
            _, inventory = await consume_receipt(
                db,
                self.verifier,
                envelope,
                operation=f"recover:{proof.execution_token}",
                source=gate["generation"],
                target=journal["target_generation"],
                config_hash=journal["target_config_hash"],
                previous=gate["inventory"],
                roles=gate["roles"],
                recovery=True,
            )
            if sorted(inventory, key=lambda v: v["instance"]) != sorted(
                gate["inventory"], key=lambda v: v["instance"]
            ):
                raise ReleaseNotReady(
                    "execution recovery cannot replace deployment inventory"
                )
            changed = await db.execute(
                update(context_gate)
                .where(
                    context_gate.c.id == 1,
                    context_gate.c.active_journal_id == journal_id,
                    context_gate.c.receipt_hash == proof.maintenance_receipt_hash,
                    context_gate.c.execution_token == proof.execution_token,
                )
                .values(execution_token=None, execution_effect=None)
            )
            if changed.rowcount != 1:
                raise ReleaseNotReady("execution recovery lost current fence")
            await db.execute(
                update(release_journal)
                .where(release_journal.c.id == journal_id)
                .values(
                    recovery_receipts=[*(journal["recovery_receipts"] or []), envelope]
                )
            )

    async def pause(self, *, token=None):
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
            if gate["execution_token"] is not None and gate["execution_token"] != token:
                raise ReleaseNotReady("release execution already owned")
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
            if gate["execution_token"] is not None:
                raise ReleaseNotReady("release execution already owned")
            if pending or gate["state"] not in ("frozen", "draining"):
                raise ReleaseNotReady(
                    "unfinished release must be resumed or compensated first"
                )
            await db.execute(
                update(context_gate)
                .where(context_gate.c.id == 1)
                .values(state="draining")
            )

    async def assert_drained(self, *, token=None):
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
            if row["execution_token"] is not None and row["execution_token"] != token:
                raise ReleaseNotReady("release execution already owned")
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

    async def set_configuration(
        self, generation, config_hash, *, journal=None, token=None
    ):
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
            if row["active_journal_id"] is not None:
                if journal is None:
                    raise ReleaseNotReady(
                        "configuration switch requires active journal execution"
                    )
                await self.execution(db, journal, token)
                if (generation, config_hash) != (
                    journal["target_generation"],
                    journal["target_config_hash"],
                ):
                    raise ReleaseNotReady("configuration differs from journal target")
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

    async def reopen(self, *, journal=None, token=None):
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
            if row["active_journal_id"] is not None:
                active = (
                    (
                        await db.execute(
                            select(release_journal).where(
                                release_journal.c.id == row["active_journal_id"]
                            )
                        )
                    )
                    .mappings()
                    .one()
                )
                self.bind_journal(row, active)
                if active["stage"] != "complete":
                    if journal is None or journal["id"] != active["id"]:
                        raise ReleaseNotReady(
                            "reopen requires active journal execution"
                        )
                    await self.execution(db, active, token)
                    if active["stage"] != "verified":
                        raise ReleaseNotReady(
                            "release journal has unfinished boundaries"
                        )
                if row["execution_effect"] is not None:
                    raise ReleaseNotReady(
                        "release execution has an uncertain external effect"
                    )
                if (row["generation"], row["config_hash"]) != (
                    active["target_generation"],
                    active["target_config_hash"],
                ):
                    raise ReleaseNotReady(
                        "reopen configuration differs from journal target"
                    )
            if row["execution_token"] is not None and row["execution_token"] != token:
                raise ReleaseNotReady("release execution already owned")
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
