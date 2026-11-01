"""Recoverable maintenance switch; DB and ES are deliberately separate steps."""

import hashlib
import json
from sqlalchemy import insert, select, update
from agentic_rag.persistence.context_registry import (
    retrieval_builds,
    current_source,
    source_parent_hash,
    identity,
)
from agentic_rag.persistence.context_release import (
    ReleaseStore,
    ReleaseNotReady,
    context_gate,
    release_journal,
    consume_receipt,
)
from agentic_rag.persistence.repositories import documents, document_versions
from agentic_rag.ingestion.manifest import VersionManifest
from agentic_rag.models.contextual import ContextArtifactRef


class ContextRelease:
    def __init__(self, factory, artifacts, aliases, verifier=None):
        self.factory, self.artifacts, self.aliases = factory, artifacts, aliases
        self.store = ReleaseStore(factory, verifier)

    async def mapping(self, generation):
        result = {}
        async with self.factory.begin() as db:
            sources = (
                await db.execute(
                    select(
                        documents.c.user_id,
                        documents.c.id,
                        documents.c.active_version_id,
                    ).where(
                        documents.c.status == "active",
                        documents.c.deletion_status.is_(None),
                        documents.c.deletion_fenced_at.is_(None),
                    )
                )
            ).all()
            for owner, document, version in sources:
                source = await current_source(db, owner, document, version)
                build = (
                    (
                        await db.execute(
                            select(retrieval_builds).where(
                                retrieval_builds.c.id
                                == identity(owner, version, generation),
                                retrieval_builds.c.state == "ready",
                            )
                        )
                    )
                    .mappings()
                    .one_or_none()
                )
                if build is None:
                    raise ReleaseNotReady(
                        f"missing compatible build: {document}/{version}/{generation}"
                    )
                if build["parent_hash"] != await source_parent_hash(
                    db, owner, document, version
                ):
                    raise ReleaseNotReady("registered Parent source changed")
                ref = ContextArtifactRef.model_validate(build["manifest_ref"])
                if not self.artifacts.verify(ref):
                    raise ReleaseNotReady("manifest artifact invalid")
                manifest = VersionManifest.model_validate(self.artifacts.read_json(ref))
                canonical = self.artifacts.describe(source["canonical_ast_path"])
                if (
                    canonical.sha256 != source["canonical_ast_hash"]
                    or manifest.canonical_ast_sha256 != canonical.sha256
                ):
                    raise ReleaseNotReady("canonical source integrity changed")
                if manifest.schema_version == 2:
                    from agentic_rag.ingestion.manifest import (
                        validate_context_artifacts,
                    )

                    parents = validate_context_artifacts(
                        manifest, self.artifacts, (owner, document, version)
                    )
                    await source_parent_hash(
                        db, owner, document, version, parents=parents
                    )
                if (
                    manifest.index_generation != generation
                    or manifest.build_fingerprint != build["build_fingerprint"]
                ):
                    raise ReleaseNotReady("manifest registry policy differs")
                result[build["id"]] = dict(build)
        return result

    @staticmethod
    def corpus_hash(mapping):
        values = sorted(
            (v["user_id"], v["document_id"], v["document_version_id"], v["parent_hash"])
            for v in mapping.values()
        )
        return hashlib.sha256(json.dumps(values).encode()).hexdigest()

    @staticmethod
    def validate_policy(mapping, snapshot):
        if any(
            build["representation_version"] != snapshot.expected_representation_version
            or build["build_config_hash"] != snapshot.context_build_fingerprint
            for build in mapping.values()
        ):
            raise ReleaseNotReady(
                "target build policy differs from frozen configuration"
            )

    async def prepare(
        self,
        journal_id,
        *,
        source,
        target,
        target_config_hash,
        target_snapshot=None,
        receipt=None,
    ):
        if (
            target_snapshot is None
            or target_snapshot.snapshot_id != target_config_hash
            or target_snapshot.index_generation != target
        ):
            raise ReleaseNotReady("release requires exact frozen target configuration")
        if source == target:
            raise ValueError("refuse active-generation in-place release")
        before, after = await self.mapping(source), await self.mapping(target)
        await self.store.assert_drained()
        if await self.mapping(source) != before or await self.mapping(target) != after:
            raise ReleaseNotReady(
                "corpus changed during drain; continue-drain and prepare again"
            )
        self.validate_policy(after, target_snapshot)
        if self.corpus_hash(before) != self.corpus_hash(after):
            raise ReleaseNotReady("source corpus changed during release preparation")
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
            if gate["execution_token"] is not None:
                raise ReleaseNotReady("release execution already owned")
            if gate["state"] != "frozen" or gate["generation"] != source:
                raise ReleaseNotReady("source deployment is not frozen")
            proof, inventory = await consume_receipt(
                db,
                self.store.verifier,
                receipt,
                operation=journal_id,
                source=source,
                target=target,
                config_hash=target_config_hash,
                previous=gate["inventory"],
                roles=gate["roles"],
            )
            await db.execute(
                update(context_gate)
                .where(context_gate.c.id == 1)
                .values(
                    active_journal_id=journal_id,
                    inventory=inventory,
                    receipt_hash=proof.sha256,
                    maintenance_receipt=receipt,
                    receipts={},
                )
            )
            values = dict(
                target_snapshot=target_snapshot.model_dump(mode="json"),
                receipt_hash=proof.sha256,
                maintenance_receipt=receipt,
                id=journal_id,
                stage="prepared",
                source_generation=source,
                target_generation=target,
                source_config_hash=gate["config_hash"],
                target_config_hash=target_config_hash,
                before_mapping=before,
                after_mapping=after,
                corpus_hash=self.corpus_hash(after),
            )
            old = (
                (
                    await db.execute(
                        select(release_journal).where(
                            release_journal.c.id == journal_id
                        )
                    )
                )
                .mappings()
                .one_or_none()
            )
            if old:
                if any(
                    old[key] != value for key, value in values.items() if key != "stage"
                ):
                    raise ValueError("release journal identity conflict")
            else:
                if await db.scalar(
                    select(release_journal.c.id).where(
                        release_journal.c.stage.not_in(("complete", "aborted"))
                    )
                ):
                    raise ReleaseNotReady("another journal is unfinished")
                await db.execute(insert(release_journal).values(**values))
        return values

    async def resume(self, journal_id, *, after_boundary=None):
        journal, token = await self.store.claim_execution(journal_id)
        if token is None:  # complete is immutable and has no external effects
            return
        try:
            await self._resume_owned(journal, token, after_boundary)
        finally:
            await self.store.release_execution(journal_id, token)

    async def _resume_owned(self, journal, token, after_boundary):
        from agentic_rag.runtime.models import RuntimeConfigSnapshot
        from agentic_rag.ingestion.indexer import StagingContext

        async def boundary(stage):
            if after_boundary:
                await after_boundary(stage)

        snapshot = RuntimeConfigSnapshot.model_validate(journal["target_snapshot"])
        if (snapshot.snapshot_id, snapshot.index_generation) != (
            journal["target_config_hash"],
            journal["target_generation"],
        ):
            raise ReleaseNotReady("journal frozen configuration identity differs")
        async with self.factory.begin() as db:
            gate = await self.store.execution(db, journal, token, allow_open=True)
            if gate["state"] == "open":
                if journal["stage"] != "verified" or (
                    gate["generation"],
                    gate["config_hash"],
                ) != (journal["target_generation"], journal["target_config_hash"]):
                    raise ReleaseNotReady("open gate differs from verified journal")
                await self._stage(db, journal, token, "complete", allow_open=True)
                return
        current = await self.mapping(journal["target_generation"])
        self.validate_policy(current, snapshot)
        if (
            current != journal["after_mapping"]
            or self.corpus_hash(current) != journal["corpus_hash"]
        ):
            raise ReleaseNotReady(
                "current corpus changed; prepare a complete release mapping"
            )
        if journal["stage"] == "prepared":
            async with self.factory.begin() as db:
                await self.store.execution(db, journal, token)
                for build in current.values():
                    await current_source(
                        db,
                        build["user_id"],
                        build["document_id"],
                        build["document_version_id"],
                    )
                    manifest = VersionManifest.model_validate(
                        self.artifacts.read_json(
                            ContextArtifactRef.model_validate(build["manifest_ref"])
                        )
                    )
                    await db.execute(
                        update(document_versions)
                        .where(document_versions.c.id == build["document_version_id"])
                        .values(
                            index_generation=build["index_generation"],
                            manifest_path=build["manifest_ref"]["uri"],
                            manifest_hash=build["manifest_ref"]["sha256"],
                            child_count=manifest.child_count,
                        )
                    )
                await self._stage(db, journal, token, "projected")
            await boundary("projected")
        if journal["stage"] == "projected":
            members = []
            async with self.factory.begin() as db:
                await self.store.execution(db, journal, token)
                for build in current.values():
                    version = await current_source(
                        db,
                        build["user_id"],
                        build["document_id"],
                        build["document_version_id"],
                    )
                    context = StagingContext(
                        user_id=build["user_id"],
                        document_id=build["document_id"],
                        document_version_id=build["document_version_id"],
                        version_no=version["version_no"],
                        pipeline_version=version["pipeline_version"],
                        embedding_version=version["embedding_version"],
                        index_generation=journal["target_generation"],
                    )
                    members.append((context, build["child_identity_digest"]))
            await self._effect(
                journal,
                token,
                "membership",
                lambda: self.aliases.prepare_release_membership(
                    journal["target_generation"],
                    members,
                    before_write=lambda: self._assert_effect(journal, token),
                ),
            )
            async with self.factory.begin() as db:
                await self._stage(db, journal, token, "membership")
            await boundary("membership")
        if journal["stage"] == "membership":
            await self._effect(
                journal,
                token,
                "alias",
                lambda: self.aliases.switch_active_alias(
                    journal["target_generation"],
                    before_write=lambda: self._assert_effect(journal, token),
                ),
            )
            async with self.factory.begin() as db:
                await self._stage(db, journal, token, "alias_switched")
            await boundary("alias_switched")
        if journal["stage"] == "alias_switched":
            await self.store.set_configuration(
                journal["target_generation"],
                journal["target_config_hash"],
                journal=journal,
                token=token,
            )
            async with self.factory.begin() as db:
                await self.store.execution(db, journal, token)
                await db.execute(
                    update(context_gate)
                    .where(context_gate.c.id == 1)
                    .values(rollback_generation=journal["source_generation"])
                )
                await self._stage(db, journal, token, "configured")
            await boundary("configured")
        if journal["stage"] == "configured":
            async with self.factory.begin() as db:
                await self.store.execution(db, journal, token)
            await self.aliases.validate_search_target(journal["target_generation"])
            checked = await self.mapping(journal["target_generation"])
            self.validate_policy(checked, snapshot)
            if checked != current:
                raise ReleaseNotReady("corpus changed before readiness")
            async with self.factory.begin() as db:
                await self._stage(db, journal, token, "verified")
            await boundary("verified")
        await self.store.reopen(journal=journal, token=token)
        await boundary("reopened")
        async with self.factory.begin() as db:
            await self._stage(db, journal, token, "complete", allow_open=True)

    async def _assert_effect(self, journal, token):
        async with self.factory.begin() as db:
            gate = await self.store.execution(db, journal, token)
            if gate["execution_effect"] is None:
                raise ReleaseNotReady("release external effect no longer owned")

    async def _effect(self, journal, token, name, invoke):
        # The durable latch is committed BEFORE dispatch. Uncertain failures keep
        # ownership forever, until an external authority proves ES quiescence.
        await self.store.mark_effect(journal, token, name)
        await invoke()
        await self.store.mark_effect(journal, token, None)

    async def rollback(self, journal_id, *, target_snapshot, receipt):
        """Compensate a partial or completed switch using CURRENT source builds.

        The original journal is retained; compensation is itself a resumable
        journal and never restores obsolete document pointers or deleted data.
        """
        old, token = await self.store.claim_execution(journal_id, compensation=True)
        try:
            return await self._rollback_owned(old, token, target_snapshot, receipt)
        finally:
            await self.store.release_execution(journal_id, token)

    async def _rollback_owned(self, old, token, target_snapshot, receipt):
        journal_id = old["id"]
        if (
            target_snapshot.index_generation != old["source_generation"]
            or target_snapshot.snapshot_id != old["source_config_hash"]
        ):
            raise ReleaseNotReady("rollback requires complete original configuration")
        await self.store.pause(token=token)
        target = await self.mapping(old["source_generation"])
        await self.store.assert_drained(token=token)
        if await self.mapping(old["source_generation"]) != target:
            raise ReleaseNotReady("rollback corpus changed during drain")
        self.validate_policy(target, target_snapshot)
        inverse_id = journal_id + ":rollback"
        if len(inverse_id) > 64:
            raise ValueError("journal id too long for rollback suffix")
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
            await self.store.execution(db, old, token)
            proof, inventory = await consume_receipt(
                db,
                self.store.verifier,
                receipt,
                operation=inverse_id,
                source=gate["generation"],
                target=old["source_generation"],
                config_hash=target_snapshot.snapshot_id,
                previous=gate["inventory"],
                roles=gate["roles"],
            )
            await db.execute(
                insert(release_journal).values(
                    id=inverse_id,
                    stage="prepared",
                    source_generation=gate["generation"],
                    target_generation=old["source_generation"],
                    source_config_hash=gate["config_hash"],
                    target_config_hash=target_snapshot.snapshot_id,
                    target_snapshot=target_snapshot.model_dump(mode="json"),
                    before_mapping=old["after_mapping"],
                    after_mapping=target,
                    corpus_hash=self.corpus_hash(target),
                    receipt_hash=proof.sha256,
                    maintenance_receipt=receipt,
                )
            )
            changed = await db.execute(
                update(release_journal)
                .where(
                    release_journal.c.id == journal_id,
                    release_journal.c.stage == old["stage"],
                )
                .values(stage="aborted")
            )
            if changed.rowcount != 1:
                raise ReleaseNotReady("compensation lost journal transition fence")
            changed = await db.execute(
                update(context_gate)
                .where(
                    context_gate.c.id == 1,
                    context_gate.c.active_journal_id == journal_id,
                    context_gate.c.execution_token == token,
                    context_gate.c.receipt_hash == old["receipt_hash"],
                )
                .values(
                    active_journal_id=inverse_id,
                    execution_token=None,
                    execution_effect=None,
                    inventory=inventory,
                    receipts={},
                    receipt_hash=proof.sha256,
                    maintenance_receipt=receipt,
                )
            )
            if changed.rowcount != 1:
                raise ReleaseNotReady("compensation lost execution ownership")
        await self.resume(inverse_id)
        return inverse_id

    async def _stage(self, db, journal, token, stage, *, allow_open=False):
        await self.store.execution(db, journal, token, allow_open=allow_open)
        sequence = (
            "prepared",
            "projected",
            "membership",
            "alias_switched",
            "configured",
            "verified",
            "complete",
        )
        previous = journal["stage"]
        if (
            previous not in sequence[:-1]
            or sequence.index(stage) != sequence.index(previous) + 1
        ):
            raise ReleaseNotReady("non-monotonic journal transition")
        changed = await db.execute(
            update(release_journal)
            .where(
                release_journal.c.id == journal["id"],
                release_journal.c.stage == previous,
                release_journal.c.receipt_hash == journal["receipt_hash"],
            )
            .values(stage=stage)
        )
        if changed.rowcount != 1:
            raise ReleaseNotReady("journal transition lost execution fence")
        journal["stage"] = stage
