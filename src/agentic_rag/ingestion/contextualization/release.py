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
        if any(
            value["representation_version"]
            != target_snapshot.expected_representation_version
            or value["build_config_hash"] != target_snapshot.context_build_fingerprint
            for value in after.values()
        ):
            raise ReleaseNotReady(
                "target build representation differs from frozen configuration"
            )
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
        async def boundary(stage):
            if after_boundary:
                await after_boundary(stage)

        async with self.factory() as db:
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
        if journal["stage"] == "complete":
            return
        async with self.factory() as db:
            gate = (
                (await db.execute(select(context_gate).where(context_gate.c.id == 1)))
                .mappings()
                .one()
            )
        if (
            gate["state"] == "open"
            and journal["stage"] == "verified"
            and (gate["generation"], gate["config_hash"])
            == (journal["target_generation"], journal["target_config_hash"])
        ):
            await self._stage(journal_id, "complete")
            return
        current = await self.mapping(journal["target_generation"])
        if (
            current != journal["after_mapping"]
            or self.corpus_hash(current) != journal["corpus_hash"]
        ):
            raise ReleaseNotReady(
                "current corpus changed; prepare a complete release mapping"
            )
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
            if gate["state"] != "frozen":
                raise ReleaseNotReady("release requires frozen admission")
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
            await db.execute(
                update(release_journal)
                .where(release_journal.c.id == journal_id)
                .values(stage="projected")
            )
        await boundary("projected")
        from agentic_rag.ingestion.indexer import StagingContext

        members = []
        async with self.factory.begin() as db:
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
        await self.aliases.prepare_release_membership(
            journal["target_generation"], members
        )
        await boundary("membership")
        await self.aliases.switch_active_alias(journal["target_generation"])
        await self._stage(journal_id, "alias_switched")
        await boundary("alias_switched")
        await self.store.set_configuration(
            journal["target_generation"], journal["target_config_hash"]
        )
        async with self.factory.begin() as db:
            await db.execute(
                update(context_gate)
                .where(context_gate.c.id == 1)
                .values(rollback_generation=journal["source_generation"])
            )
        await self._stage(journal_id, "configured")
        await boundary("configured")
        # Alias adapter verifies the actual target mapping and alias readback.
        await self.aliases.validate_search_target(journal["target_generation"])
        if await self.mapping(journal["target_generation"]) != current:
            raise ReleaseNotReady("corpus changed before readiness")
        await self._stage(journal_id, "verified")
        await boundary("verified")
        # Processes must restart with matching generation, compiler, tokenizer and
        # representation policy, and attest through their actual session config.
        await self.store.reopen()
        await boundary("reopened")
        await self._stage(journal_id, "complete")

    async def rollback(self, journal_id, *, target_snapshot, receipt):
        """Compensate a partial or completed switch using CURRENT source builds.

        The original journal is retained; compensation is itself a resumable
        journal and never restores obsolete document pointers or deleted data.
        """
        async with self.factory() as db:
            old = (
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
        if (
            target_snapshot.index_generation != old["source_generation"]
            or target_snapshot.snapshot_id != old["source_config_hash"]
        ):
            raise ReleaseNotReady("rollback requires complete original configuration")
        await self.store.pause()
        target = await self.mapping(old["source_generation"])
        await self.store.assert_drained()
        if await self.mapping(old["source_generation"]) != target:
            raise ReleaseNotReady("rollback corpus changed during drain")
        if any(
            build["representation_version"]
            != target_snapshot.expected_representation_version
            for build in target.values()
        ):
            raise ReleaseNotReady("rollback build policy is incompatible")
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
            await db.execute(
                update(release_journal)
                .where(release_journal.c.id == journal_id)
                .values(stage="aborted")
            )
            await db.execute(
                update(context_gate)
                .where(context_gate.c.id == 1)
                .values(
                    inventory=inventory,
                    receipts={},
                    receipt_hash=proof.sha256,
                    maintenance_receipt=receipt,
                )
            )
        await self.resume(inverse_id)
        return inverse_id

    async def _stage(self, journal_id, stage):
        async with self.factory.begin() as db:
            await db.execute(
                update(release_journal)
                .where(release_journal.c.id == journal_id)
                .values(stage=stage)
            )
