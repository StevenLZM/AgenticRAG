"""Child-only rebuild of an active immutable source, with durable job recovery."""

import asyncio
from sqlalchemy import select, update
from agentic_rag.ingestion.assembler import CanonicalAst
from agentic_rag.ingestion.chunker import ParentChunk, ChildBuilder
from agentic_rag.ingestion.indexer import IndexWriter, StagingContext
from agentic_rag.ingestion.manifest import child_identity_digest
from agentic_rag.models.contextual import ContextArtifactRef
from agentic_rag.persistence.context_registry import (
    current_source,
    source_parent_hash,
    register_build,
    rebuild_predicates,
    retrieval_rebuild_jobs,
    RebuildJobStore,
)
from agentic_rag.persistence.context_release import (
    context_gate,
    check_context_admission,
)
from datetime import UTC, datetime


class _ExistingParents:
    """IndexWriter staging port that cannot mutate Parent or version state."""

    def __init__(self, factory, parents, claim):
        self.factory, self.parents, self.claim = factory, parents, claim
        self.manifest_ref = None

    async def assert_writable(self, context):
        async with self.factory.begin() as db:
            await check_context_admission(db, allow_draining=True)
            source = await current_source(
                db, context.user_id, context.document_id, context.document_version_id
            )
            active = await db.scalar(
                select(context_gate.c.generation).where(context_gate.c.id == 1)
            )
            if context.index_generation in (active, source["index_generation"]):
                raise ValueError("refuse in-place rebuild of active generation")
            await source_parent_hash(
                db,
                context.user_id,
                context.document_id,
                context.document_version_id,
                parents=self.parents,
            )
            fence = await db.execute(
                update(retrieval_rebuild_jobs)
                .where(*rebuild_predicates(self.claim, datetime.now(UTC)))
                .values(lease_owner=self.claim.owner)
            )
            if fence.rowcount != 1:
                raise RuntimeError("rebuild lease lost")

    async def stage(self, context, parents):
        await self.assert_writable(context)
        return len(self.parents)

    async def count(self, context):
        await self.assert_writable(context)
        return len(self.parents)

    async def attach_manifest(self, context, **values):
        await self.assert_writable(context)
        self.manifest_uri = values["manifest_uri"]


async def stage_search_representation(
    writer,
    parents,
    *,
    factory,
    claim,
    context,
    canonical_ast,
    context_artifact_ref=None,
    enriched_chunks_ref=None,
    before_side_effect=None,
    build_config_hash=None,
):
    if claim.target_generation != context.index_generation:
        raise ValueError("rebuild claim generation mismatch")
    readonly = _ExistingParents(factory, parents, claim)
    await readonly.assert_writable(context)
    staged = IndexWriter(
        embedding=writer._embedding,
        parent_store=readonly,
        child_store=writer._child_store,
        artifacts=writer._artifacts,
        embedding_batch_size=writer._embedding_batch_size,
        embedding_model=writer._embedding_model,
    )
    manifest = await staged.stage(
        parents,
        context=context,
        canonical_ast=canonical_ast,
        context_artifact_ref=context_artifact_ref,
        enriched_chunks_ref=enriched_chunks_ref,
        before_side_effect=before_side_effect,
    )
    # Both policies verify actual final IDs and search hashes, not only counts.
    identities = await writer._child_store.child_identities(context)
    if child_identity_digest(identities) != child_identity_digest(
        (child.id, child.content_hash)
        for parent in parents
        for child in parent.children
    ):
        raise ValueError("rebuild child identity collection mismatch")
    await readonly.assert_writable(context)
    await writer._child_store.activate(context)
    ref = writer._artifacts.describe(readonly.manifest_uri)
    async with factory.begin() as db:
        await current_source(
            db, context.user_id, context.document_id, context.document_version_id
        )
        result = await db.execute(
            update(retrieval_rebuild_jobs)
            .where(*rebuild_predicates(claim, datetime.now(UTC)))
            .values(lease_owner=claim.owner)
        )
        if result.rowcount != 1:
            raise RuntimeError("rebuild lease lost")
        await register_build(
            db,
            context,
            ContextArtifactRef(**ref.__dict__),
            manifest,
            parents=parents,
            build_config_hash=build_config_hash,
        )
    return manifest


class RebuildExecutor:
    def __init__(self, *, factory, artifacts, writer, tokenizer, contextualizer, build):
        self.factory, self.artifacts, self.writer = factory, artifacts, writer
        self.tokenizer, self.contextualizer, self.build = (
            tokenizer,
            contextualizer,
            build,
        )
        self.jobs = RebuildJobStore(factory)

    async def run(self, claim, payload):
        build_hash = self.build.fingerprint("0" * 64)
        if payload["config_hash"] != build_hash:
            raise ValueError(
                "rebuild request requires matching worker build configuration"
            )
        async with self.factory.begin() as db:
            source = await current_source(
                db, claim.user_id, claim.document_id, claim.document_version_id
            )
        if claim.target_generation == source["index_generation"]:
            raise ValueError("refuse active-generation in-place rebuild")
        canonical_ref = self.artifacts.describe(source["canonical_ast_path"])
        if (
            canonical_ref.sha256 != source["canonical_ast_hash"]
            or payload.get("canonical_ast_sha256", canonical_ref.sha256)
            != canonical_ref.sha256
        ):
            raise ValueError("source canonical AST changed")
        canonical = CanonicalAst.model_validate(self.artifacts.read_json(canonical_ref))
        # Recover original Parent boundaries from the immutable upload artifact.
        uri = f"artifact://documents/{claim.user_id}/{claim.document_id}/{claim.document_version_id}/chunks/{source['pipeline_version']}/parents.json"
        raw = self.artifacts.read_json(self.artifacts.describe(uri))
        builder = ChildBuilder(
            self.tokenizer, max_tokens=320 if self.build.enabled else 384
        )
        parents = tuple(ParentChunk.model_validate(value) for value in raw["parents"])
        async with self.factory.begin() as db:
            await source_parent_hash(
                db,
                claim.user_id,
                claim.document_id,
                claim.document_version_id,
                parents=parents,
            )
        parents = tuple(
            parent.model_copy(update={"children": tuple(builder.build(parent))})
            for parent in parents
        )

        async def fence():
            await self.jobs.heartbeat(claim)

        fields = {}
        if self.build.enabled:
            result = await self.contextualizer.enrich(
                canonical,
                parents,
                canonical_ref=ContextArtifactRef(**canonical_ref.__dict__),
                build=self.build,
                claim=claim,
                before_side_effect=fence,
            )
            parents = result.parents
            fields = dict(
                context_artifact_ref=result.context_artifact_ref,
                enriched_chunks_ref=result.enriched_chunks_ref,
            )
        context = StagingContext(
            user_id=claim.user_id,
            document_id=claim.document_id,
            document_version_id=claim.document_version_id,
            version_no=source["version_no"],
            pipeline_version=source["pipeline_version"],
            embedding_version=source["embedding_version"],
            index_generation=claim.target_generation,
        )
        await stage_search_representation(
            self.writer,
            parents,
            factory=self.factory,
            claim=claim,
            context=context,
            canonical_ast=canonical_ref,
            before_side_effect=fence,
            build_config_hash=build_hash if self.build.enabled else None,
            **fields,
        )

    async def run_once(self, owner, generation):
        claimed = await self.jobs.claim_next(owner, generation)
        if claimed is None:
            return False
        claim, payload = claimed

        async def keepalive():
            while True:
                await asyncio.sleep(15)
                await self.jobs.heartbeat(claim)

        heartbeat = asyncio.create_task(keepalive())
        operation = asyncio.create_task(self.run(claim, payload))
        try:
            done, _ = await asyncio.wait(
                (operation, heartbeat), return_when=asyncio.FIRST_COMPLETED
            )
            if heartbeat in done:
                await heartbeat
            await operation
            await self.jobs.finish(claim)
        except Exception as error:
            operation.cancel()
            await asyncio.gather(operation, return_exceptions=True)
            await self.jobs.finish(claim, error=type(error).__name__)
            raise
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
        return True
