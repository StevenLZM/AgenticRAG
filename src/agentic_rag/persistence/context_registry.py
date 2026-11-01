"""Immutable multi-generation membership and separately fenced rebuild jobs."""

from datetime import UTC, datetime, timedelta
import hashlib
import json
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Integer,
    String,
    Table,
    UniqueConstraint,
    insert,
    select,
    update,
    or_,
)
from agentic_rag.persistence.repositories import (
    metadata,
    documents,
    document_versions,
    parent_chunks,
    LeaseLost,
)
from agentic_rag.ingestion.state import IngestionJobClaim
from agentic_rag.models.contextual import ContextArtifactRef
from agentic_rag.ingestion.manifest import VersionManifest
from agentic_rag.persistence.context_release import (
    context_gate,
    check_context_admission,
    ReleaseNotReady,
)

retrieval_builds = Table(
    "retrieval_builds",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("user_id", String(255), nullable=False),
    Column("document_id", String(36), nullable=False),
    Column("document_version_id", String(36), nullable=False),
    Column("index_generation", String(128), nullable=False),
    Column("manifest_ref", JSON, nullable=False),
    Column("representation_version", String(32)),
    Column("build_fingerprint", String(64)),
    Column("build_config_hash", String(64)),
    Column("parent_hash", String(64), nullable=False),
    Column("child_identity_digest", String(64), nullable=False),
    Column("state", String(16), nullable=False),
    UniqueConstraint(
        "user_id",
        "document_version_id",
        "index_generation",
        name="uq_retrieval_build_generation",
    ),
)
retrieval_rebuild_jobs = Table(
    "retrieval_rebuild_jobs",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("user_id", String(255), nullable=False),
    Column("document_id", String(36), nullable=False),
    Column("document_version_id", String(36), nullable=False),
    Column("index_generation", String(128), nullable=False),
    Column("payload", JSON),
    Column("status", String(24), nullable=False),
    Column("attempt_count", Integer, nullable=False),
    Column("lease_owner", String(255)),
    Column("lease_expires_at", DateTime(timezone=True)),
    Column("error_code", String(128)),
    UniqueConstraint(
        "user_id",
        "document_version_id",
        "index_generation",
        name="uq_rebuild_generation",
    ),
)


def identity(owner, version, generation):
    return hashlib.sha256(json.dumps([owner, version, generation]).encode()).hexdigest()


class RebuildClaim(IngestionJobClaim):
    """Issued only by RebuildJobStore; distinct type selects the SQL fence."""

    target_generation: str


def rebuild_predicates(claim, now):
    return (
        retrieval_rebuild_jobs.c.id == claim.job_id,
        retrieval_rebuild_jobs.c.user_id == claim.user_id,
        retrieval_rebuild_jobs.c.document_id == claim.document_id,
        retrieval_rebuild_jobs.c.document_version_id == claim.document_version_id,
        retrieval_rebuild_jobs.c.index_generation == claim.target_generation,
        retrieval_rebuild_jobs.c.status == "running",
        retrieval_rebuild_jobs.c.lease_owner == claim.owner,
        retrieval_rebuild_jobs.c.attempt_count == claim.claim_generation,
        retrieval_rebuild_jobs.c.lease_expires_at > now,
    )


async def current_source(db, owner, document, version):
    row = (
        (
            await db.execute(
                select(document_versions)
                .select_from(
                    document_versions.join(
                        documents,
                        documents.c.active_version_id == document_versions.c.id,
                    )
                )
                .where(
                    documents.c.user_id == owner,
                    documents.c.id == document,
                    documents.c.status == "active",
                    documents.c.deletion_status.is_(None),
                    documents.c.deletion_fenced_at.is_(None),
                    document_versions.c.id == version,
                    document_versions.c.document_id == document,
                    document_versions.c.status == "active",
                )
                .with_for_update()
            )
        )
        .mappings()
        .one_or_none()
    )
    if row is None:
        raise ValueError("source missing, deleted, revoked or no longer active")
    return row


async def source_parent_hash(db, owner, document, version, *, parents=None):
    rows = (
        (
            await db.execute(
                select(parent_chunks)
                .where(
                    parent_chunks.c.user_id == owner,
                    parent_chunks.c.document_id == document,
                    parent_chunks.c.document_version_id == version,
                )
                .order_by(parent_chunks.c.id)
            )
        )
        .mappings()
        .all()
    )
    if not rows:
        raise ValueError("source Parents missing")
    fields = (
        "id",
        "user_id",
        "document_id",
        "document_version_id",
        "ordinal",
        "heading_path",
        "content_type",
        "content",
        "page_from",
        "page_to",
        "ast_locator",
        "content_hash",
    )

    def project(row):
        value = {key: row[key] for key in fields}
        if isinstance(value["ast_locator"], str):
            value["ast_locator"] = json.loads(value["ast_locator"])
        return value

    projected = [project(row) for row in rows]
    if parents is not None:
        expected = [
            project(parent.model_dump(mode="json"))
            for parent in sorted(parents, key=lambda p: p.id)
        ]
        if projected != expected:
            raise ValueError("rebuild changed original Parent bytes or coordinates")
    return hashlib.sha256(
        json.dumps(
            projected, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


async def register_build(
    db, context, manifest_ref, manifest, *, parents=None, build_config_hash=None
):
    from agentic_rag.ingestion.manifest import (
        child_identity_digest,
        validate_context_artifacts,
    )
    from agentic_rag.ingestion.chunker import ParentChunk

    if parents is None:
        artifacts = db.info["context_artifacts"]
        if manifest.schema_version == 2:
            parents = validate_context_artifacts(
                manifest,
                artifacts,
                (context.user_id, context.document_id, context.document_version_id),
            )
        else:
            raw = artifacts.read_json(
                artifacts.describe(
                    f"artifact://documents/{context.user_id}/{context.document_id}/{context.document_version_id}/chunks/{context.pipeline_version}/parents.json"
                )
            )
            parents = tuple(
                ParentChunk.model_validate(value) for value in raw["parents"]
            )
    if manifest.representation_version == "context-v1":
        build_config_hash = build_config_hash or db.info.get(
            "context_build_config_hash"
        )
        if not build_config_hash:
            raise ValueError("context-v1 registry requires frozen build configuration")
    identities = child_identity_digest(
        (child.id, child.content_hash)
        for parent in parents
        for child in parent.children
    )
    digest = await source_parent_hash(
        db,
        context.user_id,
        context.document_id,
        context.document_version_id,
        parents=parents,
    )
    values = dict(
        id=identity(
            context.user_id, context.document_version_id, context.index_generation
        ),
        user_id=context.user_id,
        document_id=context.document_id,
        document_version_id=context.document_version_id,
        index_generation=context.index_generation,
        manifest_ref=manifest_ref.model_dump(mode="json"),
        representation_version=manifest.representation_version,
        build_fingerprint=manifest.build_fingerprint,
        build_config_hash=build_config_hash,
        parent_hash=digest,
        child_identity_digest=identities,
        state="ready",
    )
    old = (
        (
            await db.execute(
                select(retrieval_builds)
                .where(retrieval_builds.c.id == values["id"])
                .with_for_update()
            )
        )
        .mappings()
        .one_or_none()
    )
    if old is not None:
        if dict(old) != values:
            raise ValueError("immutable build registry conflict")
    else:
        await db.execute(insert(retrieval_builds).values(**values))
    return values


class RegistrySourceAuthority:
    def __init__(self, factory, artifacts):
        self.factory, self.artifacts = factory, artifacts

    async def manifest(
        self, document_id, version_id, scope, snapshot, *, build_fingerprint=None
    ):
        async with self.factory.begin() as db:
            await check_context_admission(
                db, allow_draining=True, lock=False, snapshot=snapshot
            )
            await current_source(db, scope.user_id, document_id, version_id)
            row = (
                (
                    await db.execute(
                        select(retrieval_builds).where(
                            retrieval_builds.c.id
                            == identity(
                                scope.user_id, version_id, snapshot.index_generation
                            ),
                            retrieval_builds.c.state == "ready",
                        )
                    )
                )
                .mappings()
                .one_or_none()
            )
            if row is None or row["document_id"] != document_id:
                raise ValueError("current source has no compatible registered build")
            if (
                row["representation_version"]
                != snapshot.expected_representation_version
            ):
                raise ValueError(
                    "registered representation policy differs from frozen run"
                )
            if (
                snapshot.expected_representation_version == "context-v1"
                and row["build_config_hash"] != snapshot.context_build_fingerprint
            ):
                raise ValueError(
                    "registered build configuration differs from frozen generation"
                )
            if (
                build_fingerprint is not None
                and row["build_fingerprint"] != build_fingerprint
            ):
                raise ValueError("unregistered source build fingerprint")
            if row["parent_hash"] != await source_parent_hash(
                db, scope.user_id, document_id, version_id
            ):
                raise ValueError("original Parent source changed")
        ref = ContextArtifactRef.model_validate(row["manifest_ref"])
        if not self.artifacts.verify(ref):
            raise ValueError("registered manifest integrity failure")
        return ref


class RebuildJobStore:
    def __init__(self, factory):
        self.factory = factory

    async def enqueue(self, *, owner, document, version, generation, payload=None):
        async with self.factory.begin() as db:
            await check_context_admission(db, allow_draining=True)
            source = await current_source(db, owner, document, version)
            active = await db.scalar(
                select(context_gate.c.generation).where(context_gate.c.id == 1)
            )
            if generation in (source["index_generation"], active):
                raise ValueError("refuse active-generation in-place rebuild")
            return await enqueue_rebuild(
                db, owner, document, version, generation, payload
            )

    async def claim_next(self, owner, generation, *, lease_seconds=60):
        now = datetime.now(UTC)
        async with self.factory.begin() as db:
            await check_context_admission(db, allow_draining=True)
            await db.execute(
                update(retrieval_rebuild_jobs)
                .where(
                    retrieval_rebuild_jobs.c.status == "running",
                    retrieval_rebuild_jobs.c.lease_expires_at <= now,
                    retrieval_rebuild_jobs.c.attempt_count >= 3,
                )
                .values(
                    status="failed",
                    lease_owner=None,
                    lease_expires_at=None,
                    error_code="lease_attempts_exhausted",
                )
            )
            row = (
                (
                    await db.execute(
                        select(retrieval_rebuild_jobs)
                        .where(
                            retrieval_rebuild_jobs.c.index_generation == generation,
                            or_(
                                retrieval_rebuild_jobs.c.status == "queued",
                                (retrieval_rebuild_jobs.c.status == "running")
                                & (retrieval_rebuild_jobs.c.lease_expires_at <= now),
                            ),
                        )
                        .order_by(retrieval_rebuild_jobs.c.id)
                        .limit(1)
                        .with_for_update(skip_locked=True)
                    )
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                return None
            await current_source(
                db, row["user_id"], row["document_id"], row["document_version_id"]
            )
            result = await db.execute(
                update(retrieval_rebuild_jobs)
                .where(
                    retrieval_rebuild_jobs.c.id == row["id"],
                    retrieval_rebuild_jobs.c.attempt_count == row["attempt_count"],
                    or_(
                        retrieval_rebuild_jobs.c.status == "queued",
                        retrieval_rebuild_jobs.c.lease_expires_at <= now,
                    ),
                )
                .values(
                    status="running",
                    lease_owner=owner,
                    lease_expires_at=now + timedelta(seconds=lease_seconds),
                    attempt_count=row["attempt_count"] + 1,
                )
            )
            if result.rowcount != 1:
                return None
            return RebuildClaim(
                job_id=row["id"],
                user_id=row["user_id"],
                document_id=row["document_id"],
                document_version_id=row["document_version_id"],
                owner=owner,
                claim_generation=row["attempt_count"] + 1,
                target_generation=generation,
            ), row["payload"]

    async def heartbeat(self, claim, *, lease_seconds=60):
        async with self.factory.begin() as db:
            await current_source(
                db, claim.user_id, claim.document_id, claim.document_version_id
            )
            result = await db.execute(
                update(retrieval_rebuild_jobs)
                .where(*rebuild_predicates(claim, datetime.now(UTC)))
                .values(
                    lease_expires_at=datetime.now(UTC)
                    + timedelta(seconds=lease_seconds)
                )
            )
            if result.rowcount != 1:
                raise LeaseLost("rebuild lease lost")

    async def finish(self, claim, *, error=None):
        async with self.factory.begin() as db:
            if error is None and not await db.scalar(
                select(retrieval_builds.c.id).where(
                    retrieval_builds.c.id == claim.job_id,
                    retrieval_builds.c.state == "ready",
                )
            ):
                raise ValueError("completed rebuild requires durable ready build")
            result = await db.execute(
                update(retrieval_rebuild_jobs)
                .where(*rebuild_predicates(claim, datetime.now(UTC)))
                .values(
                    status=("queued" if claim.claim_generation < 3 else "failed")
                    if error
                    else "completed",
                    error_code=error,
                    lease_owner=None,
                    lease_expires_at=None,
                )
            )
            if result.rowcount != 1:
                raise LeaseLost("rebuild lease lost")


async def enqueue_rebuild(db, owner, document, version, generation, payload=None):
    key = identity(owner, version, generation)
    old = (
        (
            await db.execute(
                select(retrieval_rebuild_jobs)
                .where(retrieval_rebuild_jobs.c.id == key)
                .with_for_update()
            )
        )
        .mappings()
        .one_or_none()
    )
    if old is None:
        await db.execute(
            insert(retrieval_rebuild_jobs).values(
                id=key,
                user_id=owner,
                document_id=document,
                document_version_id=version,
                index_generation=generation,
                payload=payload,
                status="queued" if payload else "blocked_config",
                attempt_count=0,
            )
        )
    elif payload is not None:
        if old["payload"] is not None and old["payload"] != payload:
            raise ValueError("immutable rebuild request conflict")
        if old["status"] == "blocked_config":
            await db.execute(
                update(retrieval_rebuild_jobs)
                .where(retrieval_rebuild_jobs.c.id == key)
                .values(payload=payload, status="queued")
            )
    return key


async def publication_builds(db, target):
    """Runs inside publication commit so rollback incompleteness is never lost."""
    if not getattr(db, "info", {}).get("context_config_hash"):
        return
    artifacts = db.info["context_artifacts"]
    active_generation = await db.scalar(
        select(context_gate.c.generation).where(context_gate.c.id == 1)
    )
    if target.context.index_generation != active_generation:
        raise ReleaseNotReady("publication generation differs from selected deployment")
    await db.execute(
        update(retrieval_rebuild_jobs)
        .where(
            retrieval_rebuild_jobs.c.user_id == target.context.user_id,
            retrieval_rebuild_jobs.c.document_id == target.context.document_id,
            retrieval_rebuild_jobs.c.document_version_id
            != target.context.document_version_id,
        )
        .values(status="cancelled", lease_owner=None, lease_expires_at=None)
    )
    await db.execute(
        update(retrieval_builds)
        .where(
            retrieval_builds.c.user_id == target.context.user_id,
            retrieval_builds.c.document_id == target.context.document_id,
            retrieval_builds.c.document_version_id
            != target.context.document_version_id,
        )
        .values(state="superseded")
    )
    ref = artifacts.describe(target.manifest_uri)
    if ref.sha256 != target.manifest_hash:
        raise ValueError("publication manifest changed")
    manifest = VersionManifest.model_validate(artifacts.read_json(ref))
    await register_build(
        db, target.context, ContextArtifactRef(**ref.__dict__), manifest
    )
    rollback = await db.scalar(
        select(context_gate.c.rollback_generation).where(context_gate.c.id == 1)
    )
    if rollback and rollback != target.context.index_generation:
        await enqueue_rebuild(
            db,
            target.context.user_id,
            target.context.document_id,
            target.context.document_version_id,
            rollback,
        )


async def invalidate_document_builds(db, owner, document):
    if not getattr(db, "info", {}).get("context_config_hash"):
        return
    await db.execute(
        update(retrieval_builds)
        .where(
            retrieval_builds.c.user_id == owner,
            retrieval_builds.c.document_id == document,
        )
        .values(state="revoked")
    )
    await db.execute(
        update(retrieval_rebuild_jobs)
        .where(
            retrieval_rebuild_jobs.c.user_id == owner,
            retrieval_rebuild_jobs.c.document_id == document,
        )
        .values(status="cancelled", lease_owner=None, lease_expires_at=None)
    )
