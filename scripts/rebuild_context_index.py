#!/usr/bin/env python3
"""Explicit maintenance tooling; every command defaults to an offline plan."""

from __future__ import annotations
import argparse
import asyncio
import base64
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "command",
        choices=(
            "install",
            "pause",
            "continue-drain",
            "open",
            "import-source",
            "enqueue",
            "work",
            "release",
            "resume",
            "recover-execution",
            "execution-status",
            "rollback",
        ),
    )
    value.add_argument("--source", required=True)
    value.add_argument("--target", required=True)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--config", type=Path, required=True)
    value.add_argument("--target-config", type=Path, required=True)
    value.add_argument("--journal")
    value.add_argument("--maintenance-receipt", type=Path)
    value.add_argument("--execute", action="store_true")
    value.add_argument("--max-jobs", type=int, default=1)
    return value


def load_plan(args):
    from agentic_rag.config import Settings
    from agentic_rag.runtime.query_composition import build_query_snapshot
    from agentic_rag.runtime.context_composition import validate_context_configuration

    current = Settings(**json.loads(args.config.read_text()))
    target = Settings(**json.loads(args.target_config.read_text()))
    validate_context_configuration(current)
    validate_context_configuration(target)
    manifest = json.loads(args.manifest.read_text())
    snapshot = build_query_snapshot(target)
    if (
        current.index_generation != args.source
        or target.index_generation != args.target
    ):
        raise ValueError("explicit source/target must match their configuration files")
    if (
        manifest.get("source_generation") != args.source
        or manifest.get("target_generation") != args.target
        or manifest.get("target_snapshot_id") != snapshot.snapshot_id
    ):
        raise ValueError(
            "manifest does not bind exact source, target and target snapshot"
        )
    if args.command in ("enqueue", "work", "release") and args.source == args.target:
        raise ValueError("refuse active-generation in-place rebuild/replacement")
    if not 1 <= args.max_jobs <= 1000:
        raise ValueError("max-jobs must be finite and between 1 and 1000")
    if (
        args.command
        in ("release", "resume", "rollback", "recover-execution", "execution-status")
        and not args.journal
    ):
        raise ValueError("release/resume require explicit durable journal id")
    return current, target, manifest, snapshot


async def execute(args, current, target, manifest, snapshot):
    from agentic_rag.bootstrap import build_container
    from agentic_rag.persistence.context_release import (
        ReleaseStore,
        check_context_admission,
    )
    from agentic_rag.ingestion.contextualization.release import ContextRelease
    from agentic_rag.ingestion.contextualization.maintenance import (
        Ed25519MaintenanceReceiptVerifier,
    )
    from agentic_rag.persistence.elasticsearch import ElasticsearchChildIndexStore
    from agentic_rag.persistence.context_registry import (
        RebuildJobStore,
        current_source,
        register_build,
    )
    from agentic_rag.runtime.context_composition import (
        build_contextualizer,
        validate_context_role,
    )
    from agentic_rag.models.contextual import ContextArtifactRef
    from agentic_rag.ingestion.manifest import (
        VersionManifest,
        child_identity_digest,
        validate_context_artifacts,
    )
    from agentic_rag.ingestion.chunker import ParentChunk
    from agentic_rag.ingestion.indexer import StagingContext

    if not current.context_release_enabled:
        raise ValueError(
            "execute requires independently enabled durable release control"
        )
    verifier = None
    if current.context_maintenance_public_key and current.context_maintenance_issuer:
        verifier = Ed25519MaintenanceReceiptVerifier(
            base64.b64decode(current.context_maintenance_public_key, validate=True),
            issuer=current.context_maintenance_issuer,
        )
    receipt = (
        json.loads(args.maintenance_receipt.read_text())
        if args.maintenance_receipt
        else None
    )
    if args.command in ("install", "release", "rollback", "recover-execution") and (
        verifier is None or receipt is None
    ):
        raise ValueError(
            "trusted external deployment authority and signed maintenance receipt required"
        )
    container = build_container(current)
    factory, artifacts = container.repositories.session_factory, container.artifacts
    aliases = ElasticsearchChildIndexStore(
        container.elasticsearch,
        embedding_model=target.embedding_model,
        lexical_analysis=target.lexical_analysis,
        expected_representation_version=target.expected_representation_version,
    )
    try:
        store = ReleaseStore(factory, verifier)
        if args.command == "install":
            await store.install(
                generation=args.target,
                source_generation=args.source,
                config_hash=snapshot.snapshot_id,
                roles=target.context_deployment_roles,
                receipt=receipt,
            )
        elif args.command == "pause":
            await store.pause()
        elif args.command == "continue-drain":
            await store.continue_draining()
        elif args.command == "open":
            await store.reopen()
        elif args.command == "release":
            release = ContextRelease(factory, artifacts, aliases, verifier)
            await release.prepare(
                args.journal,
                source=args.source,
                target=args.target,
                target_config_hash=snapshot.snapshot_id,
                target_snapshot=snapshot,
                receipt=receipt,
            )
            await release.resume(args.journal)
        elif args.command == "rollback":
            await ContextRelease(factory, artifacts, aliases, verifier).rollback(
                args.journal, target_snapshot=snapshot, receipt=receipt
            )
        elif args.command == "execution-status":
            print(
                json.dumps(await store.execution_status(args.journal), sort_keys=True)
            )
        elif args.command == "recover-execution":
            await store.recover_execution(args.journal, receipt)
        elif args.command == "resume":
            await ContextRelease(factory, artifacts, aliases, verifier).resume(
                args.journal
            )
        elif args.command in ("import-source", "enqueue"):
            entries = manifest.get("documents", [])
            if not entries:
                raise ValueError("explicit source document/version manifest required")
            for entry in entries:
                async with factory.begin() as db:
                    await check_context_admission(db, allow_draining=True)
                    source = await current_source(
                        db,
                        entry["user_id"],
                        entry["document_id"],
                        entry["document_version_id"],
                    )
                    if source["canonical_ast_hash"] != entry["canonical_ast_sha256"]:
                        raise ValueError("frozen rebuild source has changed")
                if args.command == "enqueue":
                    await RebuildJobStore(factory).enqueue(
                        owner=entry["user_id"],
                        document=entry["document_id"],
                        version=entry["document_version_id"],
                        generation=args.target,
                        payload={
                            "config_hash": target.context_build.fingerprint("0" * 64),
                            "canonical_ast_sha256": entry["canonical_ast_sha256"],
                            "target_snapshot_id": snapshot.snapshot_id,
                        },
                    )
                else:
                    context = StagingContext(
                        user_id=entry["user_id"],
                        document_id=entry["document_id"],
                        document_version_id=entry["document_version_id"],
                        version_no=source["version_no"],
                        pipeline_version=source["pipeline_version"],
                        embedding_version=source["embedding_version"],
                        index_generation=args.source,
                    )
                    source_store = ElasticsearchChildIndexStore(
                        container.elasticsearch,
                        embedding_model=current.embedding_model,
                        lexical_analysis=current.lexical_analysis,
                        expected_representation_version=current.expected_representation_version,
                    )
                    await source_store.validate_search_target(args.source)
                    ref = artifacts.describe(source["manifest_path"])
                    if ref.sha256 != source["manifest_hash"]:
                        raise ValueError("active manifest hash changed")
                    parsed = VersionManifest.model_validate(artifacts.read_json(ref))
                    if parsed.schema_version == 2:
                        parents = validate_context_artifacts(
                            parsed,
                            artifacts,
                            (
                                entry["user_id"],
                                entry["document_id"],
                                entry["document_version_id"],
                            ),
                        )
                    else:
                        raw = artifacts.read_json(
                            artifacts.describe(
                                f"artifact://documents/{entry['user_id']}/{entry['document_id']}/{entry['document_version_id']}/chunks/{source['pipeline_version']}/parents.json"
                            )
                        )
                        parents = tuple(
                            ParentChunk.model_validate(value)
                            for value in raw["parents"]
                        )
                    actual = await source_store.child_identities(context)
                    if child_identity_digest(actual) != child_identity_digest(
                        (child.id, child.content_hash)
                        for parent in parents
                        for child in parent.children
                    ):
                        raise ValueError(
                            "active source identity/hash collection differs"
                        )
                    async with factory.begin() as db:
                        await current_source(
                            db,
                            entry["user_id"],
                            entry["document_id"],
                            entry["document_version_id"],
                        )
                        await register_build(
                            db,
                            context,
                            ContextArtifactRef(**ref.__dict__),
                            parsed,
                            parents=parents,
                            build_config_hash=container.runtime_snapshot.context_build_fingerprint,
                        )
        else:
            from docling_core.transforms.chunker.tokenizer.huggingface import (
                HuggingFaceTokenizer,
            )
            from agentic_rag.ingestion.indexer import IndexWriter
            from agentic_rag.ingestion.contextualization.builds import RebuildExecutor
            from agentic_rag.persistence.staging import SqlAlchemyParentStagingStore
            from scripts.run_ingestion_worker import QwenEmbeddingAdapter

            tokenizer = HuggingFaceTokenizer.from_pretrained(
                target.embedding_tokenizer_model, max_tokens=384, local_files_only=True
            )
            contextualizer, model_client = build_contextualizer(
                container, target, tokenizer
            )
            embedding = QwenEmbeddingAdapter(target)
            try:
                await validate_context_role(container, "rebuild")
                executor = RebuildExecutor(
                    factory=factory,
                    artifacts=artifacts,
                    writer=IndexWriter(
                        embedding=embedding,
                        embedding_model=target.embedding_model,
                        parent_store=SqlAlchemyParentStagingStore(factory),
                        child_store=aliases,
                        artifacts=artifacts,
                    ),
                    tokenizer=tokenizer,
                    contextualizer=contextualizer,
                    build=target.context_build,
                )
                for _ in range(args.max_jobs):
                    if not await executor.run_once(
                        f"{socket.gethostname()}:{current.context_instance_id}",
                        args.target,
                    ):
                        break
            finally:
                await embedding.close()
                if model_client:
                    await model_client.close()
    finally:
        await container.close()


def main(argv=None):
    args = parser().parse_args(argv)
    current, target, manifest, snapshot = load_plan(args)
    if not args.execute:
        print(
            json.dumps(
                {
                    "mode": "plan",
                    "command": args.command,
                    "source": args.source,
                    "target": args.target,
                    "snapshot": snapshot.snapshot_id,
                    "documents": len(manifest.get("documents", [])),
                    "readiness": "not_checked_offline",
                    "requires": [
                        "schema through 0012, applied before first arm",
                        "armed durable admission",
                        "complete current corpus builds",
                        "trusted signed maintenance receipt for install/release/rollback",
                        "stopped executor and settled ES writes proof for recover-execution",
                        "all configured deployment instance startup receipts",
                    ],
                },
                sort_keys=True,
            )
        )
        return 0
    asyncio.run(execute(args, current, target, manifest, snapshot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
