"""Read-only audit of every normal production AST, chunk artifact and SQL/ES ID.

Recomputes chunks using the *same* ChunkingPipeline/HybridChunker/tokenizer used
by run_ingestion_worker. No uploads, embeddings, index writes or new source files.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

from agentic_rag.config import Settings
from agentic_rag.ingestion.assembler import CanonicalAst
from agentic_rag.ingestion.chunker import ChunkingPipeline, ParentChunk, CHILD_MAX_TOKENS, resolve_ast_locator
from agentic_rag.persistence.artifacts import LocalArtifactStore
from evals.formal_pipeline_audit import require_corpus_complete, require_current_document, require_published_fields
from evals.report import atomic_write_text


def run(root, expected_active):
    snapshot = json.loads((root / "snapshot.json").read_text())
    settings = Settings()
    artifacts = LocalArtifactStore(settings.artifact_root)
    require_corpus_complete(snapshot["status_counts"], expected_active=expected_active,
                            user_id=settings.default_user_id, document_count=len(snapshot["documents"]))
    if snapshot["scope"]["user_id"] != settings.default_user_id:
        raise ValueError("snapshot user scope mismatch")
    tokenizer = HuggingFaceTokenizer.from_pretrained(settings.embedding_tokenizer_model, max_tokens=CHILD_MAX_TOKENS)
    pipeline = ChunkingPipeline(tokenizer)
    sql_parents = {p["id"]: p for p in snapshot["parents"]}
    es_children = {c["child_id"]: c for c in snapshot["children"]}
    verified, pids, cids, token_max = [], set(), set(), 0
    for doc in snapshot["documents"]:
        require_current_document(doc)
        source = Path(doc["source_path"])
        if hashlib.sha256(source.read_bytes()).hexdigest() != doc["content_hash"]:
            raise ValueError("original hash drift")
        canonical_ref = artifacts.describe(doc["canonical_ast_path"])
        if canonical_ref.sha256 != doc["canonical_ast_hash"]:
            raise ValueError("canonical AST hash drift")
        ast = CanonicalAst.model_validate(artifacts.read_json(canonical_ref))
        if (ast.envelope.document_id != doc["document_id"] or ast.envelope.document_version_id != doc["document_version_id"]
                or ast.envelope.content_hash != doc["content_hash"] or ast.envelope.user_id != settings.default_user_id
                or ast.envelope.pipeline_version != settings.ingestion_pipeline_version):
            raise ValueError("canonical envelope scope mismatch")
        chunk_path = source.parent.parent / "chunks" / settings.ingestion_pipeline_version / "parents.json"
        stored = [ParentChunk.model_validate(p) for p in json.loads(chunk_path.read_text())["parents"]]
        rebuilt = pipeline.build(ast)
        if [p.model_dump(mode="json") for p in rebuilt] != [p.model_dump(mode="json") for p in stored]:
            raise ValueError("stored chunks differ from normal current ChunkingPipeline: " + doc["document_id"])
        manifest = artifacts.describe(doc["manifest_path"])
        if manifest.sha256 != doc["manifest_hash"]:
            raise ValueError("publication manifest hash mismatch")
        payload = artifacts.read_json(manifest)
        child_count = sum(len(p.children) for p in stored)
        if (len(stored), child_count) != (doc["parent_count"], doc["child_count"]) or (
                payload["parent_count"], payload["child_count"]) != (len(stored), child_count):
            raise ValueError("published counts mismatch")
        for parent in stored:
            row = sql_parents.get(parent.id)
            if row is None:
                raise ValueError("SQL parent missing")
            require_published_fields(row, parent.model_dump(mode="json"), (
                "user_id", "document_id", "document_version_id", "ordinal", "heading_path",
                "content_type", "content", "page_from", "page_to", "ast_locator", "content_hash"))
            resolve_ast_locator(ast, parent.ast_locator)
            pids.add(parent.id)
            for child in parent.children:
                row = es_children.get(child.id)
                if row is None:
                    raise ValueError("ES child missing")
                require_published_fields(row, child.model_dump(mode="json"), (
                    "user_id", "document_id", "document_version_id", "parent_id", "parent_ordinal",
                    "content", "contextualized_content", "token_count", "content_type",
                    "heading_path", "heading_ast_locators", "ast_locator", "page_from", "page_to"))
                if row["pipeline_version"] != settings.ingestion_pipeline_version or row["index_generation"] != settings.index_generation:
                    raise ValueError("ES generation/pipeline mismatch")
                if tokenizer.count_tokens(child.contextualized_content) != child.token_count:
                    raise ValueError("production token count mismatch")
                resolve_ast_locator(ast, child.ast_locator)
                token_max = max(token_max, child.token_count)
                cids.add(child.id)
        verified.append({"document_id": doc["document_id"], "document_version_id": doc["document_version_id"],
                         "parent_count": len(stored), "child_count": child_count, "current_chunker_identical": True})
    if pids != set(sql_parents) or cids != set(es_children):
        raise ValueError("extra or missing active SQL/ES chunks in corpus")
    report = {"snapshot_id": snapshot["snapshot_id"], "documents": len(verified),
              "parents": len(pids), "children": len(cids), "max_child_tokens": token_max,
              "all_current_chunks_identical": True, "special_eval_chunker_used": False,
              "status_counts": snapshot["status_counts"], "formats": dict(Counter(Path(d["source_path"]).suffix for d in snapshot["documents"])),
              "documents_verified": verified}
    target = root / "production-pipeline-verification.json"
    if target.exists():
        raise ValueError("verification already exists; do not overwrite frozen evidence")
    atomic_write_text(target, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "documents_verified"}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-active", type=int, default=1003)
    args = parser.parse_args()
    run(args.output_dir, args.expected_active)
