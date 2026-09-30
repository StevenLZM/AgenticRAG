"""Copy a frozen synthetic source corpus; never derive gold from retrieval."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
import shutil
import tempfile

from evals.report import atomic_write_text


SOURCE_ROOT = Path(__file__).parent / "datasets" / "real_corpus"
MIME_TYPES = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


class CorpusIntegrityError(ValueError):
    """A frozen input/output was changed, incomplete, or ambiguous."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_source() -> dict:
    source = json.loads((SOURCE_ROOT / "source.json").read_text(encoding="utf-8"))
    documents, cases = source["documents"], source["cases"]
    if len(documents) != 8 or len({d["document_key"] for d in documents}) != 8:
        raise CorpusIntegrityError("expected 8 distinct source documents")
    filenames = [d["filename"] for d in documents]
    if len(set(filenames)) != 8 or any(
        Path(name).name != name or "\\" in name or Path(name).suffix not in MIME_TYPES
        for name in filenames
    ):
        raise CorpusIntegrityError("source filenames must be unique supported basenames")
    facts = [f for d in documents for f in d["facts"]]
    ids = {f["fact_id"] for f in facts}
    if len(ids) != len(facts):
        raise CorpusIntegrityError("ambiguous source fact IDs")
    for fact in facts:
        if not fact["text"].strip() or not fact["anchor_terms"]:
            raise CorpusIntegrityError("fact requires source text and anchors")
        if any(not term or term not in fact["text"] for term in fact["anchor_terms"]):
            raise CorpusIntegrityError("source anchor is not present in declared fact")
    if len(cases) != 24 or len({c["case_id"] for c in cases}) != 24:
        raise CorpusIntegrityError("expected 24 distinct source cases")
    if Counter(c["category"] for c in cases) != {
        "single-hop": 12, "multi-hop": 8, "unanswerable": 4,
    }:
        raise CorpusIntegrityError("invalid single/multi/unanswerable case counts")
    for case in cases:
        refs = case["source_fact_ids"]
        if type(case["answerable"]) is not bool or bool(refs) != case["answerable"]:
            raise CorpusIntegrityError("answerability and gold sources disagree")
        if not set(refs) <= ids or len(set(refs)) != len(refs):
            raise CorpusIntegrityError("unresolved or duplicate gold source")
        if not case["question"].strip() or not case["reference_answer"].strip():
            raise CorpusIntegrityError("question and reference answer are required")
        if case["category"] == "multi-hop" and len(refs) < 2:
            raise CorpusIntegrityError("multihop needs at least two source facts")
    return source


def build_corpus(output_dir: Path) -> Path:
    """Create/resume a byte-identical corpus, refusing changed gold or artifacts.

    No providers, vector stores, parser outputs, or retrieval results are inputs.
    Existing unrelated files are left untouched. A manifest is written last so
    interrupted preparation can safely reuse matching already-copied files.
    """
    source = load_source()
    assets = SOURCE_ROOT / "assets"
    lock = json.loads((SOURCE_ROOT / "assets.sha256.json").read_text(encoding="utf-8"))
    documents = []
    for document in source["documents"]:
        filename = document["filename"]
        data = (assets / filename).read_bytes()
        digest = _sha256(data)
        if not data or lock.get(filename) != digest:
            raise CorpusIntegrityError(f"frozen asset hash mismatch: {filename}")
        documents.append({
            **document, "sha256": digest, "size_bytes": len(data),
            "mime_type": MIME_TYPES[Path(filename).suffix],
        })
    manifest = {
        "schema_version": source["schema_version"], "corpus_id": source["corpus_id"],
        "source_sha256": _sha256((SOURCE_ROOT / "source.json").read_bytes()),
        "gold_provenance": "source-authored-before-retrieval",
        "documents": documents, "cases": source["cases"],
    }
    encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    output_dir = Path(output_dir)
    path = output_dir / "manifest.json"
    if path.is_symlink():
        raise CorpusIntegrityError("manifest must not be a symlink")
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        raise CorpusIntegrityError("existing manifest changed; select a new corpus output")
    # Check all pre-existing targets before writing any new file.
    for document in documents:
        target = output_dir / document["filename"]
        if target.is_symlink() or (
            target.exists() and _sha256(target.read_bytes()) != document["sha256"]
        ):
            raise CorpusIntegrityError(f"existing corpus file changed: {target.name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for document in documents:
        target = output_dir / document["filename"]
        if not target.exists():
            _copy_asset_atomic(assets / document["filename"], target)
    if not path.exists():
        atomic_write_text(path, encoded)
    return path


def _copy_asset_atomic(source: Path, target: Path) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".corpus-", suffix=".tmp", dir=target.parent)
    os.close(fd)
    try:
        shutil.copyfile(source, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    print(build_corpus(parser.parse_args().output_dir))
