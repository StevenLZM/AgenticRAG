"""Frozen source corpus contracts, not retrieval-quality scores."""

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

from agentic_rag.safety.uploads import DefaultUploadSafetyScanner, UploadSafetyStatus
from evals.corpus import CorpusIntegrityError, build_corpus, load_source


def test_source_gold_has_24_resolvable_cases():
    source = load_source()
    assert len(source["documents"]) == 8
    assert len(source["cases"]) == len({q["case_id"] for q in source["cases"]}) == 24
    assert Counter(q["category"] for q in source["cases"]) == {
        "single-hop": 12, "multi-hop": 8, "unanswerable": 4,
    }
    facts = {f["fact_id"] for d in source["documents"] for f in d["facts"]}
    for case in source["cases"]:
        assert set(case["source_fact_ids"]) <= facts
        assert bool(case["source_fact_ids"]) == case["answerable"]
        if case["category"] == "multi-hop":
            assert len(case["source_fact_ids"]) >= 2


def test_build_creates_eight_hash_verified_safe_uploads(tmp_path: Path):
    manifest_path = build_corpus(tmp_path / "corpus")
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest["documents"]) == 8
    assert Counter(Path(d["filename"]).suffix for d in manifest["documents"]) == {
        ".pdf": 2, ".txt": 4, ".xlsx": 2,
    }
    scanner = DefaultUploadSafetyScanner()
    for document in manifest["documents"]:
        data = (manifest_path.parent / document["filename"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == document["sha256"]
        decision = scanner.scan(document["filename"], document["mime_type"], data)
        assert decision.status == UploadSafetyStatus.ACCEPTED, decision.reasons


def test_build_reuses_identical_manifest_without_rewriting(tmp_path: Path):
    path = build_corpus(tmp_path / "corpus")
    before = path.stat().st_mtime_ns
    content = path.read_bytes()
    assert build_corpus(path.parent) == path
    assert path.read_bytes() == content
    assert path.stat().st_mtime_ns == before


def test_build_refuses_tampered_output_without_overwrite(tmp_path: Path):
    path = build_corpus(tmp_path / "corpus")
    document = json.loads(path.read_text())["documents"][0]
    target = path.parent / document["filename"]
    target.write_bytes(b"tampered")
    with pytest.raises(CorpusIntegrityError):
        build_corpus(path.parent)
    assert target.read_bytes() == b"tampered"


def test_build_refuses_changed_manifest(tmp_path: Path):
    path = build_corpus(tmp_path / "corpus")
    manifest = json.loads(path.read_text())
    manifest["cases"][0]["reference_answer"] = "changed gold"
    path.write_text(json.dumps(manifest))
    with pytest.raises(CorpusIntegrityError):
        build_corpus(path.parent)


def test_interrupted_copy_does_not_publish_partial_asset(tmp_path: Path, monkeypatch):
    output = tmp_path / "corpus"
    original = shutil.copyfile

    def partial_copy(source, target):
        Path(target).write_bytes(b"partial")
        raise OSError("interrupted copy")

    monkeypatch.setattr(shutil, "copyfile", partial_copy)
    with pytest.raises(OSError, match="interrupted"):
        build_corpus(output)
    assert not list(output.glob("*.pdf"))
    monkeypatch.setattr(shutil, "copyfile", original)
    assert build_corpus(output).exists()
