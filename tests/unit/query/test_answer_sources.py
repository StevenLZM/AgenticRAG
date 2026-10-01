import importlib.util
import json

import pytest

from agentic_rag.query.evidence_builder import (
    EvidenceItem,
    EvidenceManifestEntry,
    PackedEvidence,
)
from agentic_rag.query.public_answer import PublicAnswer, PublicAnswerSegment


def payload(count=3, text="历史片段", headings=("章节",)):
    items = tuple(
        EvidenceItem(
            evidence_id=f"e{i}",
            parent_id=f"p{i}",
            document_id=f"d{i}",
            document_version_id=f"v{i}",
            content=text,
            ast_locator="invalid locator",
            covered_target_ids=(),
            heading_path=headings,
        )
        for i in range(1, count + 1)
    )
    manifest = {
        item.evidence_id: EvidenceManifestEntry(
            **item.model_dump(exclude={"content", "covered_target_ids"})
        )
        for item in items
    }
    return PackedEvidence(
        items=items,
        manifest=manifest,
        rendered_context="PRIVATE",
        token_count=1,
        index_generation="i",
    )


def answer(ids=("e2", "e1", "e2")):
    return PublicAnswer(
        audited=True,
        route="research",
        segments=tuple(
            PublicAnswerSegment(kind="content", text="答", evidence_ids=ids[i : i + 32])
            for i in range(0, len(ids), 32)
        ),
    )


def api():
    assert importlib.util.find_spec("agentic_rag.query.answer_sources"), (
        "sources projection missing"
    )
    from agentic_rag.query import answer_sources

    return answer_sources


def test_only_cited_packed_content_is_snapshotted():
    snapshot = api().build_answer_sources(
        answer(), payload(), run_id="r", snapshot_id="s"
    )
    assert [item.evidence_id for item in snapshot.items] == ["e2", "e1"]
    assert all(
        item.excerpt == "历史片段" and item.page_from is None for item in snapshot.items
    )
    assert (
        "PRIVATE" not in snapshot.model_dump_json()
        and "locator" not in snapshot.model_dump_json()
    )


def test_multibyte_snapshot_is_bounded_and_marks_omissions():
    snapshot = api().build_answer_sources(
        answer(tuple(f"e{i}" for i in range(1, 71))),
        payload(70, '😀\n"\\' * 1500, tuple("😀" * 40 for _ in range(20))),
        run_id="r",
        snapshot_id="s",
    )
    assert len(snapshot.items) == 64 and snapshot.omitted_source_count == 6
    assert all(
        len(item.excerpt) <= 2000
        and len(item.heading_path) <= 16
        and sum(map(len, item.heading_path)) <= 128
        and item.heading_truncated
        for item in snapshot.items
    )
    assert any(item.excerpt_omitted for item in snapshot.items)
    assert len(snapshot.model_dump_json().encode()) <= 256 * 1024
    assert len(json.dumps(snapshot.model_dump(mode="json")).encode()) <= 256 * 1024


@pytest.mark.parametrize("damage", ["missing", "manifest", "id", "duplicate"])
def test_unsafe_or_inconsistent_provenance_is_rejected(damage):
    packed = payload()
    if damage == "missing":
        packed = packed.model_copy(update={"manifest": {}})
    elif damage == "manifest":
        packed = packed.model_copy(
            update={
                "manifest": {
                    **packed.manifest,
                    "e2": packed.manifest["e2"].model_copy(
                        update={"heading_path": ("different",)}
                    ),
                }
            }
        )
    elif damage == "id":
        bad = packed.items[1].model_copy(update={"document_id": "../private"})
        packed = packed.model_copy(
            update={
                "items": (packed.items[0], bad, packed.items[2]),
                "manifest": {
                    **packed.manifest,
                    "e2": packed.manifest["e2"].model_copy(
                        update={"document_id": "../private"}
                    ),
                },
            }
        )
    else:
        packed = packed.model_copy(update={"items": (*packed.items, packed.items[1])})
    with pytest.raises(api().InvalidAnswerSources):
        api().build_answer_sources(answer(), packed, run_id="r", snapshot_id="s")


def test_non_document_answers_have_no_snapshot():
    for public in [
        PublicAnswer(
            route="chat", segments=(PublicAnswerSegment(kind="content", text="你好"),)
        ),
        PublicAnswer(status="cannot_answer"),
        PublicAnswer(
            audited=True, segments=(PublicAnswerSegment(kind="content", text="答"),)
        ),
    ]:
        assert (
            api().build_answer_sources(public, payload(), run_id="r", snapshot_id="s")
            is None
        )


@pytest.mark.parametrize("padding", range(8))
def test_byte_boundary_includes_final_truncation_flags(padding):
    snapshot = api().build_answer_sources(
        answer(tuple(f"e{i}" for i in range(1, 65))),
        payload(64, "\x00" * 2100),
        run_id="r",
        snapshot_id="s" * (padding + 1),
    )
    assert (
        snapshot is not None
        and len(json.dumps(snapshot.model_dump(mode="json")).encode()) <= 256 * 1024
    )
