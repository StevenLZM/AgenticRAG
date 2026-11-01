"""Server-resolved scope, original source selection and whole-block caps."""

from agentic_rag.ingestion.contextualization.map import DocumentContextMap
from agentic_rag.ingestion.contextualization.assembly import ContextAssembler
from agentic_rag.ingestion.contextualization.sources import SourceCatalog
from tests.unit.ingestion.test_context_service import setup
from tests.unit.ingestion.test_context_sources import _child
from agentic_rag.ingestion.chunker import ParentChunk


def isolated_target(parent, index, text):
    child = _child(text)
    span = child.ast_locator.spans[0].model_copy(
        update={
            "block_id": f"block-{index}",
            "canonical_path": f"#/text_blocks/{index}",
            "page_from": index + 1,
            "page_to": index + 1,
        }
    )
    locator = child.ast_locator.model_copy(update={"spans": (span,)})
    child = child.model_copy(
        update={"ast_locator": locator, "page_from": index + 1, "page_to": index + 1}
    )
    parent = ParentChunk(
        **{
            **child.model_dump(
                exclude={"parent_id", "parent_ordinal", "search_representation"}
            ),
            "id": child.parent_id,
            "children": (child,),
        }
    )
    return parent, child


def test_document_scope_can_cover_child_without_sending_whole_document(tmp_path):
    texts = (
        "本制度适用于星河公司。",
        "irrelevant appendix",
        "another appendix",
        "上海标准500一晚。",
        "further appendix",
    )
    artifacts, ast, ref, parent, claim = setup(tmp_path, texts)
    catalog = SourceCatalog.from_ast(ast, ref.sha256)
    mapping = DocumentContextMap.build(ast, catalog)
    parent, child = isolated_target(parent, 3, texts[3])
    input, regions = ContextAssembler(len).assemble(child, parent, mapping)
    assert not input.context_budget_exhausted
    assert "document" in input.scopes
    assert catalog.covers_child(regions["document"].source_refs, child)
    assert len(input.sources) < len(texts)


def test_selection_caps_sources_without_truncating_condition(tmp_path):
    texts = tuple(f"适用于group{i}，不含外包人员。" for i in range(20))
    artifacts, ast, ref, parent, claim = setup(tmp_path, texts)
    mapping = DocumentContextMap.build(ast, SourceCatalog.from_ast(ast, ref.sha256))
    parent, child = isolated_target(parent, 0, texts[0])
    input, _ = ContextAssembler(lambda text: len(text) // 4).assemble(
        child, parent, mapping
    )
    assert input.context_budget_exhausted and len(input.sources) <= 16
    assert all(source.text in texts for source in input.sources)


def test_explicit_reference_two_hops_and_table_scope(tmp_path):
    from tests.unit.ingestion.test_chunker import _canonical, _block

    texts = [
        "按第三章执行",
        "adjacent",
        "unused",
        "第三章 员工",
        "见第六章",
        "unused",
        "第六章 定义",
        "见第九章",
        "unused",
        "第九章 附表",
        "| 人员 | 标准 |\n| 普通员工 | 500 |",
    ]
    artifacts, _, _, parent, _ = setup(tmp_path, tuple(texts))
    ast = _canonical(
        tuple(
            _block(
                i,
                text=text,
                kind="heading"
                if i in (3, 6, 9)
                else "table"
                if i == 10
                else "paragraph",
                heading_level=1 if i in (3, 6, 9) else None,
            )
            for i, text in enumerate(texts)
        )
    )
    ref = artifacts.put_json("map.json", ast.model_dump(mode="json"))
    catalog = SourceCatalog.from_ast(ast, ref.sha256)
    mapping = DocumentContextMap.build(ast, catalog)
    # Scope candidates are separately tested; isolate the reference traversal.
    mapping = mapping.model_copy(update={"scope_candidates": ()})
    parent, child = isolated_target(parent, 0, texts[0])
    input, _ = ContextAssembler(lambda text: len(text) // 4).assemble(
        child, parent, mapping, max_hops=2
    )
    chosen = {s.source_id for s in input.sources}
    assert catalog.source_ids[3] in chosen and catalog.source_ids[6] in chosen
    assert catalog.source_ids[9] not in chosen
    assert mapping.regions[f"table:{catalog.source_ids[10]}"].kind == "table"
