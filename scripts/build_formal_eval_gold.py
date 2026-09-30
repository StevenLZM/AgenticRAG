"""Generate original-grounded v2 gold from a frozen formal corpus snapshot.

Run with the bundled document Python (pypdf/openpyxl); no API/model calls.
Gold truth comes from original files, SQL/ES are only used to bind qrels.
"""
import hashlib
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from pypdf import PdfReader
from openpyxl import load_workbook

from evals.formal_gold import bind_anchor, check_anchor, check_cached_number, split_for
from evals.report import atomic_write_text

ROOT = Path("var/artifacts/evals/formal-corpus-gold-v2-20260930")
CORPUS = Path("var/artifacts/corpora/synthetic-business-1000-20260927-v1")


def read_original(doc):
    path = Path(doc["source_path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != doc["content_hash"]:
        raise ValueError("source hash drift")
    if path.suffix == ".pdf":
        pages = [p.extract_text() or "" for p in PdfReader(path).pages]
        return "\n".join(pages), {"pages": len(pages)}
    if path.suffix == ".txt":
        return path.read_text(), {}
    workbook = load_workbook(path, data_only=True, read_only=True)
    rows = {s.title: [list(r) for r in s.values] for s in workbook}
    workbook.close()
    return "\n".join(" ".join(str(v) for v in row if v is not None)
                     for sheet in rows.values() for row in sheet), {"sheets": rows}


def main(root=ROOT):
    snapshot = json.loads((root / "snapshot.json").read_text())
    authored = json.loads((CORPUS / "source.json").read_text())
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    by_filename = {d["filename"]: d for d in authored["documents"]}
    hashes = {d["filename"]: d["sha256"] for d in manifest["documents"]}
    parents, children = defaultdict(list), defaultdict(list)
    for p in snapshot["parents"]:
        parents[p["document_version_id"]].append(p)
    for c in snapshot["children"]:
        children[c["document_version_id"]].append(c)
    cases, verified, originals, docs_by_key = [], [], {}, {}

    def add(doc, key, suffix, question, answer, claims, *, kind="single_hop", fields=None,
            additional=None, derivation=None, answerable=True):
        sources = [doc] + (additional or [])
        groups = []
        for i, (claim, anchors, source_doc) in enumerate(claims):
            for anchor in anchors:
                check_anchor(originals[source_doc["document_version_id"]][0], anchor)
            group = bind_anchor(source_doc["document_version_id"], anchors,
                                parents[source_doc["document_version_id"]], children[source_doc["document_version_id"]])
            group.update({"fact_id": f"{key}:{suffix}:F{i+1}", "claim": claim,
                          "document_version_id": source_doc["document_version_id"]})
            groups.append(group)
        cases.append({"schema_version": 2, "case_id": f"{key}:{suffix}",
            "snapshot_id": snapshot["snapshot_id"], "user_id": snapshot["scope"]["user_id"],
            "split": split_for(key), "category": kind, "question": question,
            "answerable": answerable, "reference_answer": answer,
            "expected_route": "research" if kind == "multi_hop" else "fast_rag_or_research",
            "source_documents": [{"document_id": d["document_id"],
                "document_version_id": d["document_version_id"], "sha256": d["content_hash"],
                "filename": d["filename"], "searchable_at_snapshot": d["searchable"]} for d in sources],
            "required_evidence_groups_all_of": groups, "critical_fields": fields or [],
            "derivation": derivation, "review_status": "original_verified_human_review_pending"})

    for doc in snapshot["documents"]:
        version = doc["document_version_id"]
        text, structure = read_original(doc)
        originals[version] = (text, structure)
        source = by_filename.get(doc["filename"])
        if source:
            if hashes[doc["filename"]] != doc["content_hash"]:
                raise ValueError("authored-source manifest mismatch")
            key = source["document_key"]
            docs_by_key[key] = doc
            for n, (heading, body) in enumerate(source["sections"], 1):
                check_anchor(text, body)
                sentences = [s + "。" for s in body.split("。") if s]
                add(doc, key, f"section-{n}", f'{source["title"]}中，“{heading}”有哪些规定或记录？', body,
                    [(s, [s], doc) for s in sentences], kind="section_comprehension")
            if source["category"] == "policy":
                body = source["sections"][1][1]
                amount = int(re.search(r"每人每晚(\d+)元", body)[1])
                add(doc, key, "lodging", f'{source["facts"]["company"]}{source["year"]}年{source["facts"]["city"]}住宿报销上限是多少？单位是什么？',
                    f"每人每晚{amount}元人民币，含税。", [(body.split("。")[0] + "。", [body.split("。")[0]], doc)],
                    fields=[{"name": "lodging_limit", "value": amount, "unit": "CNY/person/night", "tolerance": 0,
                             "conditions": {"company": source["facts"]["company"], "year": source["year"], "city": source["facts"]["city"]}}])
            if source["category"] in {"operations", "inventory"}:
                sheet = structure["sheets"]["台账"]
                data = sheet[5:9] if source["category"] == "operations" else sheet[5:11]
                # The original's cached results must match independently recomputed values.
                for i, row in enumerate(data):
                    expected = 1 - row[2] / row[1] if source["category"] == "operations" else row[1] + row[2] - row[3]
                    check_cached_number(row[4], expected)
                if source["category"] == "operations":
                    monitored, interrupted = sum(r[1] for r in data), sum(r[2] for r in data)
                    value = (1 - interrupted / monitored) * 100
                    claim_rows = [(f"{r[0]}：监测{r[1]}分钟，中断{r[2]}分钟。", [str(r[0]), str(r[1]), str(r[2])], doc) for r in data]
                    add(doc, key, "annual-availability", f'{source["title"]}按全年加总分钟计算的年度可用率是多少？保留四位小数，说明计算口径。',
                        f"监测共{monitored}分钟，中断共{interrupted}分钟。年度可用率=(1-{interrupted}/{monitored})×100%={value:.4f}%，不能直接平均季度百分比。",
                        claim_rows, kind="table_calculation", fields=[{"name": "annual_availability", "value": value, "unit": "percent", "tolerance": 0.00005}],
                        derivation={"operation": "(1-sum(C6:C9)/sum(B6:B9))*100", "sheet": "台账", "source_cells": "B6:C9", "inputs": data})
                else:
                    row = data[0]
                    value = row[1] + row[2] - row[3]
                    add(doc, key, "ending-inventory", f'{source["title"]}中设备{row[0]}的期末库存有多少台？说明计算。',
                        f"期初{row[1]}台+采购{row[2]}台-领用{row[3]}台={value}台。",
                        [(f"设备{row[0]}期初{row[1]}台、采购{row[2]}台、领用{row[3]}台。", [str(v) for v in row[:4]], doc)], kind="table_calculation",
                        fields=[{"name": "ending_inventory", "value": value, "unit": "device", "tolerance": 0}],
                        derivation={"operation": "B6+C6-D6", "sheet": "台账", "source_cells": "A6:D6", "inputs": row})
            if source["category"] == "incident":
                anchor = "本事件未公布精确首响时间，因此不能判断首响是否违约。"
                add(doc, key, "unknown-response", f'{source["facts"]["company"]}{source["year"]}年5月12日故障的实际首次响应用了几分钟？',
                    "无法确定：故障复盘未公布精确首响时间，SLA目标不能当作实际发生时间。",
                    [(anchor, [anchor], doc)], kind="unanswerable", answerable=False)
            verified.append({"document_version_id": version, "source_hash_verified": True,
                             "authored_sections_verified": 4, "format": source["format"]})
        else:
            # Avoid carrying phone, identity number, email or address into gold.
            key = "legacy-" + version
            if "简历" in doc["filename"]:
                anchor = "计算机科学与技术专业"
                add(doc, key, "major", "刘泽明简历中，本科就读学校和专业是什么？", "哈尔滨工业大学，计算机科学与技术专业。",
                    [("本科就读哈尔滨工业大学", ["本科", "哈尔滨工业大学"], doc), ("本科专业为计算机科学与技术", [anchor], doc)])
            elif doc["filename"] == "测试.pdf":
                add(doc, key, "score", "测试.pdf中曹士杰的分数是多少？", "90分。",
                    [("曹士杰的分数是90分", ["曹士杰", "90"], doc)], fields=[{"name": "score", "value": 90, "unit": "point", "tolerance": 0}])
            else:
                raise ValueError("unhandled original document: " + doc["document_id"])
            verified.append({"document_version_id": version, "source_hash_verified": True, "format": "pdf"})

    for key, doc in sorted(docs_by_key.items()):
        if "-2026-policy" not in key:
            continue
        old = docs_by_key.get(key.replace("2026", "2025"))
        if not old:
            continue
        new_text, old_text = originals[doc["document_version_id"]][0], originals[old["document_version_id"]][0]
        n = int(re.search(r"每人每晚(\d+)元", new_text)[1])
        o = int(re.search(r"每人每晚(\d+)元", old_text)[1])
        org = by_filename[doc["filename"]]["facts"]["company"]
        add(doc, key, "year-comparison", f"{org}2026年相较2025年的每人每晚住宿报销上限增加多少元？列出两年的标准。",
            f"2025年{o}元，2026年{n}元，每人每晚增加{n-o}元人民币。",
            [(f"2025年每人每晚{o}元", [f"每人每晚{o}元"], old), (f"2026年每人每晚{n}元", [f"每人每晚{n}元"], doc)],
            kind="multi_hop", additional=[old], fields=[{"name": "increase", "value": n-o, "unit": "CNY/person/night", "tolerance": 0}],
            derivation={"operation": "2026_limit-2025_limit", "inputs": {"2025": o, "2026": n}})

    gold_path = root / "gold.v2.2.jsonl"
    if gold_path.exists():
        raise ValueError("gold exists; do not overwrite")
    atomic_write_text(gold_path, "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases))
    represented = {d["document_version_id"] for c in cases for d in c["source_documents"]}
    assert represented == {d["document_version_id"] for d in snapshot["documents"]}
    report = {"snapshot_id": snapshot["snapshot_id"], "documents": len(represented), "cases": len(cases),
        "categories": dict(Counter(c["category"] for c in cases)), "splits": dict(Counter(c["split"] for c in cases)),
        "mapping_status": dict(Counter(g["mapping_status"] for c in cases for g in c["required_evidence_groups_all_of"])),
        "non_searchable_documents": [d["document_version_id"] for d in snapshot["documents"] if not d["searchable"]],
        "original_verification": verified, "gold_sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
        "review_status": "programmatic_original_verification_passed; human semantic review pending",
        "evaluation_executed": False}
    atomic_write_text(root / "gold-validation.v2.2.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "original_verification"}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    main(parser.parse_args().output_dir)
