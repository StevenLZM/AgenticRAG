"""Author only this approved synthetic batch; no network or database writes."""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from evals.bulk_corpus import ROOT, save_source
from evals.report import atomic_write_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    save_source(args.root)
    source = json.loads((args.root / "source.json").read_text())
    target = args.root / "documents"
    target.mkdir(parents=True, exist_ok=True)
    pdfmetrics.registerFont(TTFont("CorpusCN", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"))
    body = ParagraphStyle("body", fontName="CorpusCN", fontSize=10, leading=17,
                          spaceAfter=9, wordWrap="CJK")
    title = ParagraphStyle("title", parent=body, fontSize=16, leading=24, spaceAfter=15)
    heading = ParagraphStyle("heading", parent=body, fontSize=12, leading=19,
                             textColor=colors.HexColor("#243B53"), keepWithNext=True)

    def footer(canvas, pdf):
        canvas.setFont("CorpusCN", 8)
        canvas.drawString(44, 25, "合成测试资料 - 不作为真实业务依据")
        canvas.drawRightString(A4[0] - 44, 25, str(pdf.page))

    for doc in source["documents"]:
        path = target / doc["filename"]
        if doc["format"] == "xlsx" or path.exists():
            continue
        intro = f'{doc["notice"]}\n文档编号：{doc["document_key"]}'
        references = "关联文档编号：" + "、".join(doc["references"])
        if doc["format"] == "txt":
            atomic_write_text(path, doc["title"] + "\n\n" + intro + "\n\n" +
                              "\n\n".join(h + "\n" + text for h, text in doc["sections"]) +
                              "\n\n" + references + "\n")
        else:
            story = [Paragraph(escape(doc["title"]), title),
                     Paragraph(escape(intro).replace("\n", "<br/>"), body), Spacer(1, 8)]
            for h, text in doc["sections"]:
                story.extend([Paragraph(escape(h), heading), Paragraph(escape(text), body)])
            story.append(Paragraph(escape(references), body))
            temporary = path.with_suffix(".pdf.tmp")
            SimpleDocTemplate(str(temporary), pagesize=A4, leftMargin=44, rightMargin=44,
                              topMargin=40, bottomMargin=42, title=doc["title"],
                              author="Synthetic RAG corpus", invariant=1).build(
                                  story, onFirstPage=footer, onLaterPages=footer)
            temporary.replace(path)
    print("PDF/TXT ready", len(list(target.glob("*.pdf"))), len(list(target.glob("*.txt"))), flush=True)


if __name__ == "__main__":
    main()
