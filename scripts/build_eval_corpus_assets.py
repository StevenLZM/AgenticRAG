"""Author PDF/TXT assets with reportlab; normal eval runs use frozen assets.

Run with the bundled artifact Python. XLSX assets are authored separately with
build_eval_corpus_workbooks.mjs. Never regenerate assets in an active eval run.
"""

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path, required=True, help="embeddable Chinese TTF/TTC")
    parser.add_argument("--format", choices=("all", "pdf", "txt"), default="all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "evals/datasets/real_corpus"
    assets = root / "assets"
    assets.mkdir(exist_ok=True)
    source = json.loads((root / "source.json").read_text())
    pdfmetrics.registerFont(TTFont("EvalChinese", str(args.font), subfontIndex=0))
    body = ParagraphStyle("body", fontName="EvalChinese", fontSize=11,
                          leading=19, spaceAfter=12, wordWrap="CJK", alignment=TA_LEFT)
    title = ParagraphStyle("title", parent=body, fontSize=19, leading=28, spaceAfter=18)
    heading = ParagraphStyle("heading", parent=body, fontSize=13, leading=21,
                             spaceAfter=5, textColor=colors.HexColor("#243B53"))
    for document in source["documents"]:
        target = assets / document["filename"]
        if target.suffix not in {".pdf", ".txt"}:
            continue
        if args.format != "all" and target.suffix != f".{args.format}":
            continue
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite frozen source asset: {target}")
        if target.suffix == ".txt":
            content = document["title"] + "\n\n" + "\n\n".join(
                f'{f["heading"]}\n{f["text"]}' for f in document["facts"]
            ) + "\n"
            target.write_text(content, encoding="utf-8")
        elif target.suffix == ".pdf":
            story = [Paragraph(escape(document["title"]), title)]
            for fact in document["facts"]:
                story += [Paragraph(escape(fact["heading"]), heading),
                          Paragraph(escape(fact["text"]), body)]
            story.append(Spacer(1, 10))
            pdf = SimpleDocTemplate(str(target), pagesize=A4, leftMargin=48,
                                    rightMargin=48, topMargin=42, bottomMargin=42,
                                    title=document["title"], author="Synthetic RAG Corpus",
                                    invariant=1)
            pdf.build(story)
        else:
            continue
        print(target.name, target.stat().st_size)


if __name__ == "__main__":
    main()
