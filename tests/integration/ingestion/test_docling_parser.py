"""Docling boundary tests for every supported source type.

These tests are explicit integration tests because Docling may initialize local OCR
and layout models. They skip when the declared runtime dependency is not installed.
"""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from agentic_rag.ingestion.assembler import DocumentEnvelope
from agentic_rag.ingestion.parser import DocumentParser
from agentic_rag.persistence.artifacts import LocalArtifactStore


pytestmark = pytest.mark.integration


def _minimal_pdf(text: str | None) -> bytes:
    stream = (
        b"BT /F1 12 Tf 72 720 Td (" + text.encode("ascii") + b") Tj ET"
        if text is not None
        else b""
    )
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    payload = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(payload))
        payload.extend(f"{number} 0 obj\n".encode())
        payload.extend(obj)
        payload.extend(b"\nendobj\n")
    xref = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode())
    payload.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(payload)


def _minimal_scanned_pdf() -> bytes:
    draw = b"q 100 0 0 100 72 600 cm /Im0 Do Q"
    pixels = b"\xaa\x55\xaa\x55\xaa\x55\xaa\x55"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(draw)).encode() + b" >>\nstream\n" + draw + b"\nendstream",
        b"<< /Type /XObject /Subtype /Image /Width 8 /Height 8 "
        b"/ColorSpace /DeviceGray /BitsPerComponent 1 /Length 8 >>\nstream\n"
        + pixels
        + b"\nendstream",
    ]
    payload = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(payload))
        payload.extend(f"{number} 0 obj\n".encode())
        payload.extend(obj)
        payload.extend(b"\nendobj\n")
    xref = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode())
    payload.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(payload)


def _minimal_xlsx() -> bytes:
    payload = BytesIO()
    with ZipFile(payload, "w", ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            "</Types>",
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        archive.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
            "</Relationships>",
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            '<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>Hello</t></is></c></row></sheetData></worksheet>',
        )
    return payload.getvalue()


@pytest.mark.parametrize(
    ("source_type", "filename", "content"),
    [
        ("text", "notes.txt", b"Hello from Docling"),
        ("excel", "table.xlsx", _minimal_xlsx()),
        ("pdf", "report.pdf", _minimal_pdf("Hello")),
        ("scanned_pdf", "scan.pdf", _minimal_scanned_pdf()),
    ],
)
@pytest.mark.asyncio
async def test_docling_parser_produces_traceable_fragment_artifacts(
    source_type: str, filename: str, content: bytes, tmp_path: Path
) -> None:
    if importlib.util.find_spec("docling") is None:
        pytest.skip("the declared Docling dependency is not installed in this environment")

    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    original = artifacts.put_bytes(f"source/{filename}", content)
    envelope = DocumentEnvelope(
        document_id="doc-1",
        document_version_id="ver-1",
        user_id="user-1",
        source_type=source_type,  # type: ignore[arg-type]
        source_uri=original.uri,
        content_hash=original.sha256,
        parser_version="docling-v1",
        pipeline_version="ingestion-v1",
        created_at=datetime(2026, 8, 5, tzinfo=UTC),
    )
    parser = DocumentParser(
        source_loader=lambda ref: content if ref == original else b"",
        artifacts=artifacts,
    )

    fragments = [fragment async for fragment in parser.parse_batches(original, envelope)]

    assert len(fragments) == 1
    fragment = fragments[0]
    assert fragment.envelope == envelope
    assert fragment.page_from == 1
    assert fragment.page_to >= fragment.page_from
    assert fragment.docling_document["schema_name"] == "DoclingDocument"
    assert fragment.artifact_ref is not None
    assert artifacts.verify(fragment.artifact_ref)
