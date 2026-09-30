"""Deterministic safety gate for untrusted document uploads."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from enum import StrEnum
from io import BytesIO
from pathlib import Path
from typing import Protocol
from xml.parsers import expat
from zipfile import BadZipFile, ZipFile, ZipInfo

from pydantic import BaseModel, ConfigDict


PDF_MIME = "application/pdf"
TEXT_MIME = "text/plain"
XLS_MIME = "application/vnd.ms-excel"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
ZIP_MIME = "application/zip"
OCTET_STREAM_MIME = "application/octet-stream"
DEFAULT_MAX_UPLOAD_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_XLSX_ENTRIES = 4096

_PDF_SIGNATURE = b"%PDF-"
_OLE_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_ZIP_SIGNATURES = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_NESTED_ARCHIVE_SIGNATURES = _ZIP_SIGNATURES + (
    _OLE_SIGNATURE,
    b"Rar!\x1a\x07",
    b"7z\xbc\xaf\x27\x1c",
    b"\x1f\x8b",
    b"BZh",
    b"\xfd7zXZ\x00",
    b"MSCF",
    b"!<arch>\n",
    b"\x28\xb5\x2f\xfd",
    b"\x04\x22\x4d\x18",
)
_XML_PART_SUFFIXES = (".xml", ".rels", ".vml", ".svg")
_IMAGE_PART_SIGNATURES = {
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
    ".gif": (b"GIF87a", b"GIF89a"),
    ".bmp": (b"BM",),
    ".tif": (b"II*\x00", b"MM\x00*"),
    ".tiff": (b"II*\x00", b"MM\x00*"),
    ".emf": (b"\x01\x00\x00\x00",),
    ".wmf": (b"\xd7\xcd\xc6\x9a", b"\x01\x00\t\x00", b"\x02\x00\t\x00"),
}
_ALLOWED_OOXML_PART_SUFFIXES = (
    *_XML_PART_SUFFIXES,
    *_IMAGE_PART_SIGNATURES,
)
_MIME_EXTENSIONS = {
    PDF_MIME: {".pdf"},
    TEXT_MIME: {".txt"},
    XLS_MIME: {".xls"},
    XLSX_MIME: {".xlsx"},
}
_INSTRUCTION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|messages?)\b",
        r"\b(?:reveal|print|show|repeat)\s+(?:the\s+)?system\s+prompt\b",
        r"\b(?:system\s+prompt|developer\s+message)\b",
        r"\b(?:call|invoke|use)\s+(?:the\s+)?(?:tool|function)\b",
        r"\byou\s+are\s+(?:chatgpt|an?\s+assistant)\b",
    )
)


class UploadSafetyStatus(StrEnum):
    ACCEPTED = "accepted"
    QUARANTINED = "quarantined"
    REJECTED = "rejected"


class UploadDecision(BaseModel):
    """Complete, immutable result of scanning one original byte sequence."""

    model_config = ConfigDict(frozen=True)

    status: UploadSafetyStatus
    detected_mime: str
    content_hash: str
    reasons: tuple[str, ...] = ()


class UploadSafetyScanner(Protocol):
    def scan(
        self, filename: str, declared_mime: str, content: bytes
    ) -> UploadDecision: ...


def normalize_upload_filename(filename: str) -> str:
    """Return one normalized basename or raise for any path-like client input."""
    normalized = unicodedata.normalize("NFC", filename).strip()
    if (
        not normalized
        or "\x00" in normalized
        or normalized in {".", ".."}
        or "/" in normalized
        or "\\" in normalized
        or any(unicodedata.category(character) == "Cc" for character in normalized)
    ):
        raise ValueError("upload filename must be one safe basename")
    return normalized


class DefaultUploadSafetyScanner:
    """Validate filename, declared MIME, signature, and plain-text heuristics."""

    def __init__(
        self,
        *,
        max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
        max_xlsx_entries: int = DEFAULT_MAX_XLSX_ENTRIES,
    ) -> None:
        if max_upload_bytes <= 0:
            raise ValueError("maximum upload size must be positive")
        if max_xlsx_entries <= 0:
            raise ValueError("maximum XLSX entry count must be positive")
        self._max_upload_bytes = max_upload_bytes
        self._max_xlsx_entries = max_xlsx_entries

    def scan(
        self, filename: str, declared_mime: str, content: bytes
    ) -> UploadDecision:
        content_hash = hashlib.sha256(content).hexdigest()
        if len(content) > self._max_upload_bytes:
            return UploadDecision(
                status=UploadSafetyStatus.REJECTED,
                detected_mime=OCTET_STREAM_MIME,
                content_hash=content_hash,
                reasons=("file_too_large",),
            )
        detected_mime, parsed_text = _detect_mime_and_text(
            content, max_xlsx_entries=self._max_xlsx_entries
        )
        reasons: list[str] = []

        try:
            normalized_filename = normalize_upload_filename(filename)
        except ValueError:
            normalized_filename = ""
            reasons.append("unsafe_filename")

        normalized_mime = declared_mime.partition(";")[0].strip().lower()
        extension = Path(normalized_filename).suffix.lower()
        allowed_extensions = _MIME_EXTENSIONS.get(normalized_mime)
        if allowed_extensions is None:
            reasons.append("unsupported_mime")
        elif extension not in allowed_extensions:
            reasons.append("mime_extension_mismatch")

        if not content:
            reasons.append("empty_content")
        if detected_mime == ZIP_MIME:
            reasons.append("unsupported_container")
        elif detected_mime == OCTET_STREAM_MIME:
            reasons.append("unsupported_signature")
        if normalized_mime in _MIME_EXTENSIONS and detected_mime != normalized_mime:
            reasons.append("mime_signature_mismatch")

        if reasons:
            return UploadDecision(
                status=UploadSafetyStatus.REJECTED,
                detected_mime=detected_mime,
                content_hash=content_hash,
                reasons=tuple(dict.fromkeys(reasons)),
            )

        quarantine_reasons: list[str] = []
        if parsed_text is not None:
            if any(unicodedata.category(character) == "Cf" for character in parsed_text):
                quarantine_reasons.append("invisible_unicode")
            if any(pattern.search(parsed_text) for pattern in _INSTRUCTION_PATTERNS):
                quarantine_reasons.append("instruction_like_content")

        return UploadDecision(
            status=(
                UploadSafetyStatus.QUARANTINED
                if quarantine_reasons
                else UploadSafetyStatus.ACCEPTED
            ),
            detected_mime=detected_mime,
            content_hash=content_hash,
            reasons=tuple(quarantine_reasons),
        )


def _detect_mime_and_text(
    content: bytes, *, max_xlsx_entries: int
) -> tuple[str, str | None]:
    if content.startswith(_PDF_SIGNATURE):
        return PDF_MIME, None
    if content.startswith(_OLE_SIGNATURE):
        return XLS_MIME, None
    if content.startswith(_ZIP_SIGNATURES):
        return (
            (XLSX_MIME, None)
            if _is_safe_xlsx(content, max_entries=max_xlsx_entries)
            else (ZIP_MIME, None)
        )
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return OCTET_STREAM_MIME, None
    if any(
        unicodedata.category(character) == "Cc" and character not in "\t\n\r"
        for character in text
    ):
        return OCTET_STREAM_MIME, None
    return TEXT_MIME, text


def _is_safe_xlsx(content: bytes, *, max_entries: int) -> bool:
    if not _zip_metadata_within_limits(content, max_entries=max_entries):
        return False
    try:
        with ZipFile(BytesIO(content)) as workbook:
            entries = workbook.infolist()
            if len(entries) > max_entries:
                return False
            if any(not _is_safe_ooxml_member_name(entry.filename) for entry in entries):
                return False
            names = {entry.filename for entry in entries}
            if len(names) != len(entries):
                return False
            if not {"[Content_Types].xml", "xl/workbook.xml"}.issubset(names):
                return False
            if any(
                entry.flag_bits & 0x1
                or entry.file_size > 100 * 1024 * 1024
                or (
                    not entry.is_dir()
                    and not entry.filename.lower().endswith(
                        _ALLOWED_OOXML_PART_SUFFIXES
                    )
                )
                for entry in entries
            ):
                return False
            total_compressed = sum(max(entry.compress_size, 1) for entry in entries)
            total_uncompressed = sum(entry.file_size for entry in entries)
            if total_uncompressed > 100 * 1024 * 1024 or (
                total_uncompressed / total_compressed > 1000
            ):
                return False
            return all(
                _is_safe_ooxml_part_content(workbook, entry)
                for entry in entries
                if not entry.is_dir()
            )
    except (BadZipFile, EOFError, NotImplementedError, OSError, RuntimeError, ValueError):
        return False


def _zip_metadata_within_limits(content: bytes, *, max_entries: int) -> bool:
    # Reject metadata bombs before ZipFile materializes every central-directory
    # entry. ZIP64 and multi-disk packages are outside the XLSX upload allowlist.
    eocd_offset = content.rfind(b"PK\x05\x06", max(0, len(content) - 65_557))
    if eocd_offset < 0 or eocd_offset + 22 > len(content):
        return False
    disk_number = int.from_bytes(content[eocd_offset + 4 : eocd_offset + 6], "little")
    directory_disk = int.from_bytes(
        content[eocd_offset + 6 : eocd_offset + 8], "little"
    )
    disk_entries = int.from_bytes(
        content[eocd_offset + 8 : eocd_offset + 10], "little"
    )
    total_entries = int.from_bytes(
        content[eocd_offset + 10 : eocd_offset + 12], "little"
    )
    directory_size = int.from_bytes(
        content[eocd_offset + 12 : eocd_offset + 16], "little"
    )
    directory_offset = int.from_bytes(
        content[eocd_offset + 16 : eocd_offset + 20], "little"
    )
    max_directory_size = min(8 * 1024 * 1024, max_entries * 2048)
    metadata_bounds_are_safe = (
        disk_number == 0
        and directory_disk == 0
        and disk_entries == total_entries
        and total_entries != 0xFFFF
        and total_entries <= max_entries
        and directory_size <= max_directory_size
        and directory_offset + directory_size == eocd_offset
    )
    if not metadata_bounds_are_safe:
        return False
    parsed_entries = _count_central_directory_headers(
        content,
        directory_offset=directory_offset,
        directory_size=directory_size,
        max_entries=max_entries,
    )
    return parsed_entries == total_entries


def _count_central_directory_headers(
    content: bytes,
    *,
    directory_offset: int,
    directory_size: int,
    max_entries: int,
) -> int | None:
    cursor = directory_offset
    directory_end = directory_offset + directory_size
    count = 0
    while cursor < directory_end:
        if count >= max_entries or cursor + 46 > directory_end:
            return None
        if content[cursor : cursor + 4] != b"PK\x01\x02":
            return None
        compressed_size = int.from_bytes(
            content[cursor + 20 : cursor + 24], "little"
        )
        uncompressed_size = int.from_bytes(
            content[cursor + 24 : cursor + 28], "little"
        )
        name_length = int.from_bytes(content[cursor + 28 : cursor + 30], "little")
        extra_length = int.from_bytes(content[cursor + 30 : cursor + 32], "little")
        comment_length = int.from_bytes(
            content[cursor + 32 : cursor + 34], "little"
        )
        starting_disk = int.from_bytes(content[cursor + 34 : cursor + 36], "little")
        local_header_offset = int.from_bytes(
            content[cursor + 42 : cursor + 46], "little"
        )
        if (
            compressed_size == 0xFFFFFFFF
            or uncompressed_size == 0xFFFFFFFF
            or local_header_offset == 0xFFFFFFFF
            or starting_disk != 0
        ):
            return None
        record_size = 46 + name_length + extra_length + comment_length
        if cursor + record_size > directory_end:
            return None
        cursor += record_size
        count += 1
    return count if cursor == directory_end else None


def _is_safe_ooxml_member_name(name: str) -> bool:
    if (
        not name
        or "\\" in name
        or name.startswith("/")
        or re.match(r"^[A-Za-z]:", name)
        or any(ord(character) < 32 for character in name)
    ):
        return False
    segments = name.split("/")
    if name.endswith("/"):
        segments = segments[:-1]
    return bool(segments) and all(
        segment not in {"", ".", ".."} and ":" not in segment
        for segment in segments
    )


class _UnsafeXmlPart(ValueError):
    pass


def _is_safe_ooxml_part_content(workbook: ZipFile, entry: ZipInfo) -> bool:
    suffix = Path(entry.filename).suffix.lower()
    # The standard root relationship part is named `_rels/.rels`; pathlib
    # considers that basename a dotfile with no suffix, but it is still XML.
    if entry.filename.lower().endswith(_XML_PART_SUFFIXES):
        return _is_well_formed_xml_part(workbook, entry)
    expected_signatures = _IMAGE_PART_SIGNATURES.get(suffix)
    if expected_signatures is None:
        return False
    return _is_signature_valid_image_without_nested_archive(
        workbook, entry, expected_signatures
    )


def _is_well_formed_xml_part(workbook: ZipFile, entry: ZipInfo) -> bool:
    parser = expat.ParserCreate()

    def reject_declaration(*_args: object) -> None:
        raise _UnsafeXmlPart("DTD and entity declarations are not allowed")

    def reject_external_entity(*_args: object) -> int:
        raise _UnsafeXmlPart("external entities are not allowed")

    parser.StartDoctypeDeclHandler = reject_declaration
    parser.EntityDeclHandler = reject_declaration
    parser.ExternalEntityRefHandler = reject_external_entity
    parser.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_NEVER)
    try:
        with workbook.open(entry) as member:
            while chunk := member.read(64 * 1024):
                parser.Parse(chunk, False)
        parser.Parse(b"", True)
    except (expat.ExpatError, _UnsafeXmlPart):
        return False
    return True


def _is_signature_valid_image_without_nested_archive(
    workbook: ZipFile,
    entry: ZipInfo,
    expected_signatures: tuple[bytes, ...],
) -> bool:
    longest_signature = max(len(signature) for signature in _NESTED_ARCHIVE_SIGNATURES)
    tail = b""
    header = b""
    with workbook.open(entry) as member:
        while chunk := member.read(64 * 1024):
            if not header:
                header = chunk[:64]
            window = tail + chunk
            if any(signature in window for signature in _NESTED_ARCHIVE_SIGNATURES):
                return False
            if b"ustar" in window:
                return False
            tail = window[-(longest_signature - 1) :]
    return header.startswith(expected_signatures)
