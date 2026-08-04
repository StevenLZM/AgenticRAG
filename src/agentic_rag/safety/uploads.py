"""Deterministic safety gate for untrusted document uploads."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from enum import StrEnum
from io import BytesIO
from pathlib import Path
from typing import Protocol
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
_ALLOWED_OOXML_PART_SUFFIXES = (
    ".xml",
    ".rels",
    ".vml",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
    ".emf",
    ".wmf",
    ".svg",
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
            return not any(
                _member_looks_like_nested_archive(workbook, entry)
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
    return (
        disk_number == 0
        and directory_disk == 0
        and disk_entries == total_entries
        and total_entries != 0xFFFF
        and total_entries <= max_entries
        and directory_size <= max_directory_size
        and directory_offset + directory_size <= eocd_offset
    )


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


def _member_looks_like_nested_archive(workbook: ZipFile, entry: ZipInfo) -> bool:
    with workbook.open(entry) as member:
        header = member.read(512)
    return header.startswith(_NESTED_ARCHIVE_SIGNATURES) or (
        len(header) >= 262 and header[257:262] == b"ustar"
    )
