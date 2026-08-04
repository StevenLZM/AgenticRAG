"""Content-addressed references backed by atomic local file writes."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol


_ARTIFACT_URI_PREFIX = "artifact://"


@dataclass(frozen=True)
class ArtifactRef:
    """Stable metadata needed to locate and verify a stored artifact."""

    uri: str
    sha256: str
    size_bytes: int


class ArtifactStore(Protocol):
    """Persistence boundary for immutable, integrity-checked payloads."""

    def put_json(
        self, relative_path: str, value: Mapping[str, Any]
    ) -> ArtifactRef: ...

    def put_bytes(self, relative_path: str, value: bytes) -> ArtifactRef: ...

    def read_json(self, ref: ArtifactRef) -> Any: ...

    def verify(self, ref: ArtifactRef) -> bool: ...

    def delete(self, ref: ArtifactRef) -> None: ...


class LocalArtifactStore:
    """Store artifacts beneath one trusted, exclusively managed root.

    The configured root is a security boundary: it and its descendants must not be
    mutated by untrusted or same-identity processes while operations are in flight.
    The constructor rejects a symlink root and, on POSIX, roots not owned by the
    current user or writable by group/other users. Existing symlink escapes inside
    the root are also rejected, but path-based operations do not claim protection
    against a hostile same-owner process racing directory changes.
    """

    def __init__(self, root: Path) -> None:
        requested_root = Path(root)
        if requested_root.is_symlink():
            raise ValueError("artifact root must not be a symbolic link")
        requested_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not requested_root.is_dir():
            raise ValueError("artifact root must be a directory")

        root_stat = requested_root.stat()
        getuid = getattr(os, "getuid", None)
        if getuid is not None and root_stat.st_uid != getuid():
            raise PermissionError("artifact root must be exclusively managed")
        if root_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise PermissionError("artifact root must be exclusively managed")

        self._root = requested_root.resolve()

    def put_json(self, relative_path: str, value: Mapping[str, Any]) -> ArtifactRef:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return self.put_bytes(relative_path, encoded)

    def put_bytes(self, relative_path: str, value: bytes) -> ArtifactRef:
        destination, normalized_path = self._resolve_relative_path(relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected_sha256 = hashlib.sha256(value).hexdigest()
        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(value)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())

            written_sha256, written_size = _hash_file(temporary_path)
            if written_sha256 != expected_sha256 or written_size != len(value):
                raise OSError("artifact temporary write failed integrity verification")

            os.replace(temporary_path, destination)
            temporary_path = None
            _fsync_directory(destination.parent)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        return ArtifactRef(
            uri=f"{_ARTIFACT_URI_PREFIX}{normalized_path.as_posix()}",
            sha256=expected_sha256,
            size_bytes=len(value),
        )

    def read_json(self, ref: ArtifactRef) -> Any:
        path = self._path_from_ref(ref)
        payload = path.read_bytes()
        if (
            len(payload) != ref.size_bytes
            or hashlib.sha256(payload).hexdigest() != ref.sha256
        ):
            raise ValueError("artifact content does not match its reference")
        return json.loads(payload.decode("utf-8"))

    def verify(self, ref: ArtifactRef) -> bool:
        try:
            path = self._path_from_ref(ref)
            sha256, size_bytes = _hash_file(path)
        except (FileNotFoundError, IsADirectoryError, OSError, ValueError):
            return False
        return sha256 == ref.sha256 and size_bytes == ref.size_bytes

    def delete(self, ref: ArtifactRef) -> None:
        """Delete exactly one referenced artifact, if it still exists."""
        self._path_from_ref(ref).unlink(missing_ok=True)

    def _path_from_ref(self, ref: ArtifactRef) -> Path:
        if not ref.uri.startswith(_ARTIFACT_URI_PREFIX):
            raise ValueError("artifact reference must use the artifact:// URI scheme")
        relative_path = ref.uri.removeprefix(_ARTIFACT_URI_PREFIX)
        path, _ = self._resolve_relative_path(relative_path)
        return path

    def _resolve_relative_path(self, relative_path: str) -> tuple[Path, Path]:
        candidate_path = Path(relative_path)
        if (
            not relative_path
            or candidate_path.is_absolute()
            or ".." in candidate_path.parts
        ):
            raise ValueError("artifact path must be a non-empty relative path")

        resolved_path = (self._root / candidate_path).resolve()
        try:
            resolved_path.relative_to(self._root)
        except ValueError as exc:
            raise ValueError("artifact path must remain within the artifact root") from exc
        if resolved_path == self._root:
            raise ValueError("artifact path must identify a file below the artifact root")
        return resolved_path, candidate_path


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as artifact_file:
        while chunk := artifact_file.read(1024 * 1024):
            digest.update(chunk)
            size_bytes += len(chunk)
    return digest.hexdigest(), size_bytes


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(path, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
