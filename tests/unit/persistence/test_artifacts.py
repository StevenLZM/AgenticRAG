"""Behavioral tests for the local artifact store."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agentic_rag.persistence.artifacts import ArtifactRef, LocalArtifactStore


def test_artifact_write_is_content_verified(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    ref = store.put_json("runs/r1/evidence.json", {"evidence_ids": ["e1"]})

    assert ref.uri == "artifact://runs/r1/evidence.json"
    assert ref.size_bytes == len(b'{"evidence_ids":["e1"]}')
    assert store.verify(ref)
    assert store.read_json(ref) == {"evidence_ids": ["e1"]}


def test_verify_detects_content_and_size_mismatch(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    ref = store.put_bytes("runs/r1/payload.bin", b"trusted")
    artifact_path = tmp_path / "runs/r1/payload.bin"

    artifact_path.write_bytes(b"tampered")

    assert not store.verify(ref)


@pytest.mark.parametrize(
    "unsafe_path",
    ("../outside.json", "runs/../../outside.json", "/tmp/outside.json"),
)
def test_write_rejects_paths_outside_artifact_root(
    tmp_path: Path, unsafe_path: str
) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")

    with pytest.raises(ValueError, match="relative path"):
        store.put_bytes(unsafe_path, b"unsafe")


def test_write_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    store = LocalArtifactStore(root)

    with pytest.raises(ValueError, match="artifact root"):
        store.put_bytes("linked/escape.bin", b"unsafe")

    assert not (outside / "escape.bin").exists()


def test_read_rejects_forged_ref_outside_artifact_root(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    forged_ref = ArtifactRef(
        uri="artifact://../outside.json",
        sha256="0" * 64,
        size_bytes=0,
    )

    with pytest.raises(ValueError, match="relative path"):
        store.read_json(forged_ref)


def test_failed_fsync_preserves_previous_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalArtifactStore(tmp_path)
    original = store.put_json("runs/r1/state.json", {"version": 1})

    def fail_fsync(_fd: int) -> None:
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="simulated fsync failure"):
        store.put_json("runs/r1/state.json", {"version": 2})

    assert store.read_json(original) == {"version": 1}
    assert not list((tmp_path / "runs/r1").glob("*.tmp"))


def test_read_json_rejects_invalid_json_without_weakening_verification(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path)
    ref = store.put_bytes("runs/r1/not-json.json", b"not json")

    assert store.verify(ref)
    with pytest.raises(json.JSONDecodeError):
        store.read_json(ref)
