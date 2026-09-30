"""Unit tests for MySQL engine connection defaults."""

from __future__ import annotations

from typing import Any, cast

from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncEngine

from agentic_rag.persistence import mysql as mysql_persistence


def test_parent_heading_path_json_is_normalized_fail_closed() -> None:
    from agentic_rag.persistence.repositories import _coerce_heading_path

    assert _coerce_heading_path(["工作经历", "项目"]) == ("工作经历", "项目")
    assert _coerce_heading_path(("工作经历",)) == ("工作经历",)
    assert _coerce_heading_path(["工作经历", 7]) == ()
    assert _coerce_heading_path("工作经历") == ()


def test_mysql_engine_initializes_utc_session_timezone(monkeypatch: Any) -> None:
    """Every application connection must use UTC for server-side timestamps."""
    captured: dict[str, Any] = {}
    sentinel = cast(AsyncEngine, object())

    def fake_create_async_engine(url: URL, **options: Any) -> AsyncEngine:
        captured["url"] = url
        captured["options"] = options
        return sentinel

    monkeypatch.setattr(mysql_persistence, "create_async_engine", fake_create_async_engine)

    assert (
        mysql_persistence.create_mysql_engine(
            "mysql+asyncmy://rag:secret@127.0.0.1:3306/rag"
        )
        is sentinel
    )
    assert captured["options"]["connect_args"]["init_command"] == (
        "SET time_zone = '+00:00'"
    )
