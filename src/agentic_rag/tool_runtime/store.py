"""SQLite invocation journal with atomic run budgets and attempt fencing."""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import aiosqlite

from agentic_rag.tool_runtime.models import ToolContext, ToolDefinition, ToolError, ToolResult, json_payload

if TYPE_CHECKING:
    from agentic_rag.tool_runtime.runtime import RuntimeLimits


@dataclass(frozen=True, slots=True)
class InvocationClaim:
    """A cached result or a lease token authorizing one physical execution."""

    token: str | None
    deadline: float
    result: ToolResult | None = None


class InvocationStore:
    """Single-worker durable journal; no network exactly-once promise."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._connection: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    async def _db(self) -> aiosqlite.Connection:
        if self._closed:
            raise ToolError("runtime_closed")
        if self._connection is None:
            if self.path != ":memory:":
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            connection = await aiosqlite.connect(self.path, isolation_level=None)
            try:
                connection.row_factory = aiosqlite.Row
                await connection.execute("PRAGMA busy_timeout = 5000")
                await connection.executescript("""
                    CREATE TABLE IF NOT EXISTS tool_runs (
                        user_id TEXT NOT NULL, run_id TEXT NOT NULL,
                        session_id TEXT NOT NULL, snapshot_id TEXT NOT NULL,
                        deadline REAL NOT NULL, calls INTEGER NOT NULL DEFAULT 0,
                        discoveries INTEGER NOT NULL DEFAULT 0,
                        max_calls INTEGER NOT NULL, max_discoveries INTEGER NOT NULL,
                        cancelled INTEGER NOT NULL DEFAULT 0,
                        PRIMARY KEY (user_id, run_id)
                    );
                    CREATE TABLE IF NOT EXISTS tool_invocations (
                        user_id TEXT NOT NULL, run_id TEXT NOT NULL, call_id TEXT NOT NULL,
                        session_id TEXT NOT NULL, tool_id TEXT NOT NULL,
                        definition_version TEXT NOT NULL, definition_hash TEXT NOT NULL,
                        arguments_hash TEXT NOT NULL, state TEXT NOT NULL,
                        lease_until REAL NOT NULL, token TEXT NOT NULL,
                        result_json TEXT,
                        PRIMARY KEY (user_id, run_id, call_id)
                    );
                """)
            except BaseException:
                await connection.close()
                raise
            self._connection = connection
        return self._connection

    @staticmethod
    async def _row(db: aiosqlite.Connection, sql: str, args: tuple[Any, ...]) -> aiosqlite.Row | None:
        async with db.execute(sql, args) as cursor:
            return await cursor.fetchone()

    async def _run(
        self, db: aiosqlite.Connection, context: ToolContext, limits: RuntimeLimits,
    ) -> aiosqlite.Row:
        key = (context.scope.user_id, context.run_id)
        await db.execute(
            """INSERT OR IGNORE INTO tool_runs
               (user_id, run_id, session_id, snapshot_id, deadline, max_calls, max_discoveries)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (*key, context.session_id, context.snapshot.snapshot_id, context.deadline,
             limits.max_calls, limits.max_discoveries),
        )
        row = await self._row(db, "SELECT * FROM tool_runs WHERE user_id=? AND run_id=?", key)
        assert row is not None
        if row["session_id"] != context.session_id or row["snapshot_id"] != context.snapshot.snapshot_id:
            raise ToolError("context_mismatch")
        if row["cancelled"]:
            raise ToolError("run_cancelled")
        deadline = min(row["deadline"], context.deadline)
        if deadline <= time.time():
            raise ToolError("deadline_exceeded")
        await db.execute(
            """UPDATE tool_runs SET deadline=?, max_calls=min(max_calls, ?),
               max_discoveries=min(max_discoveries, ?) WHERE user_id=? AND run_id=?""",
            (deadline, limits.max_calls, limits.max_discoveries, *key),
        )
        updated = await self._row(db, "SELECT * FROM tool_runs WHERE user_id=? AND run_id=?", key)
        assert updated is not None
        return updated

    async def check_context(self, context: ToolContext, limits: RuntimeLimits) -> float:
        """Bind a run and return its persisted, non-extendable deadline."""
        async with self._lock:
            db = await self._db()
            await db.execute("BEGIN IMMEDIATE")
            try:
                row = await self._run(db, context, limits)
                await db.commit()
                return float(row["deadline"])
            except BaseException:
                await db.rollback()
                raise

    async def consume_discovery(self, context: ToolContext, limits: RuntimeLimits) -> float:
        async with self._lock:
            db = await self._db()
            await db.execute("BEGIN IMMEDIATE")
            try:
                row = await self._run(db, context, limits)
                if row["discoveries"] >= row["max_discoveries"]:
                    raise ToolError("discovery_budget_exhausted")
                await db.execute(
                    "UPDATE tool_runs SET discoveries=discoveries+1 WHERE user_id=? AND run_id=?",
                    (context.scope.user_id, context.run_id),
                )
                await db.commit()
                return float(row["deadline"])
            except BaseException:
                await db.rollback()
                raise

    async def reserve(
        self, context: ToolContext, definition: ToolDefinition, arguments_hash: str,
        call_id: str, limits: RuntimeLimits, *, attempt_deadline: float | None = None,
    ) -> InvocationClaim:
        """Atomically reuse a success, exclude live duplicates, or spend a call."""
        key = (context.scope.user_id, context.run_id, call_id)
        definition_hash = hashlib.sha256(json_payload(definition.model_dump(mode="json")).encode()).hexdigest()
        async with self._lock:
            db = await self._db()
            await db.execute("BEGIN IMMEDIATE")
            try:
                run = await self._run(db, context, limits)
                old = await self._row(
                    db, "SELECT * FROM tool_invocations WHERE user_id=? AND run_id=? AND call_id=?", key,
                )
                now = time.time()
                if old is not None:
                    if old["tool_id"] != definition.tool_id or old["arguments_hash"] != arguments_hash:
                        raise ToolError("call_id_conflict")
                    if old["definition_hash"] != definition_hash:
                        raise ToolError("definition_changed")
                    cached = ToolResult.model_validate_json(old["result_json"]) if old["result_json"] else None
                    if cached is not None and (cached.status == "success" or not cached.retryable):
                        await db.commit()
                        return InvocationClaim(None, run["deadline"], cached)
                    if old["lease_until"] > now:
                        raise ToolError("call_in_progress", retryable=True)
                if run["calls"] >= run["max_calls"]:
                    raise ToolError("tool_budget_exhausted")
                deadline = min(run["deadline"], now + limits.call_timeout_seconds)
                if attempt_deadline is not None:
                    deadline = min(deadline, attempt_deadline)
                if deadline <= now:
                    raise ToolError("tool_timeout", retryable=True)
                token = uuid4().hex
                await db.execute(
                    """INSERT INTO tool_invocations
                       (user_id, run_id, call_id, session_id, tool_id, definition_version,
                        definition_hash, arguments_hash, state, lease_until, token, result_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, ?, NULL)
                       ON CONFLICT(user_id, run_id, call_id) DO UPDATE SET
                       state='running', lease_until=excluded.lease_until,
                       token=excluded.token, result_json=NULL""",
                    (*key, context.session_id, definition.tool_id, definition.version,
                     definition_hash, arguments_hash, deadline, token),
                )
                await db.execute(
                    "UPDATE tool_runs SET calls=calls+1 WHERE user_id=? AND run_id=?", key[:2],
                )
                await db.commit()
                return InvocationClaim(token, deadline)
            except BaseException:
                await db.rollback()
                raise

    async def finish(self, context: ToolContext, call_id: str, token: str, result: ToolResult) -> bool:
        """A superseded lease or cancelled run cannot publish a late success."""
        async with self._lock:
            db = await self._db()
            async with db.execute(
                """UPDATE tool_invocations SET state=?, result_json=?
                   WHERE user_id=? AND run_id=? AND call_id=? AND token=?
                   AND (?='error' OR lease_until>?)
                   AND EXISTS (SELECT 1 FROM tool_runs r WHERE r.user_id=tool_invocations.user_id
                       AND r.run_id=tool_invocations.run_id AND r.session_id=?
                       AND r.cancelled=0 AND r.deadline>?)""",
                (result.status, result.model_dump_json(), context.scope.user_id, context.run_id,
                 call_id, token, result.status, time.time(), context.session_id, time.time()),
            ) as cursor:
                return cursor.rowcount == 1

    async def cancel_run(self, context: ToolContext) -> None:
        """Permanently stop an explicitly cancelled Run, not a worker attempt."""
        async with self._lock:
            db = await self._db()
            await db.execute(
                "UPDATE tool_runs SET cancelled=1 WHERE user_id=? AND run_id=? AND session_id=?",
                (context.scope.user_id, context.run_id, context.session_id),
            )

    async def interrupt(self, context: ToolContext, call_id: str, token: str) -> bool:
        """Revoke an interrupted attempt without changing its Run's budget."""
        async with self._lock:
            db = await self._db()
            async with db.execute(
                """UPDATE tool_invocations
                   SET state='interrupted', token=?, lease_until=?, result_json=NULL
                   WHERE user_id=? AND run_id=? AND call_id=? AND session_id=?
                       AND token=? AND state='running'
                       AND EXISTS (SELECT 1 FROM tool_runs r
                           WHERE r.user_id=tool_invocations.user_id
                               AND r.run_id=tool_invocations.run_id AND r.snapshot_id=?)""",
                (uuid4().hex, time.time(), context.scope.user_id, context.run_id, call_id,
                 context.session_id, token, context.snapshot.snapshot_id),
            ) as cursor:
                return cursor.rowcount == 1

    async def get_usage(self, context: ToolContext) -> dict[str, Any]:
        async with self._lock:
            db = await self._db()
            row = await self._row(db, "SELECT * FROM tool_runs WHERE user_id=? AND run_id=?",
                                  (context.scope.user_id, context.run_id))
            if row is None:
                return {"calls": 0, "discoveries": 0, "deadline": context.deadline}
            if row["session_id"] != context.session_id or row["snapshot_id"] != context.snapshot.snapshot_id:
                raise ToolError("context_mismatch")
            return {"calls": row["calls"], "discoveries": row["discoveries"], "deadline": row["deadline"]}

    async def aclose(self) -> None:
        async with self._lock:
            self._closed = True
            if self._connection is not None:
                await self._connection.close()
                self._connection = None
