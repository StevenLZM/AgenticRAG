"""Real independent processes and disposable Redis; opt-in infrastructure marker."""
import asyncio
from datetime import UTC, datetime, timedelta
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

import pytest
from redis.asyncio import Redis
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from agentic_rag.orchestration import schema
from agentic_rag.orchestration.store import CoordinationStore
from tests.unit.orchestration.test_store import plan
from tests.unit.orchestration.test_evidence import SNAPSHOT

pytestmark = pytest.mark.integration


async def process(dsn, mode, arg):
    root = Path(__file__).resolve().parents[3]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(root / "src"), str(root)))}
    proc = await asyncio.create_subprocess_exec(sys.executable, "-m", "tests.integration.orchestration.worker_process", dsn, mode, arg,
                                                cwd=root, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    output, _ = await asyncio.wait_for(proc.communicate(), 30)
    return proc.returncode, output.decode()


async def test_worker_kill_and_coordinator_restart_with_real_redis(tmp_path):
    binary = shutil.which("redis-server")
    if not binary:
        pytest.skip("redis-server executable unavailable")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    log = (tmp_path / "redis.log").open("w")
    redis_process = subprocess.Popen([binary, "--bind", "127.0.0.1", "--port", str(port), "--save", "", "--appendonly", "no", "--dir", str(tmp_path)], stdout=log, stderr=subprocess.STDOUT)
    redis = Redis.from_url(f"redis://127.0.0.1:{port}/0", decode_responses=True)
    dsn = f"sqlite+aiosqlite:///{tmp_path / 'process.db'}"
    engine = create_async_engine(dsn)
    try:
        for _ in range(50):
            try:
                await redis.ping()
                break
            except Exception:
                if redis_process.poll() is not None:
                    pytest.fail("disposable Redis startup failed: " + (tmp_path / "redis.log").read_text())
                await asyncio.sleep(.05)
        async with engine.begin() as connection:
            await connection.run_sync(schema.metadata.create_all)
        store = CoordinationStore(async_sessionmaker(engine, expire_on_commit=False), enforce_source_access=False)
        await store.initialize(run_id="run-1", user_id="user-1", question="Compare revenues", snapshot=SNAPSHOT)
        work = await store.submit_plan("run-1", "decision", 0, plan())
        completed, crashed = await asyncio.gather(process(dsn, "complete", work[0]), process(dsn, "crash", work[1]))
        assert completed[0] == 0, completed[1]
        assert crashed[0] == 17, crashed[1]
        state = await store.read_run("run-1")
        assert state["data"]["tasks"]["r2"]["usage"]["queries"] == 1
        async with store.session_factory.begin() as db:
            await db.execute(update(schema.work_items).where(schema.work_items.c.id == work[1]).values(lease_until=datetime.now(UTC) - timedelta(seconds=60)))
        url = f"redis://127.0.0.1:{port}/0"
        code, output = await process(dsn, "coordinate", url)
        assert code == 0, output
        code, output = await process(dsn, "complete", work[1])
        assert code == 0, output
        # Lose only this disposable broker's notifications; SQL still owns work.
        await redis.flushdb()
        code, output = await process(dsn, "coordinate", url)
        assert code == 0, output
        restored = CoordinationStore(store.session_factory)
        state = await restored.read_run("run-1")
        assert state["data"]["tasks"]["r2"]["usage"]["queries"] == 2
        assert state["data"]["groups"]["g1"]["snapshot_id"]
        assert len(await restored.list_ready("prepare")) == 1
        async with store.session_factory() as db:
            count = (await db.execute(select(schema.inbox))).mappings().all()
        assert len(count) == 2
        code, output = await process(dsn, "coordinate", url)
        assert code == 0, output
        assert len(await restored.list_ready("prepare")) == 1
    finally:
        await redis.aclose()
        await engine.dispose()
        redis_process.terminate()
        try:
            redis_process.wait(5)
        except subprocess.TimeoutExpired:
            redis_process.kill()
            redis_process.wait(5)
        log.close()
