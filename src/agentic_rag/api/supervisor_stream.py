"""Scoped answer-v2 projection and replay, sharing existing Run authorization."""
import asyncio
from dataclasses import replace
import json

from fastapi import Request
from fastapi.responses import StreamingResponse
from sqlalchemy.exc import NoResultFound

from agentic_rag.api.errors import ApiException
from agentic_rag.domain.models import RunStatus
from agentic_rag.orchestration.store import CoordinationStore
from agentic_rag.orchestration.streaming import public_answer, read_chunks


def store_for(request: Request) -> CoordinationStore:
    container = request.app.state.container
    configured = getattr(container, "coordination_store", None)
    return configured or CoordinationStore(container.repositories.session_factory)


async def project_run(request: Request, run):
    if run.runtime_config_snapshot.get("answer_protocol_version") != "answer-v2":
        return run
    try:
        durable = await store_for(request).read_run(run.id)
    except (KeyError, NoResultFound):
        return run
    if durable["user_id"] != run.user_id:
        raise ApiException(status_code=404, error_code="QUERY_RUN_NOT_FOUND", message="Run not found")
    from agentic_rag.orchestration.authorization import authorize_sources, SourceAccessRevoked
    try:
        async with store_for(request).session_factory() as db:
            if durable["data"].get("answer"):
                await authorize_sources(db, durable, durable["data"]["answer"]["input"]["evidence"], context_sources=store_for(request).context_sources)
    except SourceAccessRevoked:
        return replace(run, status=RunStatus.FAILED, answer=None, error_code="SOURCE_ACCESS_REVOKED")
    status = RunStatus.RUNNING if durable["status"] in ("active", "finalizing") else RunStatus(durable["status"])
    if run.status in (RunStatus.CANCELLED, RunStatus.CANCEL_REQUESTED):
        status = run.status
    return replace(run, status=status, answer=public_answer(durable).model_dump(mode="json") if durable["data"].get("answer") else None)


async def events(request: Request, run_id: str, last_event_id: str | None) -> StreamingResponse:
    from agentic_rag.api.query_runs import _owned_run, QueryRunResponse
    raw = last_event_id or "1:0"
    try:
        epoch, seq = (int(v) for v in ("1:0" if raw == "0" else raw).split(":"))
        if epoch < 0 or seq < 0:
            raise ValueError()
    except ValueError as error:
        raise ApiException(status_code=422, error_code="INVALID_ANSWER_CURSOR", message="Expected epoch:sequence") from error
    owned = await _owned_run(request, run_id)
    store = store_for(request)

    async def stream():
        nonlocal epoch, seq
        while not await request.is_disconnected():
            try:
                current = await _owned_run(request, run_id)
                rows = await read_chunks(store, run_id, owned.user_id, epoch, seq)
            except (KeyError, NoResultFound):
                rows = []
                current = await _owned_run(request, run_id)
            except PermissionError:
                return
            except ApiException as error:
                if error.status_code == 404:
                    return
                raise
            if rows:
                for row in rows:
                    # Recheck session deletion and ownership for every batch item.
                    try:
                        current = await _owned_run(request, run_id)
                    except ApiException as error:
                        if error.status_code == 404:
                            return
                        raise
                    if current.error_code == "SOURCE_ACCESS_REVOKED":
                        return
                    epoch, seq = row["epoch"], row["seq"]
                    payload = {"run_id": run_id, "epoch": epoch, "seq": seq, **row["data"]}
                    if row["event_type"] in ("answer.completed", "answer.interrupted", "answer.cancelled"):
                        payload["run"] = QueryRunResponse.from_run(current).model_dump(mode="json")
                    yield f"id: {epoch}:{seq}\nevent: {row['event_type']}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                continue
            if current.status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED):
                return
            yield ": heartbeat\n\n"
            await asyncio.sleep(0.15)
    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
