"""Real QueryGraph and HTTP clients used by acceptance evaluation modes."""

from __future__ import annotations

import asyncio
import json
import hashlib
import fcntl
from pathlib import Path
from uuid import uuid4
from collections.abc import Mapping
from typing import Any, cast

from agentic_rag.domain.models import UserScope
from agentic_rag.query.public_answer import project_public_answer
from agentic_rag.query.graph import query_checkpoint_config
from agentic_rag.query.state import new_query_state
from agentic_rag.runtime.models import RuntimeConfigSnapshot
from evals.models import EvaluationCase
from evals.report import atomic_write_text


class EvaluationClientError(RuntimeError):
    """Raised when a real Graph/API evaluation cannot produce a safe result."""


class GraphQueryClient:
    """Invoke an already-composed production QueryGraph in-process."""

    provenance = "real_query_graph"

    def __init__(self, graph: object, *, snapshot: RuntimeConfigSnapshot) -> None:
        self._graph = graph
        self._snapshot = snapshot

    async def query(self, case: EvaluationCase) -> Mapping[str, object]:
        if case.runtime_config_snapshot_id != self._snapshot.snapshot_id:
            raise EvaluationClientError(
                "evaluation case snapshot does not match the composed QueryGraph"
            )
        state = new_query_state(
            run_id=f"eval-{case.case_id}",
            question=case.question,
            scope=UserScope(user_id=case.user_id),
            snapshot=self._snapshot,
        )
        graph = self._graph
        if callable(graph) and not hasattr(graph, "ainvoke"):
            graph = graph()
        if hasattr(graph, "ainvoke"):
            invoke = cast(Any, getattr(graph, "ainvoke"))
            value = invoke(dict(state), query_checkpoint_config(state))
        else:
            raise EvaluationClientError("Graph client requires an ainvoke boundary")
        if hasattr(value, "__await__"):
            value = await value
        if not isinstance(value, Mapping):
            raise EvaluationClientError("QueryGraph returned a non-object result")
        return _project_graph_result(
            value,
            case=case,
            snapshot_id=self._snapshot.snapshot_id,
            provenance=self.provenance,
            events_ref=f"inline://graph-events/{case.case_id}",
        )


class HttpQueryClient:
    """Submit cases to the public Query API and poll the durable Run."""

    provenance = "real_query_api"

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 0.25,
        http_client: object | None = None,
        collector: object | None = None,
        ledger_dir: Path | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url must not be blank")
        if timeout_seconds <= 0 or poll_interval_seconds <= 0:
            raise ValueError("HTTP evaluation timeouts must be positive")
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._collector = collector
        self._ledger_dir = Path(ledger_dir) if ledger_dir is not None else None
        if http_client is None:
            try:
                import httpx
            except ImportError as error:  # pragma: no cover - optional CLI dependency
                raise EvaluationClientError("httpx is required for API evaluation") from error
            http_client = httpx.AsyncClient(base_url=self._base_url, timeout=timeout_seconds)
        self._client = http_client
        self._owns_client = http_client.__class__.__module__.startswith("httpx")

    async def query(self, case: EvaluationCase) -> Mapping[str, object]:
        if self._collector is None:
            raise EvaluationClientError("API quality evaluation requires a scoped runtime collector")
        if self._ledger_dir is None:
            raise EvaluationClientError("API evaluation requires a durable query ledger")
        self._ledger_dir.mkdir(parents=True, exist_ok=True)
        path = self._ledger_dir / f"{case.case_id}.json"
        with path.with_suffix(".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise EvaluationClientError("query ledger already in use") from error
            return await self._query_locked(case, path)

    async def _query_locked(self, case: EvaluationCase, path: Path) -> Mapping[str, object]:
        binding = {"case_sha256": hashlib.sha256(case.model_dump_json().encode()).hexdigest(),
                   "base_url": self._base_url}
        if path.exists():
            record = json.loads(path.read_text())
            if any(record.get(key) != value for key, value in binding.items()):
                raise EvaluationClientError("query ledger binding changed; use a new output directory")
            if not record.get("run_id"):
                raise EvaluationClientError("uncertain submission requires reconciliation; refusing another POST")
        else:
            record = {**binding, "thread_id": f"eval-{uuid4()}", "status": "submitting"}

        def persist():
            atomic_write_text(path, json.dumps(record, ensure_ascii=False, indent=2) + "\n")

        if not record.get("run_id"):
            persist()  # Crash/timeout leaves an uncertain request, never permission to resubmit.
            response = await self._request("post", "/v1/query-runs", json={
                "query": case.question, "thread_id": record["thread_id"], "wait_seconds": 0,
            })
            created = _response_json(response)
            run_id = created.get("run_id")
            if not isinstance(run_id, str) or not run_id.strip():
                raise EvaluationClientError("Query API omitted run_id")
            record.update(run_id=run_id, status="submitted")
            persist()
        run_id = record["run_id"]

        deadline = asyncio.get_running_loop().time() + self._timeout_seconds
        final: Mapping[str, object] | None = None
        while asyncio.get_running_loop().time() < deadline:
            response = await self._request("get", f"/v1/query-runs/{run_id}")
            current = _response_json(response)
            if current.get("run_id") != run_id:
                raise EvaluationClientError("Query API returned another run_id")
            status = current.get("status")
            if status in {"completed", "failed", "cancelled"}:
                final = current
                record.update(status=status)
                persist()
                break
            await asyncio.sleep(self._poll_interval_seconds)
        if final is None:
            record.update(status="poll_timeout")
            persist()
            raise EvaluationClientError("Query API did not reach a terminal Run state")
        snapshot_id = final.get("runtime_config_snapshot_id")
        if snapshot_id != case.runtime_config_snapshot_id:
            raise EvaluationClientError("Query API returned a different runtime snapshot")
        if final.get("status") != "completed":
            raise EvaluationClientError(f"Query API Run ended with {final.get('status')!r}")
        events = await self._read_events(str(run_id), case, snapshot_id)
        answer = final.get("answer")
        if not isinstance(answer, Mapping):
            raise EvaluationClientError("completed Query API Run omitted audited answer")
        observed = await self._collector.collect(
            run_id=run_id, user_id=case.user_id,
            snapshot_id=case.runtime_config_snapshot_id, question=case.question,
        )
        http_answer = project_public_answer(answer, runtime_config_snapshot_id=snapshot_id)
        stored_answer = project_public_answer(observed["answer"], runtime_config_snapshot_id=snapshot_id)
        if (answer.get("runtime_config_snapshot_id") not in {None, snapshot_id}
                or http_answer is None or http_answer != stored_answer):
            raise EvaluationClientError("HTTP answer differs from collected public answer")
        record.update(status="collected")
        persist()
        return {
            "run_id": run_id,
            "runtime_config_snapshot_id": snapshot_id,
            "answer": dict(answer),
            "evidence_parent_ids": list(_parent_ids_from_answer(answer)),
            "ranked_parent_ids": observed["ranked_parent_ids"],
            "route": observed["route"],
            "retrieval_rounds": observed["retrieval_rounds"],
            "ranking_scope": observed["ranking_scope"],
            "final_context_parent_ids": observed["final_context_parent_ids"],
            "events_ref": f"inline://api-events/{case.case_id}",
            "events": events,
            "contexts": observed["contexts"],
            "citation_coverage": _citation_coverage_from_answer(answer),
            "audited": answer.get("audited") is True,
            "client_provenance": self.provenance,
        }

    async def _read_events(
        self,
        run_id: str,
        case: EvaluationCase,
        snapshot_id: object,
    ) -> list[dict[str, object]]:
        response = await self._request("get", f"/v1/query-runs/{run_id}/events")
        text = getattr(response, "text", "")
        if not isinstance(text, str):
            return []
        events: list[dict[str, object]] = []
        event_id: str | None = None
        event_type = "PROGRESS"
        for line in text.splitlines():
            if line.startswith("id: "):
                event_id = line[4:].strip()
            elif line.startswith("event: "):
                event_type = line[7:].strip() or "PROGRESS"
            elif line.startswith("data: "):
                try:
                    payload = json.loads(line[6:])
                except (TypeError, ValueError, json.JSONDecodeError):
                    payload = {}
                if not isinstance(payload, Mapping):
                    payload = {}
                if event_id:
                    events.append(
                        {
                            "event_key": f"api:{run_id}:{event_id}",
                            "run_id": run_id,
                            "user_id": case.user_id,
                            "runtime_config_snapshot_id": snapshot_id,
                            "event_type": str(payload.get("event_type", event_type)),
                            "attributes": {},
                        }
                    )
                event_id = None
                event_type = "PROGRESS"
        return events

    async def _request(self, method: str, path: str, **kwargs: object) -> object:
        request = getattr(self._client, method, None)
        if not callable(request):
            raise EvaluationClientError("HTTP client does not expose the requested method")
        response = request(path, **kwargs)
        if hasattr(response, "__await__"):
            response = await response
        status_code = getattr(response, "status_code", None)
        if not isinstance(status_code, int) or status_code >= 400:
            raise EvaluationClientError(f"Query API request failed with status {status_code!r}")
        return response

    async def aclose(self) -> None:
        if not self._owns_client:
            return
        close = getattr(self._client, "aclose", None)
        if callable(close):
            value = close()
            if hasattr(value, "__await__"):
                await value


def _project_graph_result(
    result: Mapping[str, object],
    *,
    case: EvaluationCase,
    snapshot_id: str,
    provenance: str,
    events_ref: str,
) -> dict[str, object]:
    answer = result.get("answer")
    answer_mapping = dict(answer) if isinstance(answer, Mapping) else {}
    parent_ids = _parent_ids_from_result(result)
    contexts = _contexts_from_result(result)
    return {
        "runtime_config_snapshot_id": snapshot_id,
        "answer": answer_mapping,
        "evidence_parent_ids": parent_ids,
        "ranked_parent_ids": parent_ids,
        "route": _route_from_result(result, case.expected_route),
        "events_ref": events_ref,
        "events": _events_from_result(result),
        "contexts": contexts,
        "citation_coverage": 1.0 if _citation_passed(result) else 0.0,
        "audited": answer_mapping.get("audited") is True,
        "client_provenance": provenance,
    }


def _response_json(response: object) -> Mapping[str, object]:
    value = getattr(response, "json", None)
    if not callable(value):
        raise EvaluationClientError("HTTP response does not expose JSON")
    payload = value()
    if not isinstance(payload, Mapping):
        raise EvaluationClientError("HTTP response JSON must be an object")
    return cast(Mapping[str, object], payload)


def _route_from_result(result: Mapping[str, object], fallback: str) -> str:
    value = result.get("route", fallback)
    if isinstance(value, Mapping):
        value = value.get("route", value.get("name", fallback))
    return value if isinstance(value, str) and value.strip() else fallback


def _parent_ids_from_result(result: Mapping[str, object]) -> list[str]:
    value = result.get("evidence")
    if not isinstance(value, (list, tuple)):
        packed = result.get("packed_context")
        value = packed.get("items", []) if isinstance(packed, Mapping) else []
    parents: list[str] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        parent_id = item.get("parent_id")
        if isinstance(parent_id, str) and parent_id.strip() and parent_id not in parents:
            parents.append(parent_id)
    return parents


def _parent_ids_from_answer(answer: Mapping[str, object]) -> tuple[str, ...]:
    value = answer.get("evidence_parent_ids", ())
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(value for value in value if isinstance(value, str) and value.strip())


def _contexts_from_result(result: Mapping[str, object]) -> list[str]:
    value = result.get("evidence")
    if not isinstance(value, (list, tuple)):
        packed = result.get("packed_context")
        value = packed.get("items", []) if isinstance(packed, Mapping) else []
    contexts: list[str] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            contexts.append(content.strip())
    return contexts


def _events_from_result(result: Mapping[str, object]) -> list[Mapping[str, object]]:
    value = result.get("events", ())
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, (list, tuple)) else []


def _citation_passed(result: Mapping[str, object]) -> bool:
    audits = result.get("audit_results")
    if not isinstance(audits, (list, tuple)) or not audits:
        return False
    last = audits[-1]
    if not isinstance(last, Mapping):
        return False
    citation = last.get("citation")
    return isinstance(citation, Mapping) and citation.get("passed") is True


def _citation_coverage_from_answer(answer: Mapping[str, object]) -> float:
    value = answer.get("citation_coverage")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 1.0 if answer.get("audited") is True else 0.0


__all__ = ["EvaluationClientError", "GraphQueryClient", "HttpQueryClient"]
