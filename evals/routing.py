"""Independent, paired classifier evaluation. No retrieval or memory clients."""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal
import hashlib
import json
import time

from pydantic import BaseModel, ConfigDict, Field

from agentic_rag.models.schemas import InformationSource, RouteAssessment, RouteDecision
from agentic_rag.query.routing_context import RoutingTurn, bound_history
from agentic_rag.query.routing_policy import RuntimeCapabilities, decide_route
from agentic_rag.runtime.model_gateway import ModelCall, ModelGateway, StructuredOutputValidationError, load_prompt
from agentic_rag.runtime.models import RuntimeConfigSnapshot


class RoutingCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    group: str
    question: str
    history: tuple[dict[str, str], ...] = ()
    required_source_set: tuple[InformationSource, ...]
    expected_route: Literal["chat", "fast_rag", "research"]
    expected_response_mode: Literal["conversation", "capability_unavailable", "clarify"] | None = None
    core: bool = False
    normalized_query_must_include: tuple[str, ...] = ()


class RoutingSample(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    case_id: str
    variant: Literal["v1", "v2"]
    repeat: int = Field(ge=0)
    actual_route: str | None = None
    response_mode: str | None = None
    normalized_query: str = ""
    latency_ms: int = Field(ge=0)
    error_code: str | None = None
    model_id: str
    prompt_hash: str
    dataset_sha256: str


def load_cases(path: Path) -> list[RoutingCase]:
    cases = [RoutingCase.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]
    if len({c.id for c in cases}) != len(cases) or not cases:
        raise ValueError("dataset IDs must be nonempty and unique")
    return cases


def dataset_hash(cases: Sequence[RoutingCase]) -> str:
    canonical = "\n".join(c.model_dump_json() for c in cases)
    return hashlib.sha256(canonical.encode()).hexdigest()


async def run_routing_eval(
    cases: Sequence[RoutingCase], gateway: ModelGateway, snapshot: RuntimeConfigSnapshot,
    *, repeats: int = 3, on_sample: Callable[[RoutingSample], None] | None = None,
) -> list[RoutingSample]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    samples = []
    digest = dataset_hash(cases)
    caps = RuntimeCapabilities(knowledge_base=True)
    for case in cases:
        history = bound_history([RoutingTurn(id=f"{case.id}:{i}", **turn) for i, turn in enumerate(case.history)])
        payload = json.dumps({"question": case.question, "memory_context": {}, "routing_context": {
            "requested_at": "2026-10-01T12:00:00+08:00", "timezone": "Asia/Shanghai",
            "history": [t.model_dump(mode="json") for t in history]}}, ensure_ascii=False)
        for repeat in range(repeats):
            for variant in ("v1", "v2"):
                prompt = load_prompt(f"router_{variant}")
                system = prompt.content + "\nSERVER CAPABILITIES (not user permissions): " + caps.model_dump_json()
                call = ModelCall(model_role="light", snapshot=snapshot,
                    messages=({"role": "system", "content": system}, {"role": "user", "content": payload}))
                started = time.perf_counter()
                result: dict[str, object] = {}
                try:
                    if variant == "v1":
                        response = await gateway.complete_structured(call, RouteDecision)
                        value = RouteDecision.model_validate(response.value)
                        result.update(actual_route=value.route, response_mode="conversation" if value.route == "chat" else None,
                                      normalized_query=value.normalized_query)
                    else:
                        response = await gateway.complete_structured(call, RouteAssessment)
                        assessment = RouteAssessment.model_validate(response.value)
                        decision = decide_route(assessment, caps)
                        result.update(actual_route=decision.route, response_mode=decision.response_mode,
                                      normalized_query=assessment.normalized_query)
                    result["model_id"] = response.actual_model
                except (StructuredOutputValidationError, ValueError, TypeError):
                    result["error_code"] = "schema_invalid"
                except Exception:
                    # Do not persist provider messages, URLs, keys, or raw response bodies.
                    result["error_code"] = "provider_unavailable"
                sample = RoutingSample(case_id=case.id, variant=variant, repeat=repeat,
                    latency_ms=round((time.perf_counter() - started) * 1000),
                    prompt_hash=hashlib.sha256(prompt.content.encode()).hexdigest(), dataset_sha256=digest,
                    **{"model_id": snapshot.light_model_id, **result})
                samples.append(sample)
                if on_sample is not None:
                    on_sample(sample)
    return samples


def score_routing(samples: Sequence[RoutingSample], cases: Sequence[RoutingCase], *, repeats: int = 3) -> dict[str, object]:
    case_map = {c.id: c for c in cases}
    variants = {}
    expected = {(c.id, r) for c in cases for r in range(repeats)}
    for variant in ("v1", "v2"):
        rows = [s for s in samples if s.variant == variant]
        counts = Counter((s.case_id, s.repeat) for s in rows)
        complete = set(counts) == expected and all(n == 1 for n in counts.values())
        valid = [s for s in rows if not s.error_code and s.case_id in case_map
                 and s.actual_route in {"chat", "fast_rag", "research"}
                 and s.dataset_sha256 == dataset_hash(cases)]
        def correct(s):
            c = case_map[s.case_id]
            return s.actual_route == c.expected_route and s.response_mode == c.expected_response_mode and all(
                entity in s.normalized_query for entity in c.normalized_query_must_include)
        wrong = [s for s in valid if not correct(s)]
        retrieval = [s for s in valid if case_map[s.case_id].expected_route != "chat"]
        no_retrieval = [s for s in valid if case_map[s.case_id].expected_route == "chat"]
        false_retrieval = sum(s.actual_route != "chat" for s in no_retrieval)
        missed = sum(s.actual_route == "chat" for s in retrieval)
        cores = [s for s in valid if case_map[s.case_id].core]
        core_pass = len(cores) == sum(c.core for c in cases) * repeats and all(correct(s) for s in cores)
        accuracy = (len(valid) - len(wrong)) / len(valid) if valid else 0
        false_rate = false_retrieval / len(no_retrieval) if no_retrieval else 0
        missed_rate = missed / len(retrieval) if retrieval else 0
        status = "INCOMPLETE" if not complete or len(valid) != len(rows) else (
            "PASS" if core_pass and accuracy >= .95 and false_rate <= .05 and missed_rate <= .05 else "FAIL")
        variants[variant] = dict(status=status, expected=len(expected), received=len(rows), valid=len(valid),
            invalid=len(rows)-len(valid), accuracy=accuracy, core_pass=core_pass,
            false_retrieval_rate=false_rate, false_retrieval_denominator=len(no_retrieval),
            missed_retrieval_rate=missed_rate, missed_retrieval_denominator=len(retrieval),
            clarification_rate=sum(s.response_mode == "clarify" for s in valid) / len(valid) if valid else 0,
            failed_cases=sorted({s.case_id for s in wrong}),
            mean_latency_ms=sum(s.latency_ms for s in valid) / len(valid) if valid else None)
        variants[variant]["groups"] = {
            group: {"valid": len(group_rows), "correct": sum(correct(s) for s in group_rows)}
            for group in sorted({c.group for c in cases})
            for group_rows in [[s for s in valid if case_map[s.case_id].group == group]]
        }
    return {"status": variants["v2"]["status"], "variants": variants}
