"""Non-retrieval replies that converge on the shared memory finalizer."""

import asyncio
import json
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from agentic_rag.query.state import QueryState, question_from_state, snapshot_from_state
from agentic_rag.runtime.model_gateway import ModelCall, ModelGateway, load_prompt
from agentic_rag.models.schemas import InformationSource
from agentic_rag.query.routing_policy import PolicyDecision


class ChatReply(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)
    text: str = Field(min_length=1, max_length=8_000)


def controlled_reply(mode: str, reason_code: str, *, missing_sources: Sequence[InformationSource]) -> str:
    if mode == "capability_unavailable":
        return "目前尚未接入实时数据或外部查询服务，无法核实你请求的信息。我可以帮助分析你提供或上传的资料。"
    if mode == "clarify":
        if missing_sources:
            return "我可以分析已上传的资料，但目前无法查询外部实时信息。是否先分析文档部分？"
        return "请补充要查询的对象或文档，以及希望了解的内容，以便我准确回答。"
    if mode == "technical_error":
        return "本次处理所需的服务暂时不可用，未能完成回答，请稍后重试。"
    raise ValueError("not a controlled response mode")


async def run_chat(state: QueryState, gateway: ModelGateway) -> dict[str, object]:
    mode = state.get("response_mode")
    if mode in {"capability_unavailable", "clarify", "technical_error"}:
        policy = PolicyDecision.model_validate(state.get("policy_decision"))
        return {"answer": {"route": "chat", "status": policy.termination_reason,
                           "segments": [{"kind": "content", "text": controlled_reply(
                               mode, policy.reason_code, missing_sources=policy.missing_sources), "evidence_ids": []}]},
                "termination_reason": policy.termination_reason, "audit_results": []}
    call = ModelCall(
        model_role="light", snapshot=snapshot_from_state(state), max_output_tokens=1024,
        messages=(
            {"role": "system", "content": load_prompt("chat_v2" if state.get("routing_policy_version") == "routing-v2" else "chat_v1").content},
            {"role": "user", "content": json.dumps({
                "message": question_from_state(state),
                "memory_context": state.get("memory_context", {}),
                "routing_context": state.get("routing_context", {}),
            }, ensure_ascii=False)},
        ),
    )
    try:
        response = await gateway.complete_structured(call, ChatReply)
        reply = ChatReply.model_validate(getattr(response, "value", response))
    except asyncio.CancelledError:
        raise
    except Exception:
        return {
            "answer": {"status": "cannot_answer"},
            "termination_reason": "cannot_answer",
            "errors": [*state.get("errors", []), {"code": "chat_unavailable"}],
        }
    return {
        "answer": {"route": "chat", "segments": [
            {"kind": "content", "text": reply.text, "evidence_ids": []},
        ]},
        "evidence": [], "audit_results": [],
    }
