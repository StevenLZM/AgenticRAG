"""Non-retrieval replies that converge on the shared memory finalizer."""

import asyncio
import json

from pydantic import BaseModel, ConfigDict, Field

from agentic_rag.query.state import QueryState, question_from_state, snapshot_from_state
from agentic_rag.runtime.model_gateway import ModelCall, ModelGateway, load_prompt


class ChatReply(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)
    text: str = Field(min_length=1, max_length=8_000)


async def run_chat(state: QueryState, gateway: ModelGateway) -> dict[str, object]:
    call = ModelCall(
        model_role="light", snapshot=snapshot_from_state(state), max_output_tokens=1024,
        messages=(
            {"role": "system", "content": load_prompt("chat_v1").content},
            {"role": "user", "content": json.dumps({
                "message": question_from_state(state),
                "memory_context": state.get("memory_context", {}),
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
