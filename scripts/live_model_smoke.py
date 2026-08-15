"""Small, explicit live checks for the configured DeepSeek and Qwen providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from agentic_rag.config import Settings
from agentic_rag.models.schemas import RouteDecision
from agentic_rag.runtime.model_gateway import ModelCall, ModelGateway
from agentic_rag.runtime.models import RuntimeConfigSnapshot


class LiveModelConfigurationError(RuntimeError):
    """Raised instead of skipping when selected live tests lack credentials."""


@dataclass(frozen=True)
class LiveModelSmokeResult:
    routing_model_id: str
    completion_model_id: str
    embedding_model_id: str
    embedding_dimensions: int


class _ChatOnlyClient:
    """Use the widely supported chat-completions route through ModelGateway."""

    def __init__(self, client: AsyncOpenAI) -> None:
        self.chat = client.chat


async def run_live_model_smoke(settings: Settings) -> LiveModelSmokeResult:
    """Call the required three provider contracts and retain provider model IDs."""
    deepseek_key = _credential(settings.deepseek_api_key, "AGENTIC_RAG_DEEPSEEK_API_KEY")
    qwen_key = _credential(settings.qwen_api_key, "AGENTIC_RAG_QWEN_API_KEY")
    deepseek = AsyncOpenAI(
        api_key=deepseek_key,
        base_url=settings.deepseek_base_url,
        timeout=30.0,
        max_retries=0,
    )
    qwen = AsyncOpenAI(
        api_key=qwen_key,
        base_url=settings.qwen_embedding_base_url,
        timeout=30.0,
        max_retries=0,
    )
    try:
        gateway = ModelGateway(_ChatOnlyClient(deepseek), max_retries=0)
        snapshot = _snapshot(settings)
        routing = await gateway.complete_structured(
            ModelCall(
                model_role="light",
                snapshot=snapshot,
                temperature=0.0,
                messages=(
                    {
                        "role": "system",
                        "content": "Return one JSON object only. No markdown or explanation.",
                    },
                    {
                        "role": "user",
                        "content": (
                            'Classify this simple question. Return '
                            '{"route":"fast_rag","normalized_query":"hello",'
                            '"reason_code":"simple"}.'
                        ),
                    },
                ),
            ),
            RouteDecision,
        )
        completion = await gateway.complete(
            ModelCall(
                model_role="main",
                snapshot=snapshot,
                temperature=0.0,
                messages=(
                    {"role": "user", "content": "Reply with exactly: ok"},
                ),
            )
        )
        embedding_response: Any = await qwen.embeddings.create(
            model="text-embedding-v3", input="agentic rag live smoke"
        )
        rows: list[Any] = list(getattr(embedding_response, "data", ()))
        if len(rows) != 1:
            raise RuntimeError("Qwen embedding response did not contain exactly one vector")
        vector = getattr(rows[0], "embedding", None)
        if not isinstance(vector, list) or len(vector) != 1024:
            raise RuntimeError("Qwen text-embedding-v3 did not return 1024 dimensions")
        embedding_model = getattr(embedding_response, "model", None)
        if not isinstance(embedding_model, str) or not embedding_model:
            raise RuntimeError("Qwen embedding response did not record a model ID")
        return LiveModelSmokeResult(
            routing_model_id=routing.actual_model,
            completion_model_id=completion.actual_model,
            embedding_model_id=embedding_model,
            embedding_dimensions=len(vector),
        )
    finally:
        await deepseek.close()
        await qwen.close()


def _credential(value: Any, name: str) -> str:
    secret = value.get_secret_value() if value is not None else ""
    if not secret.strip() or secret.startswith("replace-with-"):
        raise LiveModelConfigurationError(
            f"{name} is required when pytest -m live_model is selected"
        )
    return secret


def _snapshot(settings: Settings) -> RuntimeConfigSnapshot:
    return RuntimeConfigSnapshot(
        app_version="live-smoke",
        graph_version="live-smoke",
        prompt_version="live-smoke",
        main_model_id="deepseek-v4-pro",
        light_model_id="deepseek-v4-flash",
        embedding_model="text-embedding-v3",
        embedding_dimensions=1024,
        reranker_version=settings.reranker_model,
        retrieval_config_version="live-smoke",
        index_generation=settings.index_generation,
        memory_config_version="live-smoke",
    )


__all__ = ["LiveModelConfigurationError", "LiveModelSmokeResult", "run_live_model_smoke"]
