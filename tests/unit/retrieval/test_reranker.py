"""Unit coverage for bounded, degradable cross-encoder reranking."""

import asyncio
import threading
import time
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from typing import Any

import pytest

from agentic_rag.retrieval.models import ChildHit
from agentic_rag.retrieval.reranker import Reranker, RerankerUnavailable


def hit(child_id: str) -> ChildHit:
    """Build a small but valid retrieval hit."""
    return ChildHit(
        child_id=child_id,
        parent_id=f"parent-{child_id}",
        user_id="user-1",
        document_id="document-1",
        document_version_id="version-1",
        content=f"content for {child_id}",
        ast_locator="segment-0",
        lane="dense",
        lane_rank=1,
        score=1.0,
    )


class FakeCrossEncoder:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.raise_timeout = False
        self.raise_unavailable = False
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls.append(pairs)
        if self.raise_timeout:
            time.sleep(0.05)
        if self.raise_unavailable:
            raise RerankerUnavailable("model unavailable")
        return self.scores[: len(pairs)]


class BlockingCrossEncoder:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.entered.set()
        self.release.wait(timeout=1)
        with self._lock:
            self.active -= 1
        return [1.0] * len(pairs)


class TrackingExecutor(Executor):
    def __init__(self) -> None:
        self.shutdown_calls = 0

    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Future[Any]:
        future: Future[Any] = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as error:
            future.set_exception(error)
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        self.shutdown_calls += 1


@pytest.fixture
def candidates() -> list[ChildHit]:
    return [hit("a"), hit("b"), hit("c")]


@pytest.fixture
def service() -> Reranker:
    return Reranker(
        FakeCrossEncoder([0.2, 0.9, 0.5]),
        model_version="fake-cross-encoder-v1",
        timeout_seconds=0.01,
        max_concurrent_reranks=1,
    )


async def test_reranker_orders_hits_by_cross_encoder_score(
    service: Reranker, candidates: list[ChildHit]
) -> None:
    result = await service.rerank("query", candidates, limit=2)

    assert [item.child_id for item in result.hits] == ["b", "c"]
    assert [item.rerank_score for item in result.hits] == [0.9, 0.5]
    assert [item.retrieval_score for item in result.hits] == [1.0, 1.0]
    assert result.degraded is False
    assert result.model_version == "fake-cross-encoder-v1"
    assert service.model.calls == [[("query", item.content) for item in candidates]]


async def test_reranker_timeout_returns_rrf_order(
    service: Reranker, candidates: list[ChildHit]
) -> None:
    service.model.raise_timeout = True

    result = await service.rerank("q", candidates, limit=2)

    assert result.degraded is True
    assert result.hits == candidates[:2]


async def test_reranker_unavailable_returns_rrf_order(
    service: Reranker, candidates: list[ChildHit]
) -> None:
    service.model.raise_unavailable = True

    result = await service.rerank("q", candidates, limit=2)

    assert result.degraded is True
    assert result.hits == candidates[:2]


async def test_reranker_propagates_unexpected_model_contract_errors(
    candidates: list[ChildHit],
) -> None:
    class UnexpectedFailureModel:
        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            raise ValueError("unexpected score contract failure")

    service = Reranker(UnexpectedFailureModel(), model_version="fake-cross-encoder-v1")

    with pytest.raises(ValueError, match="unexpected score contract failure"):
        await service.rerank("q", candidates)

    service.close()


async def test_reranker_limits_model_input_to_thirty_candidates() -> None:
    candidates = [hit(str(index)) for index in range(31)]
    model = FakeCrossEncoder([float(index) for index in range(30)])
    service = Reranker(model, model_version="fake-cross-encoder-v1")

    result = await service.rerank("q", candidates, limit=30)

    assert len(model.calls[0]) == 30
    assert len(result.hits) == 30


async def test_timed_out_prediction_keeps_the_concurrency_slot_until_it_finishes(
    candidates: list[ChildHit],
) -> None:
    model = BlockingCrossEncoder()
    executor = ThreadPoolExecutor(max_workers=2)
    service = Reranker(
        model,
        model_version="fake-cross-encoder-v1",
        timeout_seconds=0.01,
        max_concurrent_reranks=1,
        executor=executor,
    )

    first = asyncio.create_task(service.rerank("first", candidates, limit=1))
    assert await asyncio.to_thread(model.entered.wait, 1)
    first_result = await first
    second = asyncio.create_task(service.rerank("second", candidates, limit=1))

    await asyncio.sleep(0.02)
    assert model.max_active == 1
    assert model.active == 1
    assert first_result.degraded is True
    assert first_result.hits == candidates[:1]

    model.release.set()
    await second
    assert model.max_active == 1
    executor.shutdown(wait=True)


async def test_reranker_closes_only_its_owned_executor() -> None:
    owned_service = Reranker(FakeCrossEncoder([1.0]), model_version="owned-v1")
    external_executor = TrackingExecutor()
    external_service = Reranker(
        FakeCrossEncoder([1.0]),
        model_version="external-v1",
        executor=external_executor,
    )

    await owned_service.aclose()
    external_service.close()

    assert owned_service.closed is True
    assert external_service.closed is True
    with pytest.raises(RuntimeError, match="cannot schedule new futures"):
        owned_service._executor.submit(lambda: None)
    assert external_executor.shutdown_calls == 0
    assert external_executor.submit(lambda: "still caller-owned").result() == "still caller-owned"
