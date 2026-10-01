import importlib.util
import json


def module():
    assert importlib.util.find_spec("agentic_rag.query.routing_context") is not None
    from agentic_rag.query import routing_context
    return routing_context


def test_bound_history_limits():
    m = module()
    turns = [m.RoutingTurn(id=str(i), role="user", content=str(i) * 2000) for i in range(8)]
    bounded = m.bound_history(turns)
    assert len(bounded) <= 6
    assert sum(len(t.content) for t in bounded) == 8000
    assert [t.id for t in bounded] == ["4", "5", "6", "7"]
    long = m.bound_history([m.RoutingTurn(id="x", role="user", content="a" * 9000)])
    assert long[0].truncated and len(long[0].content) == 8000


def test_reasoning_context_is_run_scoped():
    m = module()
    state = {"run_id": "new", "scope": {"user_id": "u"}, "request": {"question": "他呢"},
             "routing_owner_run_id": "old", "route_assessment": {"normalized_query": "秘密"},
             "routing_context": {"run_id": "old", "history": []}}
    assert m.reasoning_question(state) == "他呢"
    state.update(routing_owner_run_id="new", routing_context=None)
    assert json.loads(m.reasoning_question(state))["original_question"] == "他呢"
    assert state["request"]["question"] == "他呢"


def test_fresh_run_has_a_valid_empty_research_pack():
    from tests.unit.query.test_router_fast_path import _initial_state
    from agentic_rag.query.state import snapshot_from_state
    from agentic_rag.query.research_loop import _packed_validation_error
    state = _initial_state()
    fresh = module().fresh_routing_state(state)
    assert _packed_validation_error(fresh["packed_context"], snapshot_from_state(state)) is None
