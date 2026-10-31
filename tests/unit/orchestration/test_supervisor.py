from importlib import import_module

import pytest
from pydantic import ValidationError

from tests.unit.orchestration.test_agents import Execution


async def test_supervisor_receives_task_contracts_and_feedback_without_private_context():
    module = import_module("agentic_rag.orchestration.supervisor")
    data = {"question": "q", "tasks": {"r1": {"spec": {"objective": "q"}, "state": "waiting_feedback",
        "context_ref": "private-1", "result_refs": ["private-1"], "context": {"secret": "x"}}},
        "groups": {}, "budget": {}, "accepted": {}}
    execution = Execution({"action": "clarify", "reason": "source is ambiguous", "question": "Which year?"})
    await module.Supervisor().decide(data, [], execution)
    assert "private-1" not in str(execution.calls)
    assert "secret" not in str(execution.calls)


def test_actions_do_not_admit_generic_tool_calls_or_mixed_mutations():
    module = import_module("agentic_rag.orchestration.supervisor")
    with pytest.raises(ValidationError):
        module.SupervisorDecision.model_validate({"action": "calculate", "expression": "1+2", "reason": "x"})
    with pytest.raises(ValidationError):
        module.SupervisorDecision.model_validate({"action": "answer", "reason": "ready", "question": "conflicting action"})
