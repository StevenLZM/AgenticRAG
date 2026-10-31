"""Typed supervisor decisions over compact durable feedback, not child histories."""

from typing import Any, Literal

from pydantic import Field, model_validator

from agentic_rag.orchestration.agents import Execution
from agentic_rag.orchestration.contracts import Contract, DispatchPlan, Gap, Key, Text


class RouteDecision(Contract):
    kind: Literal["fact", "research", "analysis", "clarify"]
    reason: Text
    requires_calculation: bool
    multiple_targets: bool
    clarification: str = Field(default="", max_length=2000)

    @property
    def fast(self) -> bool:
        return self.kind == "fact" and not self.requires_calculation and not self.multiple_targets


class SupervisorDecision(Contract):
    action: Literal["dispatch", "continue", "answer", "clarify"]
    reason: Text
    plan: DispatchPlan | None = None
    group_key: Key | None = Field(default=None, description="Only for action=continue. Must be null for dispatch, answer and clarify.")
    task_keys: tuple[Key, ...] = Field(default=(), max_length=16, description="Only deficient retrieval task keys for action=continue. Must be empty for dispatch, answer and clarify.")
    gaps: tuple[Gap, ...] = Field(default=(), max_length=32, description="Concrete retrieval gaps for action=continue. Must be empty for dispatch, answer and clarify.")
    answer_mode: Literal["full", "partial"] = "full"
    question: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def single_action(self) -> "SupervisorDecision":
        if (self.action == "dispatch") != (self.plan is not None):
            raise ValueError("dispatch requires exactly one plan")
        if self.action == "continue":
            if self.group_key is None or not self.task_keys or not self.gaps:
                raise ValueError("continue requires a group, selected tasks and concrete gaps")
        elif self.group_key or self.task_keys or self.gaps:
            raise ValueError("only continuation supplies gap dispatch fields")
        if (self.action == "clarify") != bool(self.question):
            raise ValueError("clarify requires one question")
        if self.action != "answer" and self.answer_mode != "full":
            raise ValueError("only answer can select partial mode")
        return self


class Supervisor:
    async def route(self, question: str, execution: Execution) -> RouteDecision:
        return await execution.structured("router", {
            "question": question,
            "instructions": "Only a single factual lookup without analysis/comparison/arithmetic may use fact. Comparisons, trends, derived metrics and reports require analysis. Use clarify for unresolved scope.",
        }, RouteDecision, light=True)

    async def decide(self, state: dict[str, Any], feedback: list[dict[str, Any]], execution: Execution) -> SupervisorDecision:
        tasks = {
            key: {k: v for k, v in task.items() if k in {
                "spec", "state", "acceptance", "usage", "supported_criteria", "revision", "input_version", "round_id", "no_progress_rounds",
            }} for key, task in state["tasks"].items()
        }
        return await execution.structured("supervisor", {
            "question": state["question"], "tasks": tasks, "groups": state["groups"],
            "budget": state["budget"], "feedback": feedback, "accepted": state["accepted"],
            "instructions": (
                "Dispatch contracts must specify objective, acceptance criteria, output, tools/source guidance and boundaries. "
                "Use the schema's default model-call and cumulative output-token budgets unless explicit task constraints require otherwise; leave room for multiple rounds and repairs. "
                "output.max_items counts individual findings plus calculations, not reports or companies; prefer its default and allow room for every required calculation and explanatory finding. "
                "Independent retrieval tasks join an explicit group; no dependency within that group. "
                "Group criterion IDs must reuse the criteria assigned to member tasks, not new summary IDs. "
                "Tasks/groups are current durable state: never redispatch existing task keys. Earlier rejected decisions are history; act on the latest grades and remaining budgets. "
                "Dispatch uses plan only; group_key=null, task_keys=[], gaps=[]. Set criteria to verifiable requirements, not guessed numerical answers. "
                "Analytical tasks use an analysis agent with retrieval dependencies; pure arithmetic entirely specified by the user goes directly to an analysis agent with no retrieval dependency. "
                "Analysis dependencies may name only retrieval tasks. For comparisons, give one analysis task all relevant retrieval dependencies; never chain analysis tasks. "
                "Analysis consuming retrieved inputs needs read_evidence and calculate for arithmetic; user-only arithmetic needs calculate. "
                "On fast-path promotion, reuse existing fast candidates by setting the new group retained_fast_criteria to its relevant criterion IDs, and dispatch only gap retrieval tasks. "
                "Use continue only for the deficient group members with concrete gap directions; retain successes. "
                "A continuation must use an existing retrieval group key; never invent a group for an analysis task. "
                "Answer only from accepted evidence and completed analyses; choose partial if required evidence is missing. "
                "Do not calculate, expand hard source scope, or access child conversations."
            ),
        }, SupervisorDecision)
