"""Request-local state shared by NAT tasks through ContextVar inheritance."""
import asyncio
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any
from cualign.core.constraints import Constraints


@dataclass
class PlanRun:
    request_id: str
    case_id: str
    base_plan_id: str | None
    constraints: Constraints
    step: str | None = None  # setup | target | stages: how far this turn goes (agent/steps.py); None = stages
    text: str | None = None  # the turn's sentence (plan_events.user_text); None outside /chat/stream: no request check
    opened: Constraints | None = None  # the conditions the turn opened with (UI form applied), before set_constraints
    last_target_id: str | None = None  # the target propose_target made last (step_done target)
    inherited_target_id: str | None = None  # the target a previous target turn made (Store.flow): plan_stages may take it
    plan_ids: set[str] = field(default_factory=set)
    target_ids: set[str] = field(default_factory=set)
    selected_plan_id: str | None = None
    compared: bool = False  # compare_strategies ran, so the answer describes several plans (rails_middleware, #91)
    review_attempts: int = 0
    review_started: float | None = None
    review_busy: bool = False
    closed: bool = False
    rails: str | None = None  # passed | flagged | blocked | error | off, set by the rails middleware
    refused: bool = False  # the rails middleware replaced the answer with its refusal
    # why: "pii" (the last message holds a personal identifier), "pii_context" (an earlier message or the case), "rails"
    refused_kind: str | None = None
    redacted: str | None = None  # "pii": the last message with its personal identifiers taken out, for the screen
    error: dict | None = None  # {"kind": "nim_overload" | "workflow_error", "message": ...} when the workflow raised
    # 건너뛰기 or a closed stream cut the turn (server/plan_events.cancel): its tasks are cancelled, its plans dropped,
    # and whatever it still produces is not offered
    cancelled: bool = False
    tasks: set = field(default_factory=set, repr=False, compare=False)  # the workflow's tasks (track_task)


CURRENT_RUN: ContextVar[PlanRun | None] = ContextVar("cualign_plan_run", default=None)
# The streamed turn's reasoning relay (rails_middleware.ReasoningRelay): react_patch feeds it the planner's reasoning
# as it arrives, while the answer is held for the output rail.
REASONING: ContextVar[Any] = ContextVar("cualign_reasoning", default=None)


def track_task() -> None:
    """Note the running task as one of the current turn's workflow tasks. NAT runs the workflow in a task of its own
    and does not cancel it when the client goes (response_helpers.generate_streaming_response); plan_events.cancel
    cancels the tasks noted here."""
    run = CURRENT_RUN.get()
    task = asyncio.current_task()
    if run is not None and task is not None:
        run.tasks.add(task)
