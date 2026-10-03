"""NAT planning tools over the shared, constraint-preserving calculation service."""
from __future__ import annotations
import asyncio
import logging
from collections.abc import AsyncGenerator
from pydantic import BaseModel, Field, model_validator
from nat.builder.builder import Builder
from nat.builder.function import FunctionGroup
from nat.cli.register_workflow import register_function_group
from nat.data_models.function import FunctionGroupBaseConfig

from cualign import sandbox_compat
from cualign.agent import nim_stream_patch, react_history_patch, react_patch, reviewer  # register the bounded reviewer
from cualign.server import rails_middleware  # noqa: F401  register the Guardrails workflow middleware
from cualign.agent.context import CURRENT_RUN, track_task
from cualign.agent import steps
from cualign.core import limits as L, planner
from cualign.core import request_check
from cualign.core import skills as S
from cualign.core.constraints import Constraints, ConstraintPatch, ToolConstraintPatch, reason_ko
from cualign.core.service import PlanningService
from cualign.core.store import OUT_DIR, STORE

logger = logging.getLogger(__name__)

react_patch.apply()
react_history_patch.apply()
nim_stream_patch.apply()
sandbox_compat.apply()  # OpenShell: route aiohttp (NIM async client) through the sandbox proxy


class NoInput(BaseModel):
    unused: str = ""


class CaseInput(BaseModel):
    case_id: str | None = None


class StrategyInput(BaseModel):
    strategy: str = Field(description="expansion | ipr | expansion_ipr | extraction")


class PlanInput(BaseModel):
    target_id: str


class PlanIdInput(BaseModel):
    plan_id: str


class CompareInput(BaseModel):
    allowed: list[str] | None = Field(default=None, description="Optional strategy subset; confirmed constraints always apply")


class SkillInput(BaseModel):
    name: str = Field(description="skill name, e.g. cualign-clinical-rules")


class ContextPreload(BaseModel):
    """What the `cuAlign server context` system message carries in advance, so the agent does not spend a model
    round-trip per preparatory tool call (#48). Off by default: an absent block changes nothing."""
    case: bool = False        # teeth, n_teeth, crowding_mm, unsupported — what load_case returns
    limits: bool = False      # what clinical_limits returns
    skill: str | None = None  # this skill's SKILL.md — what load_skill returns for that name


class CuAlignToolConfig(FunctionGroupBaseConfig, name="cualign"):
    include: list[str] = Field(default_factory=lambda: [
        "clinical_limits", "list_cases", "load_case", "get_constraints", "set_constraints",
        "propose_target", "plan_stages", "validate", "compare_strategies", "select_plan", "export_stl", "get_plan",
        "load_skill"])
    # The skills load_skill may read, with OpenClaw's agent-allowlist rules: None = every skill under
    # workspace/skills, [] = none, a list = exactly these. The workspace also holds the OpenClaw desk's skill.
    skills: list[str] | None = None
    context_preload: ContextPreload = Field(default_factory=ContextPreload)

    @model_validator(mode="after")
    def _preload_skill_allowed(self):
        """A preloaded skill outside the allowlist would hand the agent what load_skill refuses; fail at startup."""
        s = self.context_preload.skill
        if s and self.skills is not None and s not in self.skills:
            raise ValueError(f"context_preload.skill {s!r} is not in skills {self.skills}")
        return self


def limits_view() -> dict:
    """The clinical_limits tool result; also preloaded into the server context."""
    return {"max_linear_mm_per_aligner": L.MAX_LINEAR_PER_ALIGNER,
            "ipr_max_mm_per_surface": L.IPR_PER_SURFACE,
            "max_expansion_mm_per_side": L.MAX_EXPANSION_PER_SIDE,
            "wear_days_per_aligner": L.WEAR_DAYS, "strategies": list(L.STRATEGIES)}


def constraints_json(c) -> dict:
    """What the model reads for a Constraints: the fields (Universal), the tooth fields as set_constraints takes them
    (teeth_fdi, FDI) and the dentist's 조건 line (conditions_ko, FDI) to copy into the answer as it is (#113)."""
    return {**c.model_dump(mode="json"), "teeth_fdi": c.teeth_fdi(), "conditions_ko": c.describe_ko()}


def case_view(case) -> dict:
    """The case summary the load_case tool returns (without id and constraints); also preloaded into the server context."""
    return {"teeth": case.ids, "n_teeth": len(case.ids), "crowding_mm": planner.crowding_mm(case),
            "unsupported": planner.unsupported_reasons(case)}


def context_preload(settings: ContextPreload):
    """The `preload` callable for plan_events.open_run, built from the workflow config. None when nothing is enabled.
    The skill is read once here, so a wrong name fails at startup rather than as a 400 on the first request."""
    skill = S.read_skill(settings.skill) if settings.skill else None
    if not (settings.case or settings.limits or skill):
        return None

    def preload(cid, case, constraints) -> dict:
        """Fail-open per item: a view that raises is left out with a warning, and the agent reads it with the tool
        as before (the instructions say to call the tool for what the context lacks). The request itself still
        gets its 400-or-200 from open_run's own checks, never a 500 from here."""
        extra = {}
        views = [("case", lambda: case_view(case)) if settings.case else None,
                 ("limits", limits_view) if settings.limits else None,
                 ("skill", lambda: skill) if skill else None]
        for name, view in filter(None, views):
            try:
                extra[name] = view()
            except Exception as e:  # noqa: BLE001  the tool path reports the same error to the agent
                logger.warning("cuAlign: context preload of %s skipped for case %s (%s: %s)", name, cid, type(e).__name__, e)
        return extra
    return preload


def current_case(case_id=None):
    run = CURRENT_RUN.get()
    if run:
        if run.closed or (case_id is not None and case_id != run.case_id):
            raise ValueError("use the case selected in the current request")
        case_id = run.case_id
    return STORE.load_case(case_id)


def constraints_for(case_id):
    run = CURRENT_RUN.get()
    return run.constraints if run else STORE.constraints_for(case_id)


def owned_plan(plan_id, *, generated=False):
    p = STORE.plans[plan_id]
    run = CURRENT_RUN.get()
    if run and (run.closed or p["case_id"] != run.case_id or
                plan_id not in (run.plan_ids if generated else run.plan_ids | {run.base_plan_id})):
        raise ValueError("plan is not part of the current request")
    return p


def summary(pid):
    p = STORE.plan_json(pid)
    return {k: v for k, v in p.items() if k != "stages"}


def check_request(run, pids):
    """The turn's sentence against the change it made to the conditions (core/request_check), on every plan it makes:
    a mismatch blocks approval. Not only the selected one — the screen's plan menu switches to another candidate of a
    comparison without select_plan, and its review and approval are asked by hand (Codex review 2026-10-03)."""
    if run is None or run.text is None or run.opened is None:
        return                                    # MCP and tests have no sentence
    for pid in pids:
        c = STORE.plans[pid]["constraints"]
        c = c if isinstance(c, Constraints) else Constraints.model_validate(c)
        STORE.set_request_check(pid, request_check.check(run.text, run.opened, c))


@register_function_group(config_type=CuAlignToolConfig)
async def cualign(config: CuAlignToolConfig, _builder: Builder) -> AsyncGenerator[FunctionGroup, None]:
    group = FunctionGroup(config=config)
    service = PlanningService(STORE)

    async def _clinical_limits(inp: NoInput) -> dict:
        """PoC 계산 한계와 단위를 읽는다. 서버 문맥에 limits 가 있으면 다시 부르지 않는다. 임상 판단이나 의사 승인이 아니다."""
        return limits_view()

    async def _list_cases(inp: NoInput) -> dict:
        """사용 가능한 케이스와 현재 선택 케이스를 읽는다."""
        run = CURRENT_RUN.get()
        return {"cases": STORE.available_cases(), "active": run.case_id if run else STORE.active_case}

    async def _load_case(inp: CaseInput) -> dict:
        """화면에서 선택한 케이스를 읽는다. 서버 문맥에 case 가 있으면 다시 부르지 않는다. 다른 케이스로 임의 전환할 수 없다.
        unsupported 가 비어 있지 않으면 이 케이스는 계획하지 말고 그 이유를 의사에게 그대로 전한다."""
        cid, case = current_case(inp.case_id)
        return {"case_id": cid, **case_view(case), "constraints": constraints_json(constraints_for(cid))}

    async def _get_constraints(inp: NoInput) -> dict:
        """확정 조건을 읽는다. 서버 문맥의 constraints 와 같으니 그것이 있으면 다시 부르지 않는다. 비교·수정 시 이 조건을 유지한다."""
        cid, _ = current_case()
        return constraints_json(constraints_for(cid))

    async def _set_constraints(inp: ToolConstraintPatch) -> dict:
        """사용자가 명시적으로 변경한 조건만 전달한다. null과 생략은 모두 유지를 뜻한다. []는 치아 목록 해제, clear_stage_cap:true는 기간 상한 해제. IPR 한도는 면당 mm.
        치아 번호(extraction·lock·ipr_exclude·ipr_surfaces)는 의사가 말한 FDI 번호 그대로 넣는다. 변환은 서버가 한다."""
        cid, case = current_case()
        run = CURRENT_RUN.get()
        current = constraints_for(cid)
        try:
            c = current.patched(inp.changes())
            c.check_case(case.ids)
        except ValueError as e:
            # Extraction without teeth, a tooth that cannot be extracted or is not in the case, stage_cap with
            # clear_stage_cap, ...: a normal result, like the rejection below (raising makes the agent retry), so the
            # agent asks the dentist instead (#56). The request is echoed as sent, never re-derived (that could raise).
            return {**constraints_json(current), "rejected": inp.model_dump(mode="json", exclude_none=True),
                    "note": reason_ko(e) + " 조건은 바꾸지 않았습니다. 의사에게 확인한 뒤 다시 설정하세요."}
        # Re-stating the same conditions mid-loop is harmless; only a real change after targets
        # exist would make the computed plans disagree with the stored constraints.
        if c == current:
            return constraints_json(c)
        if run and run.target_ids:
            # Refuse the change, but as a normal result: raising here reads as "you called me
            # wrong", so the agent retries with new arguments until it runs out of iterations.
            # The conditions are unchanged either way — this only stops the retry loop and lets
            # the agent tell the dentist which change it could not apply.
            return {**constraints_json(current), "rejected": inp.changes(),
                    "note": "이번 요청의 조건은 이미 확정됐습니다. 현재 조건으로 계획을 마치고, "
                            "반영하지 못한 변경은 최종 답변에서 의사에게 알리세요."}
        c.check_case(case.ids)
        STORE.case_constraints[cid] = c
        if run:
            run.constraints = c
        return constraints_json(c)

    async def _propose_target(inp: StrategyInput) -> dict:
        """확정 조건으로 목표를 생성한다. 조건 변경은 먼저 set_constraints로 처리한다."""
        cid, _ = current_case()
        tid = service.target(cid, inp.strategy, constraints_for(cid))
        run = CURRENT_RUN.get()
        if run:
            run.target_ids.add(tid)
            run.last_target_id = tid
        t = STORE.targets[tid]
        return {"target_id": tid, "case_id": cid, "constraints": constraints_json(t["constraints"]), **t["info"]}

    async def _plan_stages(inp: PlanInput) -> dict:
        """목표에 저장된 조건으로 단계 생성과 검증을 수행한다. 수정은 선택된 부모 계획을 보존한다."""
        run = CURRENT_RUN.get()
        if run and (run.closed or inp.target_id not in run.target_ids | {run.inherited_target_id}):
            raise ValueError("target is not part of this request")
        pid = service.stages(inp.target_id, run.base_plan_id if run else None)
        if run:
            run.plan_ids.add(pid)
        check_request(run, [pid])
        return summary(pid)

    async def _validate(inp: PlanIdInput) -> dict:
        """계획에 저장된 동일 조건으로 재검증한다. 조건을 생략해 상한을 지울 수 없다."""
        owned_plan(inp.plan_id)
        service.validate(inp.plan_id)
        return summary(inp.plan_id)

    async def _compare_strategies(inp: CompareInput) -> dict:
        """확정된 비발치·고정·IPR·기간·순서를 모두 유지하면서 허용 전략을 비교한다."""
        cid, _ = current_case()
        run = CURRENT_RUN.get()
        ids = service.compare(cid, constraints_for(cid), inp.allowed, run.base_plan_id if run else None,
                              stop=(lambda: run.cancelled) if run else None)
        if run:
            run.plan_ids.update(ids)
            run.target_ids.update(STORE.plans[pid]["target_id"] for pid in ids)
            run.compared = True
        check_request(run, ids)
        c = constraints_for(cid)   # plans carry the same constraints (golden set tool contract); the FDI line beside them
        return {"case_id": cid, "constraints": c.model_dump(mode="json"), "conditions_ko": c.describe_ko(),
                "plans": [summary(pid) for pid in ids]}

    async def _select_plan(inp: PlanIdInput) -> dict:
        """최종 제시할 계획을 선택한다. 후보 생성 순서와 무관하게 이 계획이 3D·카드·다운로드에 표시된다. 이후 reviewer를 호출한다."""
        owned_plan(inp.plan_id, generated=True)
        p = STORE.plans[inp.plan_id]
        STORE.require_current_input(p["case_id"], p.get("input_revision"))   # not a plan of an earlier scan revision
        run = CURRENT_RUN.get()
        if run:
            if run.selected_plan_id and run.selected_plan_id != inp.plan_id:
                raise ValueError("a final plan is already selected in this request")
            run.selected_plan_id = inp.plan_id     # it carries request_check from when the turn made it
        return {"type": "plan_selected", **summary(inp.plan_id)}

    async def _export_stl(inp: PlanIdInput) -> dict:
        """의사가 현재 버전을 승인한 계획만 ZIP으로 내보낸다. 에이전트는 승인할 수 없다.
        ZIP에는 치아별 STL과, 잇몸 스캔이 있으면 단계별 프린트용 상악 모형(kind=full_arch_model)이 들어간다."""
        p = owned_plan(inp.plan_id)
        STORE.require_approved(inp.plan_id)
        _, case = STORE.load_case(p["case_id"])
        path = OUT_DIR / "stl" / f"{inp.plan_id}.zip"
        planner.export_zip(case, p["stages"], str(path))
        models = planner.export_print_models(case, p["stages"], str(path), p["case_id"])
        out = {"plan_id": inp.plan_id, "download_url": f"/api/plans/{inp.plan_id}/stl.zip",
               "kind": "full_arch_model" if models["status"] == "ok" else "per_tooth", "n_files": models["n_files"]}
        if models["status"] == "skipped":
            out["print_models_note"] = models["reason_ko"]
        elif models["failed"]:
            out["print_models_note"] = f"프린트용 모형 {len(models['failed'])}개 단계 실패: " + \
                ", ".join(f"{f['stage']}단계({f['reason']})" for f in models["failed"])
        return out

    async def _get_plan(inp: PlanIdInput) -> dict:
        """읽기 전용: 계획 조건·부모 이력·규칙 위반·검토·승인 상태를 조회한다."""
        owned_plan(inp.plan_id)
        return summary(inp.plan_id)

    async def _load_skill(inp: SkillInput) -> dict:
        """설치된 Agent Skill 의 지시문(workspace/skills/<name>/SKILL.md)을 읽는다. 서버 문맥에 skill 이 있으면 다시 부르지 않는다.
        없으면 계획·비교를 시작할 때 cualign-clinical-rules 를 한 번 읽고 따른다."""
        return S.read_skill(inp.name, allowed=config.skills)

    fns = {"clinical_limits": _clinical_limits, "list_cases": _list_cases, "load_case": _load_case,
           "get_constraints": _get_constraints, "set_constraints": _set_constraints,
           "propose_target": _propose_target, "plan_stages": _plan_stages, "validate": _validate,
           "compare_strategies": _compare_strategies, "select_plan": _select_plan,
           "export_stl": _export_stl, "get_plan": _get_plan,
           "load_skill": _load_skill}
    for name in config.include:
        group.add_function(name=name, fn=step_gated(name, fns[name]), description=fns[name].__doc__)
    yield group


# The tools whose body is seconds of planner work (000001: a comparison ~9 s). On the event loop they held every other
# request, 건너뛰기's /replay too, until they ended; in a worker thread the loop stays free and the turn can be cut.
OFF_LOOP = {"propose_target", "plan_stages", "compare_strategies", "validate"}


def step_gated(name, fn):
    """`fn` refused with steps.refusal when the current turn's step does not reach this tool (agent/steps.py): the
    model gets a normal result saying so and stops, instead of running the whole plan in a setup or target turn.
    The calling task is noted as the turn's (context.track_task), so cutting the turn cancels it."""
    async def gated(inp):
        run = CURRENT_RUN.get()
        if run is not None and not steps.allowed(run.step, name):
            return steps.refusal(run.step, name)
        track_task()
        if name in OFF_LOOP and run is not None:
            return await asyncio.to_thread(_run_off_loop, run, fn, inp)
        return await fn(inp)
    gated.__annotations__ = dict(fn.__annotations__)   # NAT reads the input schema from the annotation
    gated.__name__, gated.__doc__ = fn.__name__, fn.__doc__
    return gated


def _run_off_loop(run, fn, inp):
    """A tool's body in a worker thread (the request context comes along: to_thread copies it). A thread cannot be
    cancelled, so a turn cut meanwhile gets here late: what this call made is dropped, not left in the case's plans."""
    plans, targets = set(run.plan_ids), set(run.target_ids)
    try:
        return asyncio.run(fn(inp))
    finally:
        if run.cancelled:
            STORE.discard(run.plan_ids - plans, run.target_ids - targets)
            logger.info("cuAlign: turn %s was cut while a tool ran; its result is dropped", run.request_id)
