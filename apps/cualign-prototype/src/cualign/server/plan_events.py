"""Attach verified plan selection to NAT's existing chat SSE route."""
import asyncio
import json
import logging
from uuid import uuid4

from typing import Literal

from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from cualign.agent import steps
from cualign.core import recorded
from cualign.agent.context import CURRENT_RUN, PlanRun
from cualign.core.constraints import ConstraintPatch, ExtractionTeethNeeded, reason_ko
from cualign.core.fdi import label
from cualign.core.store import STORE

logger = logging.getLogger(__name__)
# One line per step for the server context: what this turn does and where it stops (the tools past it are refused).
STEP_RULE_KO = {
    "setup": "이번 턴은 셋업 단계: 처방 문장을 조건으로 해석해 set_constraints 까지만 하고, 계획·목표 배열은 만들지 않는다. "
             "끝나면 조건을 한 문장으로 요약하고 「이대로 목표 배열을 만들까요?」로 되묻는다.",
    "target": "이번 턴은 목표 배열 단계: propose_target 까지만 하고 단계로 나누지 않는다. 끝나면 목표 배열의 요지(전략, 공간, "
              "총생)를 한 문장으로 말하고 「단계로 나눌까요?」로 되묻는다.",
    "stages": "이번 턴은 단계 단계: plan_stages -> select_plan -> reviewer 까지 끝까지 한다(조건 변경·비교 요청도 이 턴). "
              "문맥에 target_id 가 있으면 그 목표 배열을 그대로 plan_stages 에 넣는다(조건이 바뀐 요청이면 set_constraints 뒤 "
              "propose_target 부터 새로).",
}


class ChatContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(default_factory=lambda: uuid4().hex, min_length=1, max_length=128)
    case_id: str
    base_plan_id: str | None = None
    constraints: ConstraintPatch = Field(default_factory=ConstraintPatch)
    # how far this turn goes (agent/steps.py). The screen sends it as a top-level body field `step`, beside `cualign`.
    step: Literal["setup", "target", "stages"] = steps.DEFAULT_STEP


def open_run(ctx: ChatContext, store=None, preload=None) -> tuple[PlanRun, dict]:
    """Apply the UI form to the case and open the request's PlanRun; also returns the system message that tells the
    agent the case, parent plan and confirmed constraints. `preload(cid, case, constraints)` (register.context_preload,
    from the workflow's `context_preload` block) adds what the agent would otherwise fetch with a tool call each: the
    case summary, the clinical limits and the skill text (#48). Raises ValueError/KeyError/TypeError on bad input."""
    store = store or STORE
    cid, case = store.load_case(ctx.case_id)
    if ctx.base_plan_id and ctx.base_plan_id in store.plans and store.previous_calculation(ctx.base_plan_id):
        raise ValueError("이전 계산의 계획은 기준 계획이 될 수 없습니다")   # another core version's plan (answer-polish (10))
    constraints = store.constraints_for(cid, ctx.base_plan_id).patched(ctx.constraints.changes())
    constraints.check_case(case.ids)
    store.case_constraints[cid] = constraints
    run = PlanRun(ctx.request_id, cid, ctx.base_plan_id, constraints, step=ctx.step, opened=constraints)
    context = {"case_id": cid, "base_plan_id": ctx.base_plan_id, "constraints": constraints.model_dump(mode="json"),
               "teeth_fdi": constraints.teeth_fdi(),   # the tooth fields as set_constraints takes them (FDI)
               "conditions_ko": constraints.describe_ko(),   # the answer's 조건 line, FDI, written by the server (#113)
               "step": ctx.step, "step_ko": STEP_RULE_KO[ctx.step]}   # how far this turn goes (workspace/AGENTS.md «Steps»)
    flow = store.flow.get(cid) if hasattr(store, "flow") else None
    if flow and flow.get("target_id") and ctx.step == "stages" and flow["target_id"] in store.targets:
        context["target_id"] = run.inherited_target_id = flow["target_id"]   # the target turn's target: plan_stages may take it
        tinfo = store.targets[flow["target_id"]]["info"]
        if tinfo.get("source") == "manual":   # 직접 이동 (core/manual.py): a new propose_target would drop the dentist's moves
            context["target_manual_ko"] = (f"이 목표 배열은 의사가 화면에서 직접 옮긴 것이다({label(tinfo.get('manual_teeth') or [])}). "
                                           "propose_target 으로 새로 만들지 말고 target_id 를 그대로 plan_stages 에 넣는다.")
    if ctx.base_plan_id is not None:
        context["base_plan_ko"] = base_plan_ko(store, cid, ctx.base_plan_id)   # the plan on screen, and a revert (#90)
        rep = getattr(store, "replays", {}).get(cid)
        if rep and rep.get("plan_id") == ctx.base_plan_id:   # the plan on screen came from a recorded answer (core/recorded.py)
            context["replayed_ko"] = (f"[녹화된 답 재생] 직전 턴({rep['step_ko']})은 모델이 아니라 녹화된 답의 재생이었고 계획은 규칙 엔진이 "
                                      f"다시 계산했다. 이 계획을 기준으로 이어간다.")
    if preload is not None:
        context.update(preload(cid, case, constraints))
    return run, {"role": "system", "content": "cuAlign server context: " + json.dumps(context, ensure_ascii=False)}


STRATEGY_KO = {"expansion": "확장", "ipr": "IPR", "expansion_ipr": "확장 + IPR", "extraction": "발치", "manual": "수동 배치"}


def base_plan_ko(store, case_id: str, base_plan_id: str) -> str:
    """One Korean line about the plan the screen shows (base_plan_id) for the agent's context (#90). The screen sends
    the plan it shows; when newer plans of the case exist, the dentist went back to this one («이전 안으로 되돌리기»),
    and the line says so, so the next revision starts from the plan on screen and the newer plans are not "이전 안"."""
    p = store._record(base_plan_id)
    words = f"{STRATEGY_KO.get(p['strategy'], p['strategy'])} 전략, {p['info']['n_stages']}단계(약 {p['info']['months']}개월), " \
            f"{'규칙 위반 ' + str(len(p['violations'])) + '건' if p['violations'] else '규칙 통과'}"
    later = store.plan_ids_for(case_id)
    later = later[later.index(base_plan_id) + 1:] if base_plan_id in later else []
    if later:
        return (f"의사가 화면에서 이 계획({words})으로 되돌렸다. 그 뒤에 만든 계획 {len(later)}개는 버렸다: "
                f"다음 수정은 이 계획을 기준으로 하고, 버린 계획을 이전 안으로 부르지 않는다.")
    return f"화면에 보이는 계획: {words}. 수정 요청은 이 계획을 기준으로 한다."


def user_text(messages) -> str:
    """The last user message's text (the turn's sentence), "" when there is none."""
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            return c if isinstance(c, str) else " ".join(p.get("text", "") for p in c or [] if isinstance(p, dict))
    return ""


def resolve_step(data: dict, sent: str | None) -> str | None:
    """The turn's step (step-intent): a chip's step as sent (`step_source: "chip"`, or a chip's sentence), a free
    sentence's by its words (steps.turn_step) — the screen sends the step after its progress for a free sentence
    too. A client that sends no step keeps the default (every tool)."""
    source = data.pop("step_source", None)
    if sent is None:
        return None
    text = user_text(data.get("messages"))
    return steps.turn_step(text, sent, source == "chip" or recorded.chip(text))


def event(name, payload):
    return ("\n\nevent: " + name + "\ndata: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode()


def setup_view(store, case_id: str, constraints) -> dict:
    """What a finished setup step hands the screen (step_done setup and the setup replay): the conditions as
    plan_context has them (Universal), with `ipr_surfaces` filled in — the prescription as given, else the contacts the
    planner's uniform rule would cut (step flow (12): the setup view draws its IPR marks from this alone) — and the
    FDI line for display."""
    from cualign.core import planner
    _, case = store.load_case(case_id)
    data = constraints.model_dump(mode="json")
    data["ipr_surfaces"] = planner.ipr_surfaces_for(case, constraints)
    return {"constraints": data, "conditions_ko": constraints.describe_ko()}


def step_done(run: PlanRun, store=None) -> dict | None:
    """The `step_done` event of a finished turn (None when the step did not get done), recording it in Store.flow:
    setup -> the conditions the turn set (Constraints JSON as in plan_context, plus conditions_ko for display);
    target -> the target the turn made and its summary; stages -> the plan selected (plan_selected carries the rest).
    A failed, refused or cut turn reports nothing: the screen stays where it was."""
    store = store or STORE
    if run.error or run.refused or run.cancelled:
        return None
    if run.step == "setup":
        store.set_flow(run.case_id, "setup", constraints=run.constraints, target_id=None, plan_id=None)
        return {"step": "setup", **setup_view(store, run.case_id, run.constraints)}
    if run.step == "target":
        tid = run.last_target_id
        if tid is None or tid not in store.targets:
            return None
        store.set_flow(run.case_id, "target", constraints=run.constraints, target_id=tid, plan_id=None)
        return {"step": "target", "target_id": tid, "summary": steps.target_summary(store.targets[tid]["info"])}
    if run.selected_plan_id is None:
        return None
    tid = store.plans[run.selected_plan_id].get("target_id")
    store.set_flow(run.case_id, "stages", constraints=run.constraints, target_id=tid, plan_id=run.selected_plan_id)
    return {"step": "stages", "plan_id": run.selected_plan_id, "target_id": tid}


async def review_skipped(scope, plan_id) -> bool:
    """The agent selected a plan but ended the turn without calling the reviewer (KNOWN_ISSUES). The procedure must
    not hang on the model following it, so the server runs the same bounded review before offering the plan."""
    app = scope.get("app")
    review = getattr(getattr(app, "state", None), "cualign_review", None)
    if review is None or STORE.plans[plan_id]["review"]["status"] != "not_requested":
        return False
    logger.warning("cuAlign: the agent did not call the reviewer for %s; the server reviews it", plan_id)
    await review(plan_id)
    return True


# The turn streaming now, per case: 건너뛰기's /replay cuts it (api.replay_recorded) without waiting for it.
RUNS: dict[str, PlanRun] = {}


def cancel(run: PlanRun, why: str) -> bool:
    """Cut an agent turn: the dentist closed its stream (건너뛰기 aborts the fetch) or asked for the recorded answer.
    NAT leaves the workflow task running when the client goes (response_helpers.generate_streaming_response), so the
    tasks the turn noted (context.track_task) are cancelled here, which also closes their NIM requests. Its tools
    refuse from now on (run.closed), the plans and targets it made are dropped (never offered; left in place they
    would show in the case's plan list), and its closing events are not sent. Returns False when already cut."""
    if run.cancelled:
        return False
    run.cancelled = run.closed = True
    me = asyncio.current_task()
    tasks = [t for t in list(run.tasks) if t is not me and not t.done()]
    for t in tasks:
        t.cancel()
    made = [pid for pid in list(run.plan_ids) if pid != run.base_plan_id]
    STORE.discard(made, [t for t in list(run.target_ids) if t != run.inherited_target_id])
    logger.info("cuAlign: turn %s (%s, %s) cancelled — %s; %d workflow task(s) cancelled, %d plan(s) dropped",
                run.request_id, run.case_id, run.step, why, len(tasks), len(made))
    return True


def cancel_case(case_id: str, why: str) -> bool:
    run = RUNS.get(case_id)
    return run is not None and cancel(run, why)


class PlanEventsASGI:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST" or scope.get("path") != "/chat/stream":
            return await self.app(scope, receive, send)
        body = b""
        while True:
            msg = await receive()
            if msg["type"] == "http.disconnect":
                return
            body += msg.get("body", b"")
            if not msg.get("more_body"):
                break
        try:
            data = json.loads(body)
            if not isinstance(data, dict):
                raise ValueError("expected JSON object")
            raw = data.pop("cualign", None)
            sent = data.pop("step", None)   # the turn's step is a top-level body field (the screen sends it beside `cualign`)
            step = resolve_step(data, sent)
            # Non-UI clients may still use the ordinary NAT endpoint.
            ctx = ChatContext.model_validate({**raw, "step": step} if step else raw) if raw is not None                 else ChatContext(case_id=STORE.active_case or "moderate", **({"step": step} if step else {}))
            preload = getattr(getattr(scope.get("app"), "state", None), "cualign_preload", None)  # set by worker.add_routes
            run, system = open_run(ctx, preload=preload)
            run.text = user_text(data.get("messages"))   # select_plan checks it against the conditions (core/request_check)
            if sent is not None:
                text = user_text(data.get("messages"))
                STORE.turn_steps[run.case_id] = {"screen": recorded.replay_step(text, sent, as_sent=True),
                                                 "turn": recorded.replay_step(text, run.step, as_sent=True)}
                if step != sent:
                    logger.info("cuAlign: turn step %s (the screen sent %s) for %r", step, sent, text[:60])
            data.setdefault("messages", []).insert(0, system)
            body = json.dumps(data).encode()
        except ExtractionTeethNeeded as e:     # the prescription is the dentist's: say what is missing (#56)
            return await JSONResponse({"detail": str(e)}, status_code=400)(scope, receive, send)
        except (ValueError, KeyError, TypeError) as e:
            why = reason_ko(e) if isinstance(e, ValueError) and "발치" in reason_ko(e) else ""
            return await JSONResponse({"detail": "Invalid case, parent plan or constraints" + (f": {why}" if why else "")},
                                      status_code=400)(scope, receive, send)

        replayed = finished = False
        async def replay():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            msg = await receive()   # Starlette's StreamingResponse listens here for the client going
            if msg["type"] == "http.disconnect" and not finished:
                cancel(run, "the stream was closed")
            return msg

        is_stream = False
        async def wrapped(message):
            nonlocal is_stream, finished
            if message["type"] == "http.response.start":
                is_stream = message["status"] == 200 and any(
                    k.lower() == b"content-type" and b"text/event-stream" in v for k, v in message.get("headers", []))
                if is_stream:   # the step the server runs, before any event: 건너뛰기 can come before plan_context (step-intent)
                    message = {**message, "headers": [*message.get("headers", []), (b"x-cualign-step", run.step.encode())]}
            last = message["type"] == "http.response.body" and not message.get("more_body", False)
            if run.cancelled:   # a cut turn's late events (and its step_done) are not the screen's any more
                return
            if is_stream and last:
                extra = b""
                common = {"request_id": run.request_id, "case_id": run.case_id}
                if run.error:
                    # The workflow raised (NAT already wrote its unframed workflow_error line). The kind tells the UI
                    # whether this was the NVIDIA API's overload, worth a resend button (#51); no plan is offered.
                    extra = event("plan_error", {**common, **run.error})
                elif run.refused:
                    # the rails replaced the answer, so its plan is not offered (it stays stored); the kind tells the
                    # screen whether the refused message must leave the chat it resends (a personal identifier, #157)
                    extra = event("turn_refused", {**common, "kind": run.refused_kind or "rails",
                                                   **({"text": run.redacted} if run.refused_kind == "pii" else {})})
                elif run.selected_plan_id:
                    by_server = await review_skipped(scope, run.selected_plan_id)
                    if run.cancelled:   # cut while the server's own review ran: its plan is gone
                        return
                    plan = STORE.plan_json(run.selected_plan_id)
                    extra = event("plan_selected", {**common, "schema_version": 1,
                        "plan_id": plan["plan_id"], "parent_plan_id": plan["parent_plan_id"],
                        "review": plan["review"], "reviewed_by_server": by_server})
                elif run.plan_ids and run.step != "target":
                    extra = event("plan_error", {**common, "message": "계획은 생성됐으나 최종 선택을 받지 못했습니다."})
                extra += event("plan_context", {**common, "constraints": run.constraints.model_dump(mode="json"),
                                                 "rails": run.rails, "step": run.step})
                done = step_done(run)
                if done is not None:
                    extra += event("step_done", {**common, **done})
                message = {**message, "body": message.get("body", b"") + extra}
            await send(message)
            finished = finished or last

        # Keep context alive through NAT's streaming response, including spawned tasks.
        token = CURRENT_RUN.set(run)
        RUNS[run.case_id] = run
        try:
            await self.app(scope, replay, wrapped)
        finally:
            if not finished:   # the response never ended (the client went, or the turn was cut): stop what still runs
                cancel(run, "the stream ended early")
            run.closed = True
            if RUNS.get(run.case_id) is run:
                del RUNS[run.case_id]
            CURRENT_RUN.reset(token)
