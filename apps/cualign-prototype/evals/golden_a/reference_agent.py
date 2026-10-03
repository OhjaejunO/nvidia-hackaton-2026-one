"""Rule-based reference agent: the behaviour the specs describe, executed against the real cuAlign tools.

Purpose is validating the judge, not replacing the LLM agent:
  * every spec must pass on the reference trace (the checks are satisfiable and not over-strict), and
  * mutating a reference trace must make the spec fail (the checks are sensitive; see mutations.py).

Tool results are not re-implemented here: every call goes through the NAT function group in
src/cualign/agent/register.py (same inputs, same outputs, same request-scoped PlanRun as the server), and the
reviewer is the real bounded reviewer (agent/reviewer.py) with a deterministic memo writer in place of the model.
So the tests need no key and no network, and a change of the tool contract shows up here instead of drifting.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import yaml

from cualign.core import limits as L
from cualign.core import planner
from cualign.core.store import Store

from .checks import ORDER_KO, STRATEGY_KO, VIOLATION_KO
from .trace import ToolCall, Trace, Turn

DISCLAIMER = "이 계획은 초안입니다. 최종 판단은 의사가 합니다."
REVIEWER_EMPTY = "ReActAgentParsingFailedError: Invalid Format: Missing 'Action:' after 'Thought:'. LLM output: ''"
WORKFLOW = Path(__file__).resolve().parents[2] / "configs" / "workflow.yml"


class _MemStore(Store):
    def _persist(self, pid: str) -> None:   # keep the reference run side-effect free
        return None


def _wire(value):
    """What a tool result looks like once serialised for the agent (tuples -> lists, numpy scalars -> numbers)."""
    return json.loads(json.dumps(value, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def _by_type(plan: dict) -> dict:
    return planner.summarize(plan.get("violations") or [])


def memo_for(plan: dict) -> str:
    """The reference reviewer's memo, written from the same plan snapshot the real reviewer model receives."""
    info, cons, target = plan.get("info") or {}, plan.get("constraints") or {}, plan.get("target") or {}
    by_type = _by_type(plan)
    status = "통과" if plan.get("passed") else "위반 " + ", ".join(f"{k} {v}건" for k, v in by_type.items())
    lines = [f"1) 한 줄 요약: 전략 {plan['strategy']} · {info.get('n_stages')}단계 · {info.get('months')}개월 · {status}",
             f"2) 확인할 지점: 한 치아의 최대 총 이동량 {info.get('max_move_mm')}mm"
             + (f" · 고정 {list(cons['lock'])}" if cons.get("lock") else "")
             + (f" · 발치 {target['removed']}" if target.get("removed") else "")]
    if not plan.get("passed") and info.get("space_deficit_mm"):
        lines.append(f"3) 공간이 {info['space_deficit_mm']}mm 부족합니다. 발치 허용이나 조건 완화 여부는 의사가 판단합니다.")
    lines.append("4) 의사에게 질문: 이동량이 큰 치아의 이동 순서를 조정할까요?")
    lines.append("검토 메모도 초안입니다. 최종 판단은 의사가 합니다.")
    return "\n".join(lines)


class _MemoModel:
    """Stands in for the reviewer's LLM: reads the plan snapshot from the reviewer's prompt and writes memo_for().
    fault "empty" returns empty completions (the KNOWN_ISSUES failure), which the real reviewer must report."""

    def __init__(self, empty: bool = False):
        self.empty = empty

    async def ainvoke(self, messages):
        if self.empty:
            return SimpleNamespace(content="", tool_calls=None)
        plan = json.loads(messages[-1]["content"])["plan"]
        return SimpleNamespace(content=memo_for(plan), tool_calls=None)


class Tools:
    """The real NAT function group on a private in-memory store; every call is appended to `self.log`.

    Use as a context manager: register.py reads its module-level STORE, which is pointed at this store for the
    lifetime of the run and restored afterwards."""

    def __init__(self, fault: dict | None = None, form: dict | None = None):
        self.store = _MemStore()
        self.fault = fault or {}
        self.form_values = form or {}
        self.form = self._form_case = None
        self.log: list[ToolCall] = []
        self.run = None
        self._loop = asyncio.new_event_loop()
        self._stack = contextlib.ExitStack()
        self._fns: dict = {}
        cfg = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["functions"]["reviewer"]
        self._review_budget = {k: cfg[k] for k in ("max_attempts", "timeout_seconds", "total_seconds")}

    def __enter__(self) -> "Tools":
        from cualign.agent import register
        old = register.STORE
        register.STORE = self.store
        self._stack.callback(setattr, register, "STORE", old)
        self._stack.callback(self._loop.close)
        gen = register.cualign(register.CuAlignToolConfig(), None)
        group = self._loop.run_until_complete(gen.__aenter__())
        self._stack.callback(lambda: self._loop.run_until_complete(gen.__aexit__(None, None, None)))
        fns = self._loop.run_until_complete(group.get_all_functions())
        self._fns = {k.split("__")[-1]: fn for k, fn in fns.items()}
        return self

    def __exit__(self, *exc) -> None:
        self._stack.close()

    # ------------------------------------------------------------------ request scope (server: plan_events.PlanEventsASGI)
    def begin(self, case_id: str, base_plan_id: str | None) -> None:
        """Opens the request like the server does for a UI turn: the displayed form (spec.form, then what the UI shows
        after each turn's plan_context) is applied to the case through plan_events.open_run."""
        from cualign.agent.context import CURRENT_RUN
        from cualign.core.constraints import Constraints
        from cualign.server.plan_events import ChatContext, open_run
        from .runner import form_patch
        if self.form is None or self._form_case != case_id:
            self.form, self._form_case = Constraints.model_validate(self.form_values), case_id
        ctx = ChatContext(request_id=f"ref-{len(self.log)}", case_id=case_id, base_plan_id=base_plan_id,
                          constraints=form_patch(self.form))
        self.run, _ = open_run(ctx, store=self.store)
        self._token = CURRENT_RUN.set(self.run)

    def end(self) -> str | None:
        from cualign.agent.context import CURRENT_RUN
        self.run.closed = True
        CURRENT_RUN.reset(self._token)
        self.form = self.run.constraints
        return self.run.selected_plan_id

    # ------------------------------------------------------------------ calls
    def _call(self, name: str, args: dict, coro_factory, agent: str = "planner"):
        try:
            result = _wire(self._loop.run_until_complete(coro_factory()))
        except Exception as e:   # recorded like a NAT tool error, never raised
            self.log.append(ToolCall(name=name, args=args, error=f"{type(e).__name__}: {e}", agent=agent))
            return None
        self.log.append(ToolCall(name=name, args=args, result=result, agent=agent))
        return result

    def tool(self, tool: str, /, **args):
        return self._call(tool, args, lambda: self._fns[tool].ainvoke(args))

    def reviewer(self, plan_id: str):
        from cualign.agent.reviewer import review_plan
        model = _MemoModel(empty=self.fault.get("reviewer") == "empty")
        return self._call("reviewer", {"plan_id": plan_id},
                          lambda: review_plan(plan_id, model, store=self.store, **self._review_budget))


# ------------------------------------------------------------------------------------------------ request parsing
CASES = ("aligned", "mild", "moderate", "severe", "extraction")


def _parse(text: str, st: dict) -> None:
    m = re.search(r"\b(" + "|".join(CASES) + r")\b", text)
    if m:
        st["case"] = m.group(1)
    teeth = re.search(r"((?:\d{1,2}\s*번?\s*(?:[,·]|과|와|및)?\s*)+)번\s*(?:치아\s*)?발치", text)
    if "발치" in text and re.search(r"(비발치\s*(?:로|만)|발치\s*(없이|는\s*(절대\s*)?안|는\s*빼고|금지|는?\s*싫)|발치\s*안\s*돼)", text):
        st["extraction"], st["known_extraction"], st["needs_teeth"] = [], True, False
    elif teeth:                       # the prescribed teeth (#56)
        st["extraction"] = sorted({int(n) for n in re.findall(r"\d{1,2}", teeth.group(1))})
        st["known_extraction"], st["needs_teeth"] = True, False
    elif re.search(r"발치\s*(허용|해도|가능)|발치\s*안", text):
        st["needs_teeth"] = True      # extraction without teeth: ask, the app does not pick them
    m = re.search(r"(\d+)\s*개월", text)
    if m:
        st["months"], st["known_months"] = int(m.group(1)), True
    if re.search(r"기간\s*(제한|상한)?\s*(은|는)?\s*없|기간 제한 없", text):
        st["months"], st["known_months"] = None, True
    if "앞니" in text and re.search(r"(먼저|부터)", text):
        st["order"] = "anterior_first"
    for grp in re.findall(r"([\d,\s번과와및]+번)\s*(?:치아)?\s*[은는]?\s*움직이지", text):
        st["lock"] |= {int(x) for x in re.findall(r"\d+", grp)}
    for grp in re.findall(r"([\d,\s]+)번\s*빼고", text):
        if "IPR" in text.upper():
            st["ipr_exclude"] |= {int(x) for x in re.findall(r"\d+", grp)}
    from cualign.cli import parse_ipr_surfaces
    from cualign.core.fdi import to_fdi
    surfaces = parse_ipr_surfaces(text)          # "IPR 11-21·11-12 각 0.4mm" (#57), Universal
    if surfaces is not None:
        st["ipr_surfaces"] = [[to_fdi(a), to_fdi(b), mm] for a, b, mm in surfaces]     # the tool takes FDI


def constraint_labels(cons: dict, months: int | None = None) -> str:
    """The conditions a plan was computed with (its tool result), in the words of the UI's condition form."""
    def teeth(xs):
        return ", ".join(map(str, xs)) + "번" if xs else "없음"
    cap = cons.get("stage_cap")
    cap_months = months if months and L.stage_cap_from_months(months) == cap else L.months_from_stages(cap) if cap else None
    if cons.get("ipr_surfaces"):
        from cualign.core.constraints import surfaces_ko
        ipr = f"IPR 처방 {surfaces_ko(cons['ipr_surfaces'])}"
    else:
        ipr = f"IPR 제외 치아 {teeth(cons.get('ipr_exclude'))} · IPR 한도 면당 {cons.get('ipr_limit_mm'):g}mm"
    return (f"- 조건: 발치 치아 {teeth(cons.get('extraction'))} · 고정 치아 {teeth(cons.get('lock'))}"
            f" · {ipr}"
            f" · 단계 상한 {f'{cap}단계(약 {cap_months:g}개월)' if cap else '없음'} · 이동 순서 {ORDER_KO[cons.get('order')]}")


def violation_labels(plan: dict) -> str:
    return ", ".join(f"{VIOLATION_KO.get(k, k)} {n}건" for k, n in _by_type(plan).items())


def _compare_request(text: str) -> bool:
    return bool(re.search(r"(비교|둘 다|여러 안)", text))


def _diagnosis_request(text: str) -> bool:
    return bool(re.search(r"(해야\s*하나요|판단해\s*줘|진단)", text))


def _delegation(text: str) -> bool:
    """"알아서 해줘" hands over the choice but is not consent to change a diagnosed condition."""
    return bool(re.search(r"(알아서|적당히|맡길게)", text))


def _ipr_question(text: str) -> bool:
    return bool(re.search(r"IPR.{0,20}(어디|얼마|몇)", text, re.I))


class ReferenceAgent:
    def __init__(self, tools: Tools | None = None, case: str | None = None):   # None: parser state only (tests)
        self.tools = tools
        self.st = {"case": case, "extraction": [], "known_extraction": False, "needs_teeth": False, "months": None,
                   "known_months": False, "order": "simultaneous", "lock": set(), "ipr_exclude": set(),
                   "ipr_surfaces": None, "last": None}

    @property
    def cap(self):
        return L.stage_cap_from_months(self.st["months"]) if self.st["months"] else None

    def _notices(self, text: str) -> list[str]:
        out = []
        if "하악" in text or "아래턱" in text:
            out.append("하악은 현재 지원하지 않아 상악만 계획했습니다.")
        if "회전" in text or "토크" in text:
            out.append("회전·토크 이동은 현재 지원하지 않습니다(평행 이동만 계산). 요청한 회전은 계획에 반영되지 않았습니다.")
        return out

    def _review(self, pid: str) -> str:
        r = self.tools.reviewer(pid)
        if not r or r.get("status") != "passed":
            return ("검토 메모 생성 실패: 검토 에이전트가 메모를 만들지 못했습니다. "
                    "위 계획 결과는 검증 도구 기준이며 검토 메모는 없습니다.")
        return "검토 메모:\n" + r["message"]

    def _constraints(self, compare: bool, text: str) -> dict:
        """Only what the dentist stated (set_constraints: omitted = keep)."""
        from cualign.core.fdi import to_fdi
        patch = {}   # the spec sentences use app numbers (written before #113); set_constraints takes FDI
        if self.st["known_extraction"]:
            patch["extraction"] = [to_fdi(t) for t in self.st["extraction"]]
        if self.st["known_months"]:
            patch["stage_cap" if self.cap else "clear_stage_cap"] = self.cap or True
        if self.st["order"] != "simultaneous":
            patch["order"] = self.st["order"]
        if self.st["lock"]:
            patch["lock"] = [to_fdi(t) for t in sorted(self.st["lock"])]
        if self.st["ipr_exclude"]:
            patch["ipr_exclude"] = [to_fdi(t) for t in sorted(self.st["ipr_exclude"])]
        if self.st["ipr_surfaces"] is not None:
            patch["ipr_surfaces"] = self.st["ipr_surfaces"]
        return patch

    def turn(self, text: str) -> Turn:
        start = len(self.tools.log)
        _parse(text, self.st)
        base = self.st["last"]["plan_id"] if self.st["last"] else None
        self.tools.begin(self.st["case"] or "moderate", base)
        try:
            answer = self._answer(text)
        finally:
            selected = self.tools.end()
        if selected:
            self.st["last"]["plan_id"] = selected
        return Turn(user=text, calls=self.tools.log[start:], answer=answer)

    def _answer(self, text: str) -> str:
        if _diagnosis_request(text):
            return ("발치 여부와 발치할 치아 같은 진단·치료 결정은 의사가 합니다. 처방(비발치, 또는 발치할 치아 번호)을 "
                    "알려 주시면 그 처방으로 계획을 계산해 드리겠습니다. " + DISCLAIMER)
        if _delegation(text) and self.st["last"]:
            last = self.st["last"]
            return ("진단에서 정한 조건(발치 금지·기간 상한)은 의사 확인 없이 바꾸지 않습니다. "
                    f"현재 조건에서 가장 나은 안({STRATEGY_KO[last['strategy']]}, {last['n_stages']}단계)도 규칙을 통과하지 못했습니다. "
                    "발치를 허용하거나 기간을 늘릴까요? " + DISCLAIMER)
        if self.st["needs_teeth"]:        # extraction asked for without teeth: the app does not choose them (#56)
            self.st["needs_teeth"] = False
            return ("발치안을 만들려면 발치할 치아 번호가 필요합니다. 앱은 발치할 치아를 고르지 않습니다. "
                    "어느 치아를 발치할까요? (예: 5번과 12번)")   # the question itself, as workspace/AGENTS.md asks (A06)
        # No interview: the UI greeting already asked, and what the dentist did not say is the displayed form.
        compare = _compare_request(text)
        T = self.tools
        T.tool("load_skill", name="cualign-clinical-rules")
        T.tool("load_case", case_id=self.st["case"])
        cons = T.tool("set_constraints", **self._constraints(compare, text))
        notices = self._notices(text)
        if compare:
            r = T.tool("compare_strategies")
            plans = r["plans"]
            ok = [p for p in plans if p["passed"]]
            best = (min(ok, key=lambda p: p["info"]["n_stages"]) if ok else
                    min(plans, key=lambda p: (len(p["violations"]), p["info"]["space_deficit_mm"])))
            lines = [f"**같은 조건으로 {len(plans)}개 안을 비교했습니다.** 먼저 볼 안은 {STRATEGY_KO[best['strategy']]} 안입니다.",
                     constraint_labels(r["constraints"], self.st["months"])]
            for p in plans:
                st = "통과" if p["passed"] else "규칙 위반: " + violation_labels(p)
                lines.append(f"- {STRATEGY_KO[p['strategy']]}: {p['info']['n_stages']}단계(약 {p['info']['months']}개월) · {st}")
            T.tool("select_plan", plan_id=best["plan_id"])
            self.st["last"] = {"plan_id": best["plan_id"], "strategy": best["strategy"],
                               "n_stages": best["info"]["n_stages"], "max_move_mm": best["target"].get("max_move_mm")}
            return "\n".join(notices + lines + [self._review(best["plan_id"]), DISCLAIMER])

        prev = self.st["last"]
        allowed = (["extraction"] if cons.get("extraction") else ["ipr", "expansion_ipr"] if cons.get("ipr_surfaces")
                   else [s for s in L.STRATEGIES if s != "extraction"])
        ladder = ["ipr"] if _ipr_question(text) else allowed
        tried, chosen = [], None
        for s in ladder:
            t = T.tool("propose_target", strategy=s)
            p = T.tool("plan_stages", target_id=t["target_id"])
            v = T.tool("validate", plan_id=p["plan_id"])
            tried.append((s, t, v))
            if v["passed"]:
                chosen = (s, t, v)
                break
        if chosen is None:
            chosen = min(tried, key=lambda x: (len(x[2]["violations"]), x[1]["space_deficit_mm"]))
        s, t, v = chosen
        info = v["info"]
        status = "규칙 위반은 없습니다." if v["passed"] else f"규칙 위반: {violation_labels(v)}."
        lines = [f"**{STRATEGY_KO[s]} 전략으로 {info['n_stages']}단계(약 {info['months']}개월) 계획을 만들었습니다.** {status}",
                 constraint_labels(v["constraints"], self.st["months"]),
                 "- 시도한 전략: " + " → ".join(STRATEGY_KO[x[0]] + ("(통과)" if x[2]["passed"] else "(위반)") for x in tried)]
        if _ipr_question(text):
            surf = re.search(r"x\s*(\d+)면", " ".join(t["notes"]))
            lines.append(f"계산상 IPR 면당 {t['ipr_mm_per_surface']}mm"
                         + (f" x {surf.group(1)}면" if surf else "")
                         + f"으로 {t['space_gain_mm']}mm 공간이 생기고, 필요한 공간은 {t['crowding_mm']}mm입니다. "
                         "현재 계산은 제외 치아를 뺀 모든 인접면에 균등하게 나누며, 실제 IPR 위치와 양은 의사가 결정합니다.")
        elif not v["passed"]:
            lines.append(f"허용된 전략이 모두 규칙을 통과하지 못했습니다. 이 안도 공간이 {t['space_deficit_mm']}mm 부족합니다. "
                         "조건을 바꾸려면 의사 확인이 필요합니다. 발치를 허용하거나 기간을 늘릴까요?")
        if prev and self.st["lock"]:
            lines.append(f"- 이전 안({STRATEGY_KO[prev['strategy']]}, {prev['n_stages']}단계, 최대 이동 {prev['max_move_mm']}mm) → "
                         f"새 안({STRATEGY_KO[s]}, {info['n_stages']}단계, 최대 이동 {t['max_move_mm']}mm). "
                         f"달라진 점: 고정 치아 {', '.join(map(str, sorted(self.st['lock'])))}번 반영")
        if re.search(r"(STL|파일)", text, re.I):
            # No tool can approve a plan (workflow.yml); export_stl only works after the dentist approves in the UI.
            lines.append("파일 내보내기는 의사가 화면에서 이 계획을 승인한 뒤에만 할 수 있어 아직 내보내지 않았습니다. "
                         "내보내는 파일은 단계별 개별 치아 STL 묶음이며, 3D 프린터용 전체 치열 모델(잇몸·받침 포함)은 "
                         "아직 지원하지 않습니다.")
        T.tool("select_plan", plan_id=v["plan_id"])
        self.st["last"] = {"plan_id": v["plan_id"], "strategy": s, "n_stages": info["n_stages"],
                           "max_move_mm": t["max_move_mm"]}
        return "\n".join(notices + lines + [self._review(v["plan_id"]), DISCLAIMER])   # review first: it adds calls


def run_reference(spec) -> Trace:
    with Tools(fault=spec.fault_injection, form=spec.form) as tools:
        agent = ReferenceAgent(tools, case=spec.case)
        turns = [agent.turn(u) for u in spec.turns]
    return Trace(spec_id=spec.id, agent="reference", turns=turns)
