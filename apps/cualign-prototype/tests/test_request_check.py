"""set_constraints takes the dentist's FDI numbers (a) and an agent turn's plan is checked against its sentence (b).

A live revision «13번 치아는 움직이지 말고 다시 짜줘.» locked FDI 25 (app tooth 13): the model passed 13 to a tool that took
app numbers, and the rules, the reviewer and the rails all passed the plan. (a) moves the conversion from the model to
the server, at the tool input only: the UI form, the recorded answers and the MCP tools keep ConstraintPatch (app
numbers) and Tooth keeps 2..15. (b) blocks approval when the sentence and the plan's conditions disagree.
"""
import asyncio
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from cualign.agent import register
from cualign.agent.context import CURRENT_RUN, PlanRun
from cualign.core import recorded, request_check
from cualign.core import store as store_module
from cualign.core.constraints import ConstraintPatch, Constraints, ToolConstraintPatch
from cualign.core.service import PlanningService
from cualign.server import api
from cualign.server.plan_events import ChatContext, open_run

REVISION = "13번 치아는 움직이지 말고 다시 짜줘."
SAMPLES = ("poseidon-000001", "poseidon-000097", "poseidon-000131")
REVIEW = {"status": "passed", "attempts": 1, "message": "memo", "error": None}


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    s = store_module.Store()
    monkeypatch.setattr(store_module, "OUT_DIR", tmp_path)
    monkeypatch.setattr(api, "OUT_DIR", tmp_path)
    monkeypatch.setattr(api, "STORE", s)
    monkeypatch.setattr(register, "STORE", s)
    return s


def client(review=None):
    app = FastAPI()
    api.add_api_routes(app, review=review)
    return TestClient(app)


def run_tools(calls, run):
    """The agent's tools called in order under `run`; an argument may be a function of the results so far."""
    async def scenario():
        async with register.cualign(register.CuAlignToolConfig(), None) as group:
            fns = {k.split("__")[-1]: f for k, f in (await group.get_all_functions()).items()}
            token = CURRENT_RUN.set(run)
            try:
                out = []
                for name, arg in calls:
                    out.append(await fns[name].ainvoke(arg(out) if callable(arg) else arg))
                return out
            finally:
                CURRENT_RUN.reset(token)
    return asyncio.run(scenario())


def C(**changes):
    return Constraints().patched(changes)


# ---- (a) the tool takes FDI; everything behind it keeps app numbers -------------------------------------------------

def test_set_constraints_takes_fdi_and_stores_app_numbers(isolated):
    lock, excl, ext = run_tools([("set_constraints", {"lock": [13]}),
                                 ("set_constraints", {"ipr_exclude": [12, 11, 21, 22]}),
                                 ("set_constraints", {"extraction": [14, 24]})], PlanRun("fdi", "moderate", None, Constraints()))
    assert lock["lock"] == [6] and lock["teeth_fdi"]["lock"] == [13] and "고정 치아 13번" in lock["conditions_ko"]
    assert excl["ipr_exclude"] == [7, 8, 9, 10] and excl["lock"] == [6]           # omitted = keep
    assert excl["teeth_fdi"]["ipr_exclude"] == [12, 11, 21, 22]
    assert ext["extraction"] == [5, 12] and ext["teeth_fdi"]["extraction"] == [14, 24]
    assert isolated.case_constraints["moderate"].lock == (6,)


@pytest.mark.parametrize("teeth", [[6], [3], [19]])
def test_a_number_outside_the_fdi_upper_arch_is_refused_at_the_tool(isolated, teeth):
    # app numbers (6, 3) and FDI 19 are not a tooth the dentist reads: refused before the tool runs, nothing stored
    with pytest.raises(ValidationError):
        run_tools([("set_constraints", {"lock": teeth})], PlanRun("fdi", "moderate", None, Constraints()))
    assert "moderate" not in isolated.case_constraints


def test_the_tool_schema_asks_for_fdi():
    props = ToolConstraintPatch.model_json_schema()["properties"]
    for field in ("extraction", "lock", "ipr_exclude", "ipr_surfaces"):
        assert "FDI" in props[field]["description"], field


def test_shared_tooth_range_and_internal_patch_are_unchanged():   # acceptance (c): Tooth keeps 2..15
    assert ConstraintPatch(lock=[13]).lock == [13]                 # app numbers, no conversion
    for bad in (1, 16, 25):
        with pytest.raises(ValidationError):
            ConstraintPatch(lock=[bad])
        with pytest.raises(ValidationError):
            Constraints(lock=(bad,))
    assert Constraints(lock=(2, 15)).lock == (2, 15)
    assert ToolConstraintPatch(lock=[13]).lock == [13]              # kept as sent: a second validation must not reconvert
    assert ToolConstraintPatch(lock=[13]).changes() == {"lock": [6]} and ToolConstraintPatch(lock=[25]).changes() == {"lock": [13]}


def test_the_agent_path_converts_once(isolated):
    # The ReAct agent calls the tool through LangChain's StructuredTool, which validates the arguments with the input
    # schema and hands the validated values to NAT, which validates them again. Calling ainvoke directly validates once
    # and hid that e7fa32b refused every ipr_surfaces prescription on this path (FDI 11 -> 8 -> refused).
    from nat.plugins.langchain.tool_wrapper import langchain_tool_wrapper

    async def scenario():
        async with register.cualign(register.CuAlignToolConfig(), None) as group:
            name, fn = next((k, f) for k, f in (await group.get_all_functions()).items() if k.endswith("set_constraints"))
            tool, out = langchain_tool_wrapper(name, fn, None), []
            for args in ({"lock": [13]}, {"ipr_exclude": [12, 11, 21, 22]}, {"ipr_surfaces": [[11, 21, 0.4]]},
                         {"extraction": [14, 24]}):
                token = CURRENT_RUN.set(PlanRun("agent-path", "moderate", None, Constraints()))
                isolated.case_constraints.pop("moderate", None)
                try:
                    out.append(await tool.ainvoke(args))
                finally:
                    CURRENT_RUN.reset(token)
            return out
    lock, excl, surf, ext = asyncio.run(scenario())
    assert lock["lock"] == [6] and excl["ipr_exclude"] == [7, 8, 9, 10] and ext["extraction"] == [5, 12]
    assert surf["ipr_surfaces"] == [[8, 9, 0.4]] and surf["teeth_fdi"]["ipr_surfaces"] == [[11, 21, 0.4]]


def test_ui_form_and_chat_context_keep_app_numbers(isolated):     # acceptance (b): the UI path
    # state.js turns the form's FDI into app numbers before it sends (#113); the server reads them as they are
    ctx = ChatContext(case_id="moderate", constraints={"lock": [13], "ipr_exclude": [7, 8]})
    assert ctx.constraints.lock == [13]
    run, msg = open_run(ctx, store=isolated)
    context = json.loads(msg["content"].removeprefix("cuAlign server context: "))
    assert context["constraints"]["lock"] == [13] and context["teeth_fdi"]["lock"] == [25]
    assert context["teeth_fdi"]["ipr_exclude"] == [12, 11] and run.opened.lock == (13,)
    with client() as c:
        r = c.post("/api/plan", json={"case_id": "moderate", "lock": [13]}).json()
    assert (r["chosen"] or r["best_failed"])["constraints"]["lock"] == [13]


@pytest.mark.parametrize("case_id", SAMPLES)
def test_shipped_setup_recordings_replay_with_their_app_numbers(isolated, case_id):   # acceptance (a)
    rec = recorded.load(case_id, "setup")
    with client() as c:
        r = c.post(f"/api/cases/{case_id}/replay", json={"step": "setup"})
    assert r.status_code == 200, r.text
    stored = isolated.case_constraints[case_id]
    for field in ("extraction", "lock", "ipr_exclude"):
        assert list(getattr(stored, field)) == rec["constraints"][field], field


@pytest.mark.slow
@pytest.mark.parametrize("case_id", SAMPLES)
def test_every_shipped_recording_replays(isolated, case_id):     # acceptance (a): all 15 files, as the screen walks them
    with client() as c:
        base = None
        for step in recorded.STEPS:
            rec = recorded.load(case_id, step)
            if rec is None:
                continue
            r = c.post(f"/api/cases/{case_id}/replay", json={"step": step, "base_plan_id": base})
            assert r.status_code == 200, (step, r.text)
            sel = r.json().get("plan_selected")
            if sel:
                base = sel["plan_id"]
                got = c.get(f"/api/plans/{base}").json()["constraints"]
                for field in ("extraction", "lock", "ipr_exclude"):
                    assert got[field] == rec["constraints"].get(field, got[field]), (step, field)


# ---- (b) the sentence against the plan's conditions --------------------------------------------------------------

# The plan's four acceptance sentences (기획 ① ⓑ) word for word: «13번도» names no field after it and takes the one
# of the number before it in the same sentence («25번 고정에»).
PLAN_ADD = "기존 25번 고정에 13번도"


@pytest.mark.parametrize("text, before, after", [
    ("13번 고정", C(), C(lock=[6])),                                                   # app 6 added
    (PLAN_ADD, C(lock=[13]), C(lock=[6, 13])),                                        # 25 (app 13) kept, 6 added
    ("13번 고정 풀어줘", C(lock=[6, 13]), C(lock=[13])),                                # 6 released, the rest kept
    ("13번 고정하고 12개월 안에", C(), C(lock=[6], stage_cap=26)),                      # and the cap round(12*30.4/14)
    (REVISION, C(), C(lock=[6])),                                                     # the live revision, applied
    ("13번도 움직이지 마.", C(lock=[13]), C(lock=[6, 13])),                              # a full list keeps the old tooth
    ("14번 고정하고 24번도.", C(), C(lock=[5, 12])),                                     # «도» takes the field before it
    ("13번 고정 풀고 14번도.", C(lock=[5, 6, 13]), C(lock=[13])),                        # ... and its action (release)
    ("13번 고정해줘. 14번도 봐줘.", C(), C(lock=[6])),                                   # not across a sentence: 14 unread
    ("발치 없이 12개월 안에 끝나는 계획 짜줘. 앞니 총생부터 풀고.", C(), C(stage_cap=26, order="anterior_first")),
    ("IPR 은 앞니 12,11,21,22번 빼고 해줘.", C(), C(ipr_exclude=[7, 8, 9, 10])),
    ("14번과 24번 발치로 계획 짜줘.", C(), C(extraction=[5, 12])),
    ("IPR 11-21·11-12 각 0.4mm 로 해줘.", C(), C()),                                    # contacts, not teeth
    ("단계로 나눠줘.", C(lock=[6]), C(lock=[6])),                                         # nothing asked, nothing changed
])
def test_requests_that_were_applied_are_not_blocked(text, before, after):
    assert request_check.check(text, before, after) == {"status": "match", "reasons_ko": [], "unchecked": []}


@pytest.mark.parametrize("text, before, after, reasons", [
    (REVISION, C(), C(lock=[13]),                                                     # the live run: FDI 25 locked
     ["고정 치아에 요청한 13번이 없습니다", "고정 치아에 요청에 없던 25번이 들어갔습니다"]),
    ("13번 고정 풀어줘", C(lock=[6, 13]), C(lock=[6, 13]), ["고정 치아에 요청에 없던 13번이 들어갔습니다"]),
    (PLAN_ADD, C(lock=[13]), C(lock=[13]), ["고정 치아에 요청한 13번이 없습니다"]),      # the addition left out
    (PLAN_ADD, C(lock=[13]), C(lock=[6]), ["고정 치아에 요청한 25번이 없습니다"]),       # the old tooth dropped
    ("13번 고정하고 12개월 안에", C(), C(lock=[6]), ["요청한 12개월(단계 상한 26단계)과 저장된 단계 상한(없음)이 다릅니다"]),
    ("14번 고정하고 24번도.", C(), C(lock=[5]), ["고정 치아에 요청한 24번이 없습니다"]),
    ("12개월 안에 끝나게 다시 짜줘.", C(stage_cap=26), C(lock=[6], stage_cap=26),
     ["고정 치아가 요청 문장에 없이 바뀌었습니다(없음 → 13번)"]),
    ("14번과 24번 발치로 계획 짜줘.", C(), C(extraction=[4, 13]),
     ["발치 치아에 요청한 14·24번이 없습니다", "발치 치아에 요청에 없던 15·25번이 들어갔습니다"]),
])
def test_a_plan_that_differs_from_the_sentence_is_a_mismatch(text, before, after, reasons):
    assert request_check.check(text, before, after) == {"status": "mismatch", "reasons_ko": reasons, "unchecked": []}


def test_a_tooth_named_by_words_is_left_unchecked_not_matched():
    # «오른쪽 송곳니» has no number to read: the change is listed as unchecked (못 잡는 것), never as checked
    r = request_check.check("오른쪽 송곳니는 움직이지 마.", C(), C(lock=[6]))
    assert r == {"status": "match", "reasons_ko": [], "unchecked": ["lock"]}


@pytest.mark.parametrize("lock, status", [([13], "match"), ([25], "mismatch"), (None, None)])
def test_select_plan_records_the_check_on_the_turn_plan(isolated, lock, status):
    run = PlanRun("rc", "moderate", None, Constraints())
    if status:
        run.text, run.opened = REVISION, Constraints()          # a /chat/stream turn; MCP and tests have no sentence
    calls = [("set_constraints", {"lock": lock or [13]}), ("propose_target", {"strategy": "expansion_ipr"}),
             ("plan_stages", lambda out: {"target_id": out[1]["target_id"]}),
             ("select_plan", lambda out: {"plan_id": out[2]["plan_id"]})]
    selected = run_tools(calls, run)[-1]
    if status is None:
        assert "request_check" not in selected
        return
    assert selected["request_check"]["status"] == status
    saved = json.loads((store_module.OUT_DIR / "plans" / f"{selected['plan_id']}.json").read_text(encoding="utf-8"))
    assert saved["request_check"] == selected["request_check"]


def test_a_mismatch_blocks_approval_and_survives_a_restart(isolated):
    s = isolated
    svc = PlanningService(s)
    pid = svc.stages(svc.target("moderate", "expansion_ipr", Constraints()), None)
    s.set_review(pid, REVIEW)
    s.approve(pid)
    mismatch = request_check.check(REVISION, Constraints(), Constraints())
    assert mismatch["status"] == "mismatch"
    s.set_request_check(pid, mismatch)
    assert s.plans[pid]["approval"] is None                       # like a review, it resets approval
    with client() as c:
        r = c.post(f"/api/plans/{pid}/approval", json={"confirmed": True})
    assert r.status_code == 409 and "요청 문장과 계획의 조건이 다릅니다" in r.json()["detail"]
    assert "고정 치아에 요청한 13번이 없습니다" in r.json()["detail"]
    with pytest.raises(ValueError, match="요청 문장과"):
        store_module.Store().approve(pid)                         # the plan file read back after a restart
    s.set_request_check(pid, request_check.check(REVISION, Constraints(), C(lock=[6])))
    assert s.approve(pid)["approval"]["status"] == "approved"


@pytest.mark.parametrize("lock, status, code", [([25], "mismatch", 409), ([13], "match", 200)])
def test_every_compared_candidate_is_checked_so_switching_to_it_cannot_skip_the_check(isolated, lock, status, code):
    """Codex review 2026-10-03 (1): compare_strategies stores several plans and select_plan picks one; the screen's
    plan menu switches to another with a GET (no select_plan), the dentist asks for its review by hand and approves.
    Only the selected plan carried the check, so the switch walked around it. Every plan the turn makes carries it.
    lock [25] is the live mistake (FDI 25 for «13번»); [13] the same path applied right, which approval lets through."""
    run = PlanRun("rc", "moderate", None, Constraints())
    run.text, run.opened = REVISION, Constraints()
    compared, selected = run_tools([("set_constraints", {"lock": lock}), ("compare_strategies", {}),
                                    ("select_plan", lambda out: {"plan_id": out[1]["plans"][0]["plan_id"]})], run)[1:]
    assert len(compared["plans"]) >= 2
    assert [p["request_check"]["status"] for p in compared["plans"]] == [status] * len(compared["plans"])
    other = next(p["plan_id"] for p in compared["plans"][1:] if not isolated.plans[p["plan_id"]]["violations"])

    async def review(pid):                                        # the reviewer passes the memo, as the fake one does
        isolated.set_review(pid, REVIEW)

    with client(review) as c:
        assert c.get(f"/api/plans/{other}").json()["request_check"]["status"] == status     # the menu's switch
        assert c.post(f"/api/plans/{other}/review").status_code == 200
        r = c.post(f"/api/plans/{other}/approval", json={"confirmed": True})
    assert r.status_code == code, r.text
    if code == 409:
        assert r.json()["detail"].startswith("요청 문장과 계획의 조건이 다릅니다: 고정 치아에 요청한 13번이 없습니다")
